"""
NASA SMAP/MSL quantum forecasting: threshold sensitivity and interval coverage.

Compares:
  1. Persistence forecast
  2. Classical RBF Kernel Ridge Regression
  3. QMLib Quantum Kernel Ridge Regression (ZZ fidelity kernel)

Runs all requested channel/window combinations. Each forecaster is fitted once;
calibration residuals and test scores are then evaluated at multiple thresholds.

Example:
    python experiments/nasa_smap_msl/run_smap_quantum_forecasting_v2.py

Default run:
    channels = P-1,S-1
    windows = 10,25
    thresholds = 0.95,0.975,0.99

Test labels are used only for reporting metrics. They are not used for fitting,
preprocessing, selecting thresholds, or choosing model hyperparameters.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.decomposition import PCA
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import StandardScaler

from qmlib.feature_maps import ZZFeatureMap
from qmlib.kernels import CachedFidelityKernel
from qmlib.models import QuantumKernelRidge


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "NASA Data"
RESULTS_DIR = PROJECT_ROOT / "experiments" / "nasa_smap_msl" / "results"


def find_data_dirs() -> tuple[Path, Path]:
    """Find the train/test directories in supported Kaggle archive layouts."""
    candidates = [
        (
            DATA_ROOT / "data" / "data" / "train",
            DATA_ROOT / "data" / "data" / "test",
        ),
        (DATA_ROOT / "data" / "train", DATA_ROOT / "data" / "test"),
    ]
    for train_dir, test_dir in candidates:
        if train_dir.is_dir() and test_dir.is_dir():
            return train_dir, test_dir

    raise FileNotFoundError(
        "Could not find NASA Data train/test folders. Expected "
        "NASA Data/data/data/{train,test}."
    )


def parse_intervals(value) -> list[tuple[int, int]]:
    """Parse zero-based inclusive anomaly intervals."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []

    parsed = ast.literal_eval(value) if isinstance(value, str) else value
    if not isinstance(parsed, (list, tuple)):
        raise ValueError(f"Invalid anomaly_sequences value: {value!r}")

    if len(parsed) == 2 and all(np.isscalar(item) for item in parsed):
        parsed = [parsed]

    intervals = []
    for pair in parsed:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(f"Invalid anomaly interval: {pair!r}")
        start, end = int(pair[0]), int(pair[1])
        if start < 0 or end < start:
            raise ValueError(f"Invalid anomaly interval: {pair!r}")
        intervals.append((start, end))
    return intervals


def parse_anomaly_classes(value) -> list[str]:
    """
    Extract class tokens from CSV strings such as:
      [point], [contextual, contextual], [point, contextual].
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    return re.findall(r"\b(point|contextual)\b", str(value).lower())


def collect_intervals(metadata_rows: pd.DataFrame) -> list[dict]:
    """
    Keep each interval aligned with its class, then merge overlapping
    duplicate intervals (e.g. repeated metadata rows for one channel).
    """
    records = []

    for _, row in metadata_rows.iterrows():
        intervals = parse_intervals(row["anomaly_sequences"])
        classes = parse_anomaly_classes(row.get("class"))

        if len(classes) == len(intervals):
            interval_classes = classes
        else:
            interval_classes = ["unspecified"] * len(intervals)

        for (start, end), anomaly_class in zip(intervals, interval_classes):
            records.append({
                "start": start,
                "end": end,
                "class": anomaly_class,
            })

    records.sort(key=lambda item: (item["start"], item["end"]))
    merged: list[dict] = []

    for item in records:
        if merged and item["start"] <= merged[-1]["end"]:
            merged[-1]["end"] = max(merged[-1]["end"], item["end"])
            classes = set(merged[-1]["class"].split("|"))
            classes.add(item["class"])
            merged[-1]["class"] = "|".join(sorted(classes))
        else:
            merged.append(item.copy())

    return merged


def make_labels(n_rows: int, intervals: list[dict], channel: str) -> np.ndarray:
    """Build timestep labels; anomaly intervals have inclusive endpoints."""
    labels = np.zeros(n_rows, dtype=np.int8)
    for interval in intervals:
        start, end = interval["start"], interval["end"]
        if end >= n_rows:
            raise ValueError(
                f"{channel}: anomaly interval [{start}, {end}] exceeds "
                f"the test length {n_rows}."
            )
        labels[start:end + 1] = 1
    return labels


def build_windows(
    sequence: np.ndarray,
    target_indices: np.ndarray,
    window: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Flatten the previous `window` complete multivariate rows.

    Input for target time t:
        sequence[t-window:t, :]

    Target:
        sequence[t, 0] (current telemetry value)
    """
    target_indices = np.asarray(target_indices, dtype=int)
    n_features = sequence.shape[1]

    X = np.empty((len(target_indices), window * n_features), dtype=float)
    y = np.empty(len(target_indices), dtype=float)

    for row, t in enumerate(target_indices):
        X[row] = sequence[t - window:t, :].reshape(-1)
        y[row] = sequence[t, 0]

    return X, y


def angle_encode(X: np.ndarray) -> np.ndarray:
    """Map real PCA features to bounded rotation angles."""
    return 2.0 * np.arctan(np.asarray(X, dtype=float))


def evenly_spaced_indices(n_rows: int, maximum: int) -> np.ndarray:
    """Subsample chronological training windows without using test labels."""
    if n_rows <= maximum:
        return np.arange(n_rows, dtype=int)
    return np.linspace(0, n_rows - 1, maximum).round().astype(int)


def batched_predict(model, X: np.ndarray, batch_size: int) -> np.ndarray:
    """Predict quantum-kernel outputs in manageable batches."""
    predictions = []
    for start in range(0, len(X), batch_size):
        predictions.append(model.predict(X[start:start + batch_size]))
    return np.concatenate(predictions)


def quantile_tag(quantile: float) -> str:
    """Create a stable CSV column suffix, e.g. 0.975 -> q975."""
    return f"q{int(round(quantile * 1000)):03d}"


def evaluate_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    quantile: float,
    intervals: list[dict],
    window: int,
    calibration_scores: np.ndarray,
) -> tuple[dict, np.ndarray, list[dict]]:
    """Calculate point metrics and per-interval coverage for one threshold."""
    predicted = (scores >= threshold).astype(np.int8)
    matrix = confusion_matrix(y_true, predicted, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()

    if np.unique(y_true).size == 2:
        roc_auc = float(roc_auc_score(y_true, scores))
        average_precision = float(average_precision_score(y_true, scores))
    else:
        roc_auc = float("nan")
        average_precision = float("nan")

    interval_rows = []
    coverages = []

    for event_number, interval in enumerate(intervals, start=1):
        # Test indices before `window` are warm-up and are not scored.
        start = max(int(interval["start"]), window)
        end = int(interval["end"])

        if end < start:
            continue

        local_start = start - window
        local_end_exclusive = end - window + 1
        event_predictions = predicted[local_start:local_end_exclusive]
        event_length = len(event_predictions)

        if event_length == 0:
            continue

        predicted_inside = int(event_predictions.sum())
        coverage = float(predicted_inside / event_length)
        coverages.append(coverage)

        interval_rows.append({
            "event_number": event_number,
            "event_start_test_index": int(interval["start"]),
            "event_end_test_index_inclusive": int(interval["end"]),
            "event_class": interval["class"],
            "scored_event_start": int(start),
            "scored_event_end_inclusive": int(end),
            "event_points_scored": int(event_length),
            "event_points_flagged": predicted_inside,
            "interval_coverage": coverage,
            "threshold_quantile": float(quantile),
            "threshold": float(threshold),
        })

    if coverages:
        mean_coverage = float(np.mean(coverages))
        median_coverage = float(np.median(coverages))
        min_coverage = float(np.min(coverages))
        coverage_25_rate = float(np.mean(np.asarray(coverages) >= 0.25))
        coverage_50_rate = float(np.mean(np.asarray(coverages) >= 0.50))
    else:
        mean_coverage = median_coverage = min_coverage = float("nan")
        coverage_25_rate = coverage_50_rate = float("nan")

    metrics = {
        "threshold_quantile": float(quantile),
        "threshold": float(threshold),
        "calibration_exceedance_rate": float(
            np.mean(calibration_scores >= threshold)
        ),
        "test_points_scored": int(len(y_true)),
        "actual_anomaly_points": int(np.sum(y_true)),
        "actual_anomaly_prevalence": float(np.mean(y_true)),
        "predicted_anomaly_points": int(np.sum(predicted)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "accuracy": float(accuracy_score(y_true, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "roc_auc": roc_auc,
        "average_precision": average_precision,
        "eligible_intervals": int(len(coverages)),
        "mean_interval_coverage": mean_coverage,
        "median_interval_coverage": median_coverage,
        "minimum_interval_coverage": min_coverage,
        "intervals_with_at_least_25pct_coverage": coverage_25_rate,
        "intervals_with_at_least_50pct_coverage": coverage_50_rate,
    }

    return metrics, predicted, interval_rows


def fit_and_score_channel(
    channel: str,
    window: int,
    args,
    train_dir: Path,
    test_dir: Path,
    metadata: pd.DataFrame,
) -> tuple[list[dict], list[pd.DataFrame], list[dict], dict]:
    """Fit each forecaster once, then evaluate several calibration thresholds."""
    train_path = train_dir / f"{channel}.npy"
    test_path = test_dir / f"{channel}.npy"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"{channel}: train/test arrays are missing.")

    train = np.asarray(np.load(train_path, allow_pickle=False), dtype=float)
    test = np.asarray(np.load(test_path, allow_pickle=False), dtype=float)

    if train.ndim != 2 or test.ndim != 2:
        raise ValueError(f"{channel}: train/test arrays must be 2D.")
    if train.shape[1] != test.shape[1] or train.shape[1] < 2:
        raise ValueError(f"{channel}: incompatible train/test feature dimensions.")
    if not np.isfinite(train).all() or not np.isfinite(test).all():
        raise ValueError(
            f"{channel}: non-finite values found. Resolve them before this "
            "experiment to preserve timestep alignment."
        )
    if len(train) <= window + 20 or len(test) <= window:
        raise ValueError(f"{channel}: sequence too short for window={window}.")

    channel_rows = metadata[metadata["chan_id"].astype(str) == channel]
    if channel_rows.empty:
        raise ValueError(f"{channel}: no labelled_anomalies.csv metadata.")
    intervals = collect_intervals(channel_rows)
    labels = make_labels(len(test), intervals, channel)

    # Chronological model-fit/calibration split within training only.
    split = int(0.75 * len(train))
    fit_targets = np.arange(window, split)
    calibration_targets = np.arange(split, len(train))

    X_fit_raw, y_fit_all = build_windows(train, fit_targets, window)
    X_cal_raw, y_cal = build_windows(train, calibration_targets, window)

    # Complete test sequence is evaluated in order; initial rows provide
    # context only and are excluded from scoring.
    test_targets = np.arange(window, len(test))
    X_test_raw, y_test = build_windows(test, test_targets, window)
    labels_test = labels[window:]

    if len(X_fit_raw) <= args.n_qubits or len(X_cal_raw) < 10:
        raise ValueError(f"{channel}: insufficient fit/calibration windows.")

    # Preprocessing is fitted only on the first 75% of training windows.
    scaler = StandardScaler()
    X_fit_scaled = scaler.fit_transform(X_fit_raw)
    X_cal_scaled = scaler.transform(X_cal_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    pca = PCA(n_components=args.n_qubits, svd_solver="full")
    X_fit_encoded = angle_encode(pca.fit_transform(X_fit_scaled))
    X_cal_encoded = angle_encode(pca.transform(X_cal_scaled))
    X_test_encoded = angle_encode(pca.transform(X_test_scaled))

    selected = evenly_spaced_indices(
        len(X_fit_encoded), args.max_train_samples
    )
    X_model_train = X_fit_encoded[selected]
    y_model_train = y_fit_all[selected]

    # --- Method 1: persistence baseline ---
    persistence_cal = train[calibration_targets - 1, 0]
    persistence_test = test[test_targets - 1, 0]

    # --- Method 2: classical RBF kernel ridge regression ---
    classical_start = time.perf_counter()
    gamma = 1.0 / args.n_qubits
    classical_train_gram = rbf_kernel(
        X_model_train, X_model_train, gamma=gamma
    )
    classical_col_mean = classical_train_gram.mean(axis=0, keepdims=True)
    classical_grand_mean = float(classical_train_gram.mean())
    classical_train_centered = (
        classical_train_gram
        - classical_train_gram.mean(axis=1, keepdims=True)
        - classical_col_mean
        + classical_grand_mean
    )
    classical_y_mean = float(np.mean(y_model_train))

    classical = KernelRidge(alpha=args.alpha, kernel="precomputed")
    classical.fit(
        classical_train_centered,
        y_model_train - classical_y_mean,
    )

    def classical_predict(X_new):
        cross = rbf_kernel(X_new, X_model_train, gamma=gamma)
        cross_centered = (
            cross
            - cross.mean(axis=1, keepdims=True)
            - classical_col_mean
            + classical_grand_mean
        )
        return classical.predict(cross_centered) + classical_y_mean

    classical_cal = classical_predict(X_cal_encoded)
    classical_test = classical_predict(X_test_encoded)
    classical_runtime = time.perf_counter() - classical_start

    # --- Method 3: quantum ZZ fidelity-kernel ridge regression ---
    quantum_start = time.perf_counter()
    feature_map = ZZFeatureMap(n_qubits=args.n_qubits, reps=args.reps)
    quantum_kernel = CachedFidelityKernel(feature_map=feature_map)
    quantum = QuantumKernelRidge(kernel=quantum_kernel, alpha=args.alpha)
    quantum.fit(X_model_train, y_model_train)

    quantum_cal = batched_predict(
        quantum, X_cal_encoded, args.batch_size
    )
    quantum_test = batched_predict(
        quantum, X_test_encoded, args.batch_size
    )
    quantum_runtime = time.perf_counter() - quantum_start

    cache = getattr(quantum_kernel, "_state_cache", {})
    cache_entries = len(cache)
    cache_bytes = sum(
        np.asarray(state).nbytes for state in cache.values()
    )
    cache_memory_mb = float(cache_bytes / (1024.0 ** 2))
    quantum_kernel.clear_cache()

    methods = [
        {
            "name": "Persistence",
            "calibration_actual": y_cal,
            "calibration_prediction": persistence_cal,
            "test_actual": y_test,
            "test_prediction": persistence_test,
            "runtime_seconds": 0.0,
            "forecast_model": "previous telemetry value",
            "n_qubits": 0,
            "cache_entries": 0,
            "cache_memory_mb": 0.0,
        },
        {
            "name": "Classical_RBF_KRR",
            "calibration_actual": y_cal,
            "calibration_prediction": classical_cal,
            "test_actual": y_test,
            "test_prediction": classical_test,
            "runtime_seconds": float(classical_runtime),
            "forecast_model": "RBF KernelRidge",
            "n_qubits": 0,
            "cache_entries": 0,
            "cache_memory_mb": 0.0,
        },
        {
            "name": "Quantum_ZZ_KRR",
            "calibration_actual": y_cal,
            "calibration_prediction": quantum_cal,
            "test_actual": y_test,
            "test_prediction": quantum_test,
            "runtime_seconds": float(quantum_runtime),
            "forecast_model": "QMLib QuantumKernelRidge + ZZ fidelity kernel",
            "n_qubits": int(args.n_qubits),
            "cache_entries": int(cache_entries),
            "cache_memory_mb": cache_memory_mb,
        },
    ]

    metric_rows = []
    prediction_frames = []
    interval_rows_all = []

    for method in methods:
        calibration_scores = np.abs(
            method["calibration_actual"] - method["calibration_prediction"]
        )
        test_scores = np.abs(
            method["test_actual"] - method["test_prediction"]
        )

        prediction_row = pd.DataFrame({
            "channel": channel,
            "window": int(window),
            "method": method["name"],
            "test_index": test_targets,
            "y_true": labels_test,
            "actual_telemetry": method["test_actual"],
            "predicted_telemetry": method["test_prediction"],
            "anomaly_score": test_scores,
        })

        for quantile in args.threshold_quantiles:
            threshold = float(np.quantile(calibration_scores, quantile))
            metrics, predicted, event_rows = evaluate_threshold(
                y_true=labels_test,
                scores=test_scores,
                threshold=threshold,
                quantile=quantile,
                intervals=intervals,
                window=window,
                calibration_scores=calibration_scores,
            )
            metrics.update({
                "channel": channel,
                "window": int(window),
                "method": method["name"],
                "calibration_points": int(len(calibration_scores)),
                "calibration_mae": float(mean_absolute_error(
                    method["calibration_actual"],
                    method["calibration_prediction"],
                )),
                "test_forecast_mae": float(mean_absolute_error(
                    method["test_actual"],
                    method["test_prediction"],
                )),
                "test_forecast_rmse": float(np.sqrt(mean_squared_error(
                    method["test_actual"],
                    method["test_prediction"],
                ))),
                "runtime_seconds": method["runtime_seconds"],
                "n_qubits": method["n_qubits"],
                "cache_entries": method["cache_entries"],
                "cache_memory_mb": method["cache_memory_mb"],
            })
            metric_rows.append(metrics)

            tag = quantile_tag(quantile)
            prediction_row[f"threshold_{tag}"] = threshold
            prediction_row[f"predicted_anomaly_{tag}"] = predicted

            for event_row in event_rows:
                interval_rows_all.append({
                    "channel": channel,
                    "window": int(window),
                    "method": method["name"],
                    "threshold_quantile": float(quantile),
                    **event_row,
                })

        prediction_frames.append(prediction_row)

    config = {
        "channel": channel,
        "window": int(window),
        "train_shape": list(train.shape),
        "test_shape": list(test.shape),
        "feature_count": int(train.shape[1]),
        "fit_training_windows": int(len(X_fit_raw)),
        "calibration_windows": int(len(X_cal_raw)),
        "quantum_training_samples": int(len(X_model_train)),
        "n_qubits": int(args.n_qubits),
        "reps": int(args.reps),
        "alpha": float(args.alpha),
        "classical_rbf_gamma": float(gamma),
        "pca_explained_variance_ratio_sum": float(
            pca.explained_variance_ratio_.sum()
        ),
        "threshold_quantiles": [float(q) for q in args.threshold_quantiles],
        "threshold_calibration": (
            "Absolute calibration residuals from the final 25% of training "
            "windows; each requested quantile is evaluated without refitting."
        ),
        "anomaly_intervals": intervals,
        "test_anomaly_prevalence_full": float(labels.mean()),
        "test_anomaly_prevalence_scored": float(labels_test.mean()),
        "test_warmup_points_not_scored": int(window),
        "quantum_runtime_seconds": float(quantum_runtime),
        "quantum_cache_entries_before_clear": int(cache_entries),
        "quantum_cache_memory_mb": cache_memory_mb,
    }

    print(
        f"\n{channel} | window={window} | fit={len(X_fit_raw)} "
        f"| calibration={len(X_cal_raw)} | test scored={len(labels_test)} "
        f"| anomaly points={int(labels_test.sum())} "
        f"| intervals={len(intervals)}"
    )
    for row in metric_rows:
        print(
            f"  {row['method']:18s} q={row['threshold_quantile']:.3f} "
            f"F1={row['f1']:.4f} P={row['precision']:.4f} "
            f"R={row['recall']:.4f} AP={row['average_precision']:.4f} "
            f"mean interval coverage={row['mean_interval_coverage']:.3f}"
        )

    return metric_rows, prediction_frames, interval_rows_all, config


def parse_csv_strings(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_csv_ints(value: str) -> list[int]:
    try:
        output = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected comma-separated integers.") from exc
    if not output or any(item < 1 for item in output):
        raise argparse.ArgumentTypeError("Window values must be positive integers.")
    return output


def parse_csv_quantiles(value: str) -> list[float]:
    try:
        output = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Expected comma-separated quantiles, e.g. 0.95,0.975,0.99."
        ) from exc
    if not output or any(not 0.0 < item < 1.0 for item in output):
        raise argparse.ArgumentTypeError("Quantiles must be strictly between 0 and 1.")
    if len(set(output)) != len(output):
        raise argparse.ArgumentTypeError("Threshold quantiles must be unique.")
    return sorted(output)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare persistence, classical RBF KRR and quantum ZZ KRR "
            "at several training-calibrated anomaly thresholds."
        )
    )
    parser.add_argument("--channels", default="P-1,S-1")
    parser.add_argument("--windows", default="10,25")
    parser.add_argument("--threshold-quantiles", default="0.95,0.975,0.99")
    parser.add_argument("--n-qubits", type=int, default=4)
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--max-train-samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--alpha", type=float, default=1e-2)
    parser.add_argument(
        "--run-name",
        default="smap_forecasting_threshold_sensitivity",
    )
    args = parser.parse_args()

    args.channels = parse_csv_strings(args.channels)
    args.windows = parse_csv_ints(args.windows)
    args.threshold_quantiles = parse_csv_quantiles(args.threshold_quantiles)

    if not args.channels:
        parser.error("--channels must contain at least one channel.")
    if not 2 <= args.n_qubits <= 8:
        parser.error("--n-qubits must be between 2 and 8 for this statevector pilot.")
    if args.reps < 1:
        parser.error("--reps must be at least 1.")
    if args.max_train_samples < 2:
        parser.error("--max-train-samples must be at least 2.")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1.")
    if args.alpha <= 0:
        parser.error("--alpha must be greater than 0.")

    return args


def main():
    args = parse_args()
    train_dir, test_dir = find_data_dirs()

    metadata_path = DATA_ROOT / "labeled_anomalies.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)
    metadata = pd.read_csv(metadata_path)

    required = {"chan_id", "anomaly_sequences"}
    if not required.issubset(metadata.columns):
        raise ValueError(
            f"Metadata must contain columns {sorted(required)}."
        )

    all_metrics = []
    all_predictions = []
    all_intervals = []
    all_configs = []

    print("=" * 86)
    print("NASA SMAP/MSL: THRESHOLD SENSITIVITY EXPERIMENT")
    print("=" * 86)
    print("Channels:", ", ".join(args.channels))
    print("Windows:", ", ".join(map(str, args.windows)))
    print("Threshold quantiles:", ", ".join(map(str, args.threshold_quantiles)))
    print("Qubits:", args.n_qubits)
    print("Quantum training samples:", args.max_train_samples)
    print("Each model is fitted once per channel/window.")
    print("Test labels are used only for evaluation.")

    for channel in args.channels:
        for window in args.windows:
            metric_rows, prediction_frames, interval_rows, config = (
                fit_and_score_channel(
                    channel=channel,
                    window=window,
                    args=args,
                    train_dir=train_dir,
                    test_dir=test_dir,
                    metadata=metadata,
                )
            )
            all_metrics.extend(metric_rows)
            all_predictions.extend(prediction_frames)
            all_intervals.extend(interval_rows)
            all_configs.append(config)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = RESULTS_DIR / f"{args.run_name}_metrics.csv"
    predictions_path = RESULTS_DIR / f"{args.run_name}_predictions.csv"
    intervals_path = RESULTS_DIR / f"{args.run_name}_interval_coverage.csv"
    config_path = RESULTS_DIR / f"{args.run_name}_config.json"

    metrics_df = pd.DataFrame(all_metrics)
    predictions_df = pd.concat(all_predictions, ignore_index=True)
    intervals_df = pd.DataFrame(all_intervals)

    metrics_df.to_csv(metrics_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)
    intervals_df.to_csv(intervals_path, index=False)

    config = {
        "experiment": "one-step-ahead telemetry anomaly detection",
        "methods": [
            "Persistence",
            "Classical_RBF_KRR",
            "Quantum_ZZ_KRR",
        ],
        "channels": args.channels,
        "windows": args.windows,
        "threshold_quantiles": args.threshold_quantiles,
        "threshold_rule": (
            "Per-method absolute residual quantile from the final 25% "
            "of training windows; thresholds are evaluated without "
            "refitting models."
        ),
        "window_input": (
            "Flattened complete multivariate telemetry and command/context "
            "rows from the preceding window timesteps."
        ),
        "target": "Feature 0 at the current timestep.",
        "preprocessing": (
            "StandardScaler and PCA are fitted on the first 75% of "
            "chronological training windows only."
        ),
        "test_evaluation": (
            "All test timesteps from index window onward, in original order. "
            "The first window points are history warm-up and are not scored."
        ),
        "event_metric": (
            "For each labelled interval, coverage is the fraction of its "
            "scored points predicted anomalous. Reported summary is the "
            "macro-average across intervals; no any-hit event-recall metric."
        ),
        "n_qubits": args.n_qubits,
        "reps": args.reps,
        "max_train_samples": args.max_train_samples,
        "batch_size": args.batch_size,
        "alpha": args.alpha,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scikit_learn_version": sklearn.__version__,
        "dataset_caveat": (
            "The distributed Telemanom arrays were pre-scaled using "
            "test-set ranges upstream. This inherited limitation is "
            "disclosed; the experiment's additional scaler and PCA are "
            "fitted on the model-fit training partition only."
        ),
        "scope_caveat": (
            "This is a simplified one-step-ahead kernel forecasting "
            "experiment inspired by Telemanom, not a reproduction of "
            "the original LSTM and nonparametric dynamic thresholding."
        ),
        "runs": all_configs,
        "outputs": {
            "metrics": str(metrics_path),
            "predictions": str(predictions_path),
            "interval_coverage": str(intervals_path),
        },
    }
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    print("\nSaved:")
    print(metrics_path)
    print(predictions_path)
    print(intervals_path)
    print(config_path)

    display_columns = [
        "channel",
        "window",
        "method",
        "threshold_quantile",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "average_precision",
        "mean_interval_coverage",
        "test_forecast_mae",
        "runtime_seconds",
    ]
    print("\nFINAL METRICS")
    print(metrics_df[display_columns].to_string(index=False))


if __name__ == "__main__":
    main()
