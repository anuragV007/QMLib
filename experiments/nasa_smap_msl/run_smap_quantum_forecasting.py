"""
NASA SMAP/MSL one-step-ahead telemetry anomaly detection.

Methods:
  1. Persistence forecast (previous telemetry value).
  2. Classical RBF Kernel Ridge Regression.
  3. QMLib Quantum Kernel Ridge Regression with a ZZ fidelity kernel.

The target is feature 0. Model inputs contain the previous `window` rows
of the complete multivariate telemetry + command/context sequence.

Training is chronological:
  - first 75% of training windows: fit models and preprocessing;
  - final 25%: calibrate each model's absolute-residual threshold.
The test sequence is evaluated chronologically and labels are used only
for final evaluation. Its first `window` rows are warm-up history and are
not scored; no assumption is made that train and test are contiguous.

Run from the QMLib project root:
    python experiments/nasa_smap_msl/run_smap_quantum_forecasting.py
"""

import argparse
import ast
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.decomposition import PCA
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from qmlib.feature_maps import ZZFeatureMap
from qmlib.kernels import CachedFidelityKernel
from qmlib.models import QuantumKernelRidge


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "NASA Data"
RESULTS_DIR = ROOT / "experiments" / "nasa_smap_msl" / "results"


def find_data_dirs():
    candidates = [
        (DATA_ROOT / "data" / "data" / "train",
         DATA_ROOT / "data" / "data" / "test"),
        (DATA_ROOT / "data" / "train",
         DATA_ROOT / "data" / "test"),
    ]
    for train_dir, test_dir in candidates:
        if train_dir.is_dir() and test_dir.is_dir():
            return train_dir, test_dir
    raise FileNotFoundError(
        "Could not find NASA Data train/test folders. "
        "Expected NASA Data/data/data/{train,test}."
    )


def parse_intervals(value):
    """Parse zero-based, inclusive anomaly intervals."""
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


def make_labels(n_rows, intervals, channel):
    labels = np.zeros(n_rows, dtype=np.int8)
    for start, end in intervals:
        if end >= n_rows:
            raise ValueError(
                f"{channel}: anomaly interval [{start}, {end}] exceeds "
                f"test length {n_rows}."
            )
        labels[start:end + 1] = 1
    return labels


def build_windows(sequence, target_indices, window):
    """
    For target time t, flatten all features from sequence[t-window:t].
    Predict feature 0 at t. No current/future row is included in the input.
    """
    target_indices = np.asarray(target_indices, dtype=int)
    n_features = sequence.shape[1]

    X = np.empty((len(target_indices), window * n_features), dtype=float)
    y = np.empty(len(target_indices), dtype=float)

    for row, t in enumerate(target_indices):
        X[row] = sequence[t - window:t, :].reshape(-1)
        y[row] = sequence[t, 0]

    return X, y


def angle_encode(X):
    """Bound PCA features to valid rotation-angle inputs."""
    return 2.0 * np.arctan(np.asarray(X, dtype=float))


def evenly_spaced_indices(n_rows, maximum):
    if n_rows <= maximum:
        return np.arange(n_rows, dtype=int)
    return np.linspace(0, n_rows - 1, maximum).round().astype(int)


def batched_predict(model, X, batch_size):
    outputs = []
    for start in range(0, len(X), batch_size):
        outputs.append(model.predict(X[start:start + batch_size]))
    return np.concatenate(outputs)


def get_metrics(y_true, predicted, scores, threshold, intervals, window):
    y_true = np.asarray(y_true, dtype=np.int8)
    predicted = np.asarray(predicted, dtype=np.int8)
    scores = np.asarray(scores, dtype=float)

    metrics = {
        "threshold": float(threshold),
        "test_points_scored": int(len(y_true)),
        "actual_anomaly_points": int(y_true.sum()),
        "predicted_anomaly_points": int(predicted.sum()),
        "accuracy": float(accuracy_score(y_true, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
    }

    # Ranking metrics are undefined when the evaluated sequence has one class.
    if np.unique(y_true).size == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, scores))
        metrics["average_precision"] = float(average_precision_score(y_true, scores))
    else:
        metrics["roc_auc"] = float("nan")
        metrics["average_precision"] = float("nan")

    eligible_events = []
    for start, end in intervals:
        if end < window:
            continue
        eligible_events.append((max(start, window), end))

    if eligible_events:
        detected = 0
        for start, end in eligible_events:
            local_start = start - window
            local_end = end - window + 1
            if np.any(predicted[local_start:local_end]):
                detected += 1
        metrics["eligible_anomaly_events"] = int(len(eligible_events))
        metrics["detected_anomaly_events"] = int(detected)
        metrics["event_recall"] = float(detected / len(eligible_events))
    else:
        metrics["eligible_anomaly_events"] = 0
        metrics["detected_anomaly_events"] = 0
        metrics["event_recall"] = float("nan")

    return metrics


def score_method(name, y_cal, pred_cal, y_test, pred_test,
                 labels_test, intervals, window):
    calibration_scores = np.abs(y_cal - pred_cal)
    threshold = float(np.quantile(calibration_scores, 0.99))
    test_scores = np.abs(y_test - pred_test)
    predicted = (test_scores >= threshold).astype(np.int8)

    metrics = get_metrics(
        labels_test,
        predicted,
        test_scores,
        threshold,
        intervals,
        window,
    )
    metrics.update({
        "method": name,
        "calibration_points": int(len(y_cal)),
        "calibration_threshold_quantile": 0.99,
        "calibration_mae": float(mean_absolute_error(y_cal, pred_cal)),
        "test_forecast_mae": float(mean_absolute_error(y_test, pred_test)),
        "test_forecast_rmse": float(np.sqrt(mean_squared_error(y_test, pred_test))),
    })
    return metrics, test_scores, predicted


def run_channel(channel, args, train_dir, test_dir, metadata):
    train_path = train_dir / (channel + ".npy")
    test_path = test_dir / (channel + ".npy")

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"{channel}: train or test .npy file is missing.")

    train = np.asarray(np.load(train_path, allow_pickle=False), dtype=float)
    test = np.asarray(np.load(test_path, allow_pickle=False), dtype=float)

    if train.ndim != 2 or test.ndim != 2:
        raise ValueError(f"{channel}: train/test arrays must be 2D.")
    if train.shape[1] != test.shape[1]:
        raise ValueError(f"{channel}: train/test feature counts differ.")
    if train.shape[1] < 2:
        raise ValueError(f"{channel}: expected target + command/context features.")
    if not np.isfinite(train).all() or not np.isfinite(test).all():
        raise ValueError(
            f"{channel}: non-finite values found. Resolve them before running "
            "to preserve exact timestep-label alignment."
        )
    if len(train) <= args.window + 10 or len(test) <= args.window:
        raise ValueError(f"{channel}: sequence too short for window={args.window}.")

    channel_rows = metadata[metadata["chan_id"].astype(str) == channel]
    if channel_rows.empty:
        raise ValueError(f"{channel}: no labelled_anomalies.csv metadata found.")

    intervals = []
    for value in channel_rows["anomaly_sequences"]:
        intervals.extend(parse_intervals(value))

    labels = make_labels(len(test), intervals, channel)

    # Chronological fit/calibration split inside the training sequence.
    split = int(0.75 * len(train))
    fit_targets = np.arange(args.window, split)
    calibration_targets = np.arange(split, len(train))

    X_fit_raw, y_fit_all = build_windows(train, fit_targets, args.window)
    X_cal_raw, y_cal = build_windows(train, calibration_targets, args.window)

    # Test windows are constructed only within the test sequence.
    # Its first `window` observations are context warm-up, not scored.
    test_targets = np.arange(args.window, len(test))
    X_test_raw, y_test = build_windows(test, test_targets, args.window)
    labels_test = labels[args.window:]

    if len(X_fit_raw) < 20 or len(X_cal_raw) < 20:
        raise ValueError(f"{channel}: insufficient fit/calibration windows.")

    # Fit all preprocessing strictly on the model-fit portion of training.
    scaler = StandardScaler()
    X_fit_scaled = scaler.fit_transform(X_fit_raw)
    X_cal_scaled = scaler.transform(X_cal_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    pca = PCA(n_components=args.n_qubits, svd_solver="full")
    X_fit_encoded = angle_encode(pca.fit_transform(X_fit_scaled))
    X_cal_encoded = angle_encode(pca.transform(X_cal_scaled))
    X_test_encoded = angle_encode(pca.transform(X_test_scaled))

    # Use the same evenly-spaced training examples for both kernel models.
    selected = evenly_spaced_indices(len(X_fit_encoded), args.max_train_samples)
    X_model_train = X_fit_encoded[selected]
    y_model_train = y_fit_all[selected]

    channel_results = []
    prediction_frames = []

    # Baseline 1: persistence forecast (last observed telemetry value).
    persist_cal = train[calibration_targets - 1, 0]
    persist_test = test[test_targets - 1, 0]
    metrics, scores, predicted = score_method(
        "Persistence",
        y_cal,
        persist_cal,
        y_test,
        persist_test,
        labels_test,
        intervals,
        args.window,
    )
    metrics.update({
        "channel": channel,
        "fit_windows": int(len(X_model_train)),
        "n_qubits": 0,
        "runtime_seconds": 0.0,
        "cache_entries": 0,
        "cache_memory_mb": 0.0,
    })
    channel_results.append(metrics)
    prediction_frames.append(pd.DataFrame({
        "channel": channel,
        "method": "Persistence",
        "test_index": test_targets,
        "y_true": labels_test,
        "actual_telemetry": y_test,
        "predicted_telemetry": persist_test,
        "anomaly_score": scores,
        "threshold": metrics["threshold"],
        "predicted_anomaly": predicted,
    }))

    # Baseline 2: classical RBF Kernel Ridge Regression.
    start = time.perf_counter()
    classical = KernelRidge(alpha=args.alpha, kernel="precomputed")
    gamma = 1.0 / args.n_qubits
    classical_train_kernel = rbf_kernel(
        X_model_train, X_model_train, gamma=gamma
    )
    classical_train_column_mean = classical_train_kernel.mean(
        axis=0, keepdims=True
    )
    classical_train_kernel_mean = float(classical_train_kernel.mean())
    classical_train_kernel_centered = (
        classical_train_kernel
        - classical_train_kernel.mean(axis=1, keepdims=True)
        - classical_train_column_mean
        + classical_train_kernel_mean
    )
    classical_target_mean = float(np.mean(y_model_train))
    classical.fit(
        classical_train_kernel_centered,
        y_model_train - classical_target_mean,
    )

    def classical_predict(X_new):
        cross = rbf_kernel(X_new, X_model_train, gamma=gamma)
        cross_centered = (
            cross
            - cross.mean(axis=1, keepdims=True)
            - classical_train_column_mean
            + classical_train_kernel_mean
        )
        return classical.predict(cross_centered) + classical_target_mean

    classical_cal = classical_predict(X_cal_encoded)
    classical_test = classical_predict(X_test_encoded)
    classical_runtime = time.perf_counter() - start

    metrics, scores, predicted = score_method(
        "Classical_RBF_KRR",
        y_cal,
        classical_cal,
        y_test,
        classical_test,
        labels_test,
        intervals,
        args.window,
    )
    metrics.update({
        "channel": channel,
        "fit_windows": int(len(X_model_train)),
        "n_qubits": 0,
        "runtime_seconds": float(classical_runtime),
        "cache_entries": 0,
        "cache_memory_mb": 0.0,
    })
    channel_results.append(metrics)
    prediction_frames.append(pd.DataFrame({
        "channel": channel,
        "method": "Classical_RBF_KRR",
        "test_index": test_targets,
        "y_true": labels_test,
        "actual_telemetry": y_test,
        "predicted_telemetry": classical_test,
        "anomaly_score": scores,
        "threshold": metrics["threshold"],
        "predicted_anomaly": predicted,
    }))

    # Quantum model: QMLib ZZ feature map + cached fidelity kernel.
    start = time.perf_counter()
    feature_map = ZZFeatureMap(
        n_qubits=args.n_qubits,
        reps=args.reps,
    )
    quantum_kernel = CachedFidelityKernel(feature_map=feature_map)
    quantum = QuantumKernelRidge(
        kernel=quantum_kernel,
        alpha=args.alpha,
    )
    quantum.fit(X_model_train, y_model_train)
    quantum_cal = batched_predict(
        quantum, X_cal_encoded, args.batch_size
    )
    quantum_test = batched_predict(
        quantum, X_test_encoded, args.batch_size
    )
    quantum_runtime = time.perf_counter() - start

    cache = getattr(quantum_kernel, "_state_cache", {})
    cache_entries = len(cache)
    cache_bytes = sum(
        np.asarray(state).nbytes for state in cache.values()
    )

    metrics, scores, predicted = score_method(
        "Quantum_ZZ_KRR",
        y_cal,
        quantum_cal,
        y_test,
        quantum_test,
        labels_test,
        intervals,
        args.window,
    )
    metrics.update({
        "channel": channel,
        "fit_windows": int(len(X_model_train)),
        "n_qubits": int(args.n_qubits),
        "runtime_seconds": float(quantum_runtime),
        "cache_entries": int(cache_entries),
        "cache_memory_mb": float(cache_bytes / (1024.0 ** 2)),
    })
    channel_results.append(metrics)
    prediction_frames.append(pd.DataFrame({
        "channel": channel,
        "method": "Quantum_ZZ_KRR",
        "test_index": test_targets,
        "y_true": labels_test,
        "actual_telemetry": y_test,
        "predicted_telemetry": quantum_test,
        "anomaly_score": scores,
        "threshold": metrics["threshold"],
        "predicted_anomaly": predicted,
    }))

    # Release quantum statevectors after this channel.
    quantum_kernel.clear_cache()

    print(f"\n{channel}: {len(train)} train rows, {len(test)} test rows")
    print(
        f"  scored test points={len(test_targets)}, "
        f"anomaly points={int(labels_test.sum())}, "
        f"events={len(intervals)}"
    )
    for row in channel_results:
        print(
            "  {method:20s} F1={f1:.4f} P={precision:.4f} "
            "R={recall:.4f} PR-AUC={average_precision:.4f} "
            "time={runtime_seconds:.2f}s".format(**row)
        )

    return channel_results, prediction_frames, {
        "channel": channel,
        "train_shape": list(train.shape),
        "test_shape": list(test.shape),
        "n_features": int(train.shape[1]),
        "window_size": int(args.window),
        "fit_window_count": int(len(X_fit_raw)),
        "calibration_window_count": int(len(X_cal_raw)),
        "quantum_training_samples": int(len(X_model_train)),
        "n_qubits": int(args.n_qubits),
        "pca_explained_variance_ratio_sum": float(
            pca.explained_variance_ratio_.sum()
        ),
        "anomaly_intervals": intervals,
        "test_anomaly_prevalence_full": float(labels.mean()),
        "test_anomaly_prevalence_scored": float(labels_test.mean()),
        "warmup_test_points_not_scored": int(args.window),
        "quantum_cache_entries_before_clear": int(cache_entries),
        "quantum_cache_memory_mb": float(cache_bytes / (1024.0 ** 2)),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Classical and quantum kernel forecasting for NASA SMAP/MSL."
    )
    parser.add_argument(
        "--channels",
        default="P-1",
        help="Comma-separated labelled channels, e.g. P-1,S-1.",
    )
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--n-qubits", type=int, default=4)
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--max-train-samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--alpha", type=float, default=1e-2)
    parser.add_argument("--run-name", default="smap_quantum_forecasting_P1")
    args = parser.parse_args()

    if args.window < 1:
        parser.error("--window must be at least 1.")
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

    channels = [item.strip() for item in args.channels.split(",") if item.strip()]
    if not channels:
        raise ValueError("Provide at least one channel using --channels.")

    all_metrics = []
    all_predictions = []
    channel_configs = []

    print("=" * 78)
    print("NASA SMAP/MSL: CLASSICAL VS QUANTUM KERNEL FORECASTING")
    print("=" * 78)
    print("Channels:", ", ".join(channels))
    print("Window:", args.window)
    print("Qubits:", args.n_qubits)
    print("Training windows used per model:", args.max_train_samples)
    print("Calibration threshold: 99th percentile of training residuals")
    print("Test labels are used only for evaluation.")

    for channel in channels:
        metrics, predictions, config = run_channel(
            channel, args, train_dir, test_dir, metadata
        )
        all_metrics.extend(metrics)
        all_predictions.extend(predictions)
        channel_configs.append(config)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = RESULTS_DIR / f"{args.run_name}_metrics.csv"
    predictions_path = RESULTS_DIR / f"{args.run_name}_predictions.csv"
    config_path = RESULTS_DIR / f"{args.run_name}_config.json"

    metrics_df = pd.DataFrame(all_metrics)
    predictions_df = pd.concat(all_predictions, ignore_index=True)

    metrics_df.to_csv(metrics_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)

    config = {
        "experiment": "one-step-ahead telemetry anomaly detection",
        "target_feature": 0,
        "input_features": "previous multivariate telemetry + command/context window",
        "fit_calibration_split": "first 75% / final 25% of training windows",
        "test_evaluation": "chronological; first window rows used as warm-up",
        "threshold": "99th percentile of absolute calibration residuals",
        "methods": [
            "Persistence",
            "Classical_RBF_KRR",
            "Quantum_ZZ_KRR",
        ],
        "window": args.window,
        "n_qubits": args.n_qubits,
        "reps": args.reps,
        "max_train_samples": args.max_train_samples,
        "batch_size": args.batch_size,
        "alpha": args.alpha,
        "channels": channel_configs,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scikit_learn_version": sklearn.__version__,
        "important_dataset_caveat": (
            "The original Telemanom data README states telemetry values "
            "were pre-scaled using test-set min/max values. This inherited "
            "preprocessing limitation is disclosed; this experiment's "
            "StandardScaler and PCA are fitted on the model-fit training "
            "windows only."
        ),
        "interpretation_caveat": (
            "This is a simplified one-step-ahead quantum-kernel forecasting "
            "experiment inspired by Telemanom. It does not reproduce the "
            "original LSTM architecture or its dynamic thresholding method."
        ),
    }
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    print("\nSaved:")
    print(metrics_path)
    print(predictions_path)
    print(config_path)

    print("\nMETRICS")
    print(
        metrics_df[
            [
                "channel", "method", "precision", "recall", "f1",
                "roc_auc", "average_precision", "event_recall",
                "test_forecast_mae", "runtime_seconds",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
