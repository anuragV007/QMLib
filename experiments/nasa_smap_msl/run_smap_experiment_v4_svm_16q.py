"""
NASA SMAP v4: Classical RBF One-Class SVM vs Quantum One-Class SVM.

This experiment is separate from the historical v1-v3 scripts.

Methodology
-----------
* Each labelled SMAP channel is processed independently.
* Training is assumed to be predominantly normal.
* The first 65% of the training sequence is used to fit preprocessing
  and provide normal reference candidates.
* The 65%-75% interval is left as a temporal gap.
* The final 25% of the training sequence supplies normal calibration data.
* PCA is fitted only on the early training partition.
* The same encoded reference, calibration and test samples are used by
  the classical and quantum models.
* Test observations are sampled in a class-stratified way per channel:
  up to N test normals and N test anomalies.
* Each method chooses its own threshold from normal calibration scores.
* This is a bounded, balanced evaluation subset, not a full-prevalence
  operational benchmark.

Run a small pilot first. Example:
python experiments/nasa_smap_msl/run_smap_experiment_v4_svm_16q.py ^
    --channels A-1 ^
    --reference-size 16 ^
    --calibration-size 32 ^
    --max-test-per-class 16 ^
    --run-name smap_experiment_v4_pilot

After validating runtime and memory, run more channels and larger samples.
"""

import argparse
import ast
import gc
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pennylane as qml
import sklearn
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from qmlib.feature_maps import ZFeatureMap, ZZFeatureMap
from qmlib.kernels import CachedFidelityKernel
from qmlib.models import QuantumOneClassSVM


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "NASA Data"
TRAIN_DIR = DATA_ROOT / "data" / "data" / "train"
TEST_DIR = DATA_ROOT / "data" / "data" / "test"
METADATA_PATH = DATA_ROOT / "labeled_anomalies.csv"
RESULTS_DIR = (
    PROJECT_ROOT / "experiments" / "nasa_smap_msl" / "results"
)


# ---------------------------------------------------------------------
# Data and sampling helpers
# ---------------------------------------------------------------------

def parse_intervals(value):
    """Parse zero-based, inclusive anomaly intervals from metadata."""
    if value is None:
        return []

    if isinstance(value, float) and np.isnan(value):
        return []

    parsed = ast.literal_eval(value) if isinstance(value, str) else value

    if not isinstance(parsed, (list, tuple)):
        raise ValueError(
            "Anomaly interval metadata must contain a list or tuple."
        )

    # Support one interval written as [start, end], as well as
    # multiple intervals written as [[start, end], ...].
    if (
        len(parsed) == 2
        and all(np.isscalar(item) for item in parsed)
    ):
        parsed = [parsed]

    intervals = []

    for interval in parsed:
        if not isinstance(interval, (list, tuple)) or len(interval) != 2:
            raise ValueError(f"Invalid anomaly interval: {interval!r}")

        start, end = interval

        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, (int, np.integer))
            or not isinstance(end, (int, np.integer))
        ):
            raise ValueError(f"Interval endpoints must be integers: {interval!r}")

        start, end = int(start), int(end)

        if start < 0 or end < start:
            raise ValueError(f"Invalid interval bounds: {interval!r}")

        intervals.append((start, end))

    return intervals


def make_labels(n_rows, intervals, channel):
    """Create labels using zero-based inclusive interval endpoints."""
    labels = np.zeros(n_rows, dtype=np.int8)

    for start, end in intervals:
        if start >= n_rows or end >= n_rows:
            raise ValueError(
                f"{channel}: anomaly interval [{start}, {end}] exceeds "
                f"test sequence length {n_rows}."
            )

        labels[start:end + 1] = 1

    return labels


def sample_rows(X, maximum, rng):
    """Sample up to maximum rows without replacement."""
    if len(X) <= maximum:
        return X.copy()

    indices = rng.choice(len(X), size=maximum, replace=False)
    return X[indices]


def stratified_test_indices(y, maximum_per_class, rng):
    """
    Select up to maximum_per_class normal and anomalous observations.

    The returned indices refer to y, which is the filtered test-label array.
    """
    normal_indices = np.flatnonzero(y == 0)
    anomaly_indices = np.flatnonzero(y == 1)

    if len(normal_indices) == 0 or len(anomaly_indices) == 0:
        return np.array([], dtype=int)

    n_normal = min(maximum_per_class, len(normal_indices))
    n_anomaly = min(maximum_per_class, len(anomaly_indices))

    selected_normal = rng.choice(
        normal_indices,
        size=n_normal,
        replace=False,
    )
    selected_anomaly = rng.choice(
        anomaly_indices,
        size=n_anomaly,
        replace=False,
    )

    selected = np.concatenate([selected_normal, selected_anomaly])
    rng.shuffle(selected)

    return selected.astype(int)


def encode_angles(X):
    """Map PCA features into a bounded rotation-angle range."""
    return 2.0 * np.arctan(np.asarray(X, dtype=float))


# ---------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------

def evaluate_scores(y_true, scores, threshold):
    """Evaluate anomaly scores; larger scores indicate anomalies."""
    y_true = np.asarray(y_true, dtype=np.int8)
    scores = np.asarray(scores, dtype=float)

    if scores.ndim != 1 or len(scores) != len(y_true):
        raise ValueError("Scores and labels must be aligned 1D arrays.")

    if not np.isfinite(scores).all():
        raise ValueError("Anomaly scores contain NaN or infinite values.")

    predictions = (scores >= threshold).astype(np.int8)

    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()

    has_both_classes = np.unique(y_true).size == 2

    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": (
            float(balanced_accuracy_score(y_true, predictions))
            if has_both_classes else float("nan")
        ),
        "precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, predictions, zero_division=0)
        ),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": (
            float(roc_auc_score(y_true, scores))
            if has_both_classes else float("nan")
        ),
        "average_precision": (
            float(average_precision_score(y_true, scores))
            if has_both_classes else float("nan")
        ),
        "threshold": float(threshold),
        "predicted_anomalies": int(predictions.sum()),
        "actual_anomalies": int(y_true.sum()),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "predictions": predictions,
    }


def load_metadata():
    """Load and group all labelled SMAP metadata rows by channel."""
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"NASA metadata file not found: {METADATA_PATH}"
        )

    metadata = pd.read_csv(METADATA_PATH)

    required = {"chan_id", "spacecraft", "anomaly_sequences"}
    missing = required - set(metadata.columns)

    if missing:
        raise ValueError(
            f"Metadata is missing required columns: {sorted(missing)}"
        )

    smap = metadata[
        metadata["spacecraft"].astype(str).str.upper() == "SMAP"
    ].copy()

    if smap.empty:
        raise RuntimeError("No SMAP rows were found in the metadata.")

    return smap.groupby("chan_id", sort=True)


# ---------------------------------------------------------------------
# Per-channel data preparation
# ---------------------------------------------------------------------

def prepare_channel(
    channel,
    rows,
    rng,
    n_qubits,
    reference_size,
    calibration_size,
    max_test_per_class,
):
    """Load one channel and return leakage-controlled model inputs."""
    train_path = TRAIN_DIR / f"{channel}.npy"
    test_path = TEST_DIR / f"{channel}.npy"

    if not train_path.exists() or not test_path.exists():
        return None, "missing train or test array"

    X_train_original = np.asarray(
        np.load(train_path, allow_pickle=False),
        dtype=np.float64,
    )
    X_test_original = np.asarray(
        np.load(test_path, allow_pickle=False),
        dtype=np.float64,
    )

    if X_train_original.ndim != 2 or X_test_original.ndim != 2:
        return None, "train/test arrays are not 2D"

    if X_train_original.shape[1] != X_test_original.shape[1]:
        return None, "train/test feature counts differ"

    if X_train_original.shape[1] < n_qubits:
        return None, (
            f"only {X_train_original.shape[1]} features; "
            f"{n_qubits} are required"
        )

    intervals = []
    for _, row in rows.iterrows():
        intervals.extend(parse_intervals(row["anomaly_sequences"]))

        if "num_values" in rows.columns and pd.notna(row["num_values"]):
            if int(row["num_values"]) != len(X_test_original):
                raise ValueError(
                    f"{channel}: metadata num_values does not match "
                    "the test array length."
                )

    y_test_original = make_labels(
        len(X_test_original),
        intervals,
        channel,
    )

    # Preserve the original temporal split locations. Invalid rows are
    # removed within each temporal partition, not before partitioning.
    n_train_original = len(X_train_original)
    reference_end = int(0.65 * n_train_original)
    calibration_start = int(0.75 * n_train_original)

    reference_region = X_train_original[:reference_end]
    calibration_region = X_train_original[calibration_start:]

    reference_region = reference_region[
        np.isfinite(reference_region).all(axis=1)
    ]
    calibration_region = calibration_region[
        np.isfinite(calibration_region).all(axis=1)
    ]

    if len(reference_region) < max(n_qubits, 2):
        return None, "insufficient finite reference-period rows"

    if len(calibration_region) < 2:
        return None, "insufficient finite calibration-period rows"

    # Keep original test row indices so output predictions can be traced
    # back to the downloaded NASA test arrays.
    valid_test_mask = np.isfinite(X_test_original).all(axis=1)
    valid_test_original_indices = np.flatnonzero(valid_test_mask)

    X_test_valid = X_test_original[valid_test_mask]
    y_test_valid = y_test_original[valid_test_mask]

    selected_test_indices = stratified_test_indices(
        y_test_valid,
        max_test_per_class,
        rng,
    )

    if len(selected_test_indices) == 0:
        return None, "test sequence does not contain both classes"

    X_reference_raw = sample_rows(
        reference_region,
        reference_size,
        rng,
    )
    X_calibration_raw = sample_rows(
        calibration_region,
        calibration_size,
        rng,
    )

    if len(X_reference_raw) < 2 or len(X_calibration_raw) < 2:
        return None, "insufficient sampled reference/calibration rows"

    X_evaluation_raw = X_test_valid[selected_test_indices]
    y_evaluation = y_test_valid[selected_test_indices]
    original_test_indices = valid_test_original_indices[
        selected_test_indices
    ]

    # Fit scaler and PCA only on the early reference training period.
    scaler = StandardScaler()
    reference_region_scaled = scaler.fit_transform(reference_region)

    pca = PCA(
        n_components=n_qubits,
        svd_solver="full",
    )
    pca.fit(reference_region_scaled)

    X_reference_pca = pca.transform(
        scaler.transform(X_reference_raw)
    )
    X_calibration_pca = pca.transform(
        scaler.transform(X_calibration_raw)
    )
    X_evaluation_pca = pca.transform(
        scaler.transform(X_evaluation_raw)
    )

    # Both models receive the same 16-dimensional encoded representation.
    X_reference = encode_angles(X_reference_pca)
    X_calibration = encode_angles(X_calibration_pca)
    X_evaluation = encode_angles(X_evaluation_pca)

    prepared = {
        "X_reference": X_reference,
        "X_calibration": X_calibration,
        "X_evaluation": X_evaluation,
        "y_evaluation": y_evaluation,
        "test_row_indices": original_test_indices,
        "n_original_train": int(len(X_train_original)),
        "n_original_test": int(len(X_test_original)),
        "n_original_test_anomalies": int(y_test_original.sum()),
        "n_valid_test": int(len(X_test_valid)),
        "n_valid_reference_region": int(len(reference_region)),
        "n_valid_calibration_region": int(len(calibration_region)),
        "n_reference": int(len(X_reference)),
        "n_calibration": int(len(X_calibration)),
        "n_evaluation": int(len(X_evaluation)),
        "n_evaluation_normals": int((y_evaluation == 0).sum()),
        "n_evaluation_anomalies": int((y_evaluation == 1).sum()),
        "pca_explained_variance": float(
            pca.explained_variance_ratio_.sum()
        ),
        "n_original_features": int(X_train_original.shape[1]),
        "X_reference_raw_diag": X_reference_raw,
        "X_calibration_raw_diag": X_calibration_raw,
        "X_evaluation_raw_diag": X_evaluation_raw,

        "X_reference_pca_diag": X_reference_pca,
        "X_calibration_pca_diag": X_calibration_pca,
        "X_evaluation_pca_diag": X_evaluation_pca,

        "n_unique_reference_pool_raw": int(
            np.unique(reference_region, axis=0).shape[0]
        ),
        "n_unique_calibration_pool_raw": int(
            np.unique(calibration_region, axis=0).shape[0]
        ),
        "n_unique_test_pool_raw": int(
            np.unique(X_test_valid, axis=0).shape[0]
        ),
    }

    return prepared, None


# ---------------------------------------------------------------------
# Model execution
# ---------------------------------------------------------------------

def run_classical_ocsvm(data, nu, threshold_quantile):
    """Run the classical RBF One-Class SVM and calibrate its score threshold."""
    model = OneClassSVM(
        kernel="rbf",
        gamma="scale",
        nu=nu,
    )

    fit_start = time.perf_counter()
    model.fit(data["X_reference"])
    fit_seconds = time.perf_counter() - fit_start

    score_start = time.perf_counter()
    calibration_scores = -model.decision_function(
        data["X_calibration"]
    ).ravel()
    test_scores = -model.decision_function(
        data["X_evaluation"]
    ).ravel()
    score_seconds = time.perf_counter() - score_start

    threshold = float(np.quantile(calibration_scores, threshold_quantile))

    metrics = evaluate_scores(
        data["y_evaluation"],
        test_scores,
        threshold,
    )

    return {
        "metrics": metrics,
        "test_scores": test_scores,
        "fit_seconds": fit_seconds,
        "score_seconds": score_seconds,
        "total_seconds": fit_seconds + score_seconds,
    }


def build_feature_map(name, n_qubits, reps):
    if name == "z":
        return ZFeatureMap(n_qubits=n_qubits, reps=reps)
    if name == "zz":
        return ZZFeatureMap(n_qubits=n_qubits, reps=reps)
    raise ValueError(f"Unsupported feature map: {name}")


def run_quantum_ocsvm(data, args):
    """Run QMLib's quantum-kernel One-Class SVM with state caching."""
    feature_map = build_feature_map(
        args.feature_map,
        args.n_qubits,
        args.reps,
    )

    kernel = CachedFidelityKernel(feature_map=feature_map)
    model = QuantumOneClassSVM(
        kernel=kernel,
        nu=args.nu,
    )

    fit_start = time.perf_counter()
    model.fit(data["X_reference"])
    fit_seconds = time.perf_counter() - fit_start

    score_start = time.perf_counter()
    calibration_scores = model.anomaly_score(
        data["X_calibration"]
    )
    test_scores = model.anomaly_score(
        data["X_evaluation"]
    )
    score_seconds = time.perf_counter() - score_start

    threshold = float(np.quantile(calibration_scores, args.threshold_quantile))

    metrics = evaluate_scores(
        data["y_evaluation"],
        test_scores,
        threshold,
    )

    # Record cache memory before releasing statevectors.
    cache = getattr(kernel, "_state_cache", {})
    cache_entries = len(cache)
    cache_bytes = sum(
        np.asarray(state).nbytes for state in cache.values()
    )

    kernel.clear_cache()
    del model
    del kernel
    gc.collect()

    return {
        "metrics": metrics,
        "test_scores": test_scores,
        "fit_seconds": fit_seconds,
        "score_seconds": score_seconds,
        "total_seconds": fit_seconds + score_seconds,
        "cache_entries": int(cache_entries),
        "cache_memory_mb": float(cache_bytes / (1024.0 ** 2)),
        "feature_map_config": feature_map.get_config(),
    }


# ---------------------------------------------------------------------
# Aggregation and command-line entry point
# ---------------------------------------------------------------------

def aggregate_results(per_channel_df, predictions_df, methods, runtime):
    """Aggregate thresholded metrics and macro ranking metrics."""
    summaries = []

    for method in methods:
        channel_subset = per_channel_df[
            per_channel_df["method"] == method
        ]
        prediction_subset = predictions_df[
            predictions_df["method"] == method
        ]

        y_true = prediction_subset["y_true"].to_numpy(dtype=np.int8)
        y_pred = prediction_subset["y_pred"].to_numpy(dtype=np.int8)

        matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = matrix.ravel()

        summaries.append({
            "method": method,
            "channels_evaluated": int(
                channel_subset["channel"].nunique()
            ),
            "evaluation_samples": int(len(y_true)),
            "actual_anomalies": int(y_true.sum()),
            "predicted_anomalies": int(y_pred.sum()),
            "pooled_accuracy": float(accuracy_score(y_true, y_pred)),
            "pooled_balanced_accuracy": float(
                balanced_accuracy_score(y_true, y_pred)
            ),
            "pooled_precision": float(
                precision_score(y_true, y_pred, zero_division=0)
            ),
            "pooled_recall": float(
                recall_score(y_true, y_pred, zero_division=0)
            ),
            "pooled_f1": float(
                f1_score(y_true, y_pred, zero_division=0)
            ),
            "pooled_tn": int(tn),
            "pooled_fp": int(fp),
            "pooled_fn": int(fn),
            "pooled_tp": int(tp),
            "macro_precision": float(channel_subset["precision"].mean()),
            "macro_recall": float(channel_subset["recall"].mean()),
            "macro_f1": float(channel_subset["f1"].mean()),
            "macro_roc_auc": float(channel_subset["roc_auc"].mean()),
            "macro_average_precision": float(
                channel_subset["average_precision"].mean()
            ),
            "total_fit_seconds": float(
                channel_subset["fit_seconds"].sum()
            ),
            "total_score_seconds": float(
                channel_subset["score_seconds"].sum()
            ),
            "total_runtime_seconds": float(runtime[method]),
        })

    return pd.DataFrame(summaries)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "NASA SMAP: classical RBF One-Class SVM versus "
            "16-qubit quantum-kernel One-Class SVM."
        )
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-qubits", type=int, default=16)
    parser.add_argument("--feature-map", choices=["z", "zz"], default="zz")
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--reference-size", type=int, default=32)
    parser.add_argument("--calibration-size", type=int, default=256)
    parser.add_argument("--max-test-per-class", type=int, default=50)
    parser.add_argument("--threshold-quantile", type=float, default=0.99)
    parser.add_argument("--nu", type=float, default=0.05)
    parser.add_argument("--channels", nargs="+", default=None)
    parser.add_argument("--max-channels", type=int, default=None)
    parser.add_argument(
        "--run-name",
        default="smap_experiment_v4",
        help="Prefix used for output filenames.",
    )

    args = parser.parse_args()

    if args.n_qubits < 1:
        parser.error("--n-qubits must be positive.")
    if args.reps < 1:
        parser.error("--reps must be positive.")
    if args.reference_size < 2:
        parser.error("--reference-size must be at least 2.")
    if args.calibration_size < 2:
        parser.error("--calibration-size must be at least 2.")
    if args.max_test_per_class < 1:
        parser.error("--max-test-per-class must be positive.")
    if not 0.0 < args.threshold_quantile < 1.0:
        parser.error("--threshold-quantile must be between 0 and 1.")
    if not 0.0 < args.nu <= 1.0:
        parser.error("--nu must be in (0, 1].")
    if args.max_channels is not None and args.max_channels < 1:
        parser.error("--max-channels must be positive.")
    if args.channels and args.max_channels is not None:
        parser.error("Use either --channels or --max-channels, not both.")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_name):
        parser.error("--run-name may contain only letters, digits, _ and -.")

    return args


def main():
    args = parse_args()

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {METADATA_PATH}. "
            "Check the NASA Data directory structure."
        )

    grouped_metadata = load_metadata()
    all_channels = [str(channel) for channel in grouped_metadata.groups]
    channel_seed_index = {
        channel: index for index, channel in enumerate(all_channels)
    }

    if args.channels:
        unknown = sorted(set(args.channels) - set(all_channels))
        if unknown:
            raise ValueError(
                f"Requested channels not found in SMAP metadata: {unknown}"
            )
        channels = list(args.channels)
    else:
        channels = all_channels
        if args.max_channels is not None:
            channels = channels[:args.max_channels]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    methods = [
        "ClassicalOneClassSVM_RBF",
        f"QuantumOneClassSVM_{args.feature_map.upper()}",
    ]

    per_channel_rows = []
    prediction_frames = []
    skipped_channels = []
    runtime_by_method = {method: 0.0 for method in methods}
    processed_channels = []
    feature_map_config = None
    total_available_test_timesteps = 0
    total_available_test_anomalies = 0
    total_evaluation_samples = 0
    total_evaluation_anomalies = 0

    print("=" * 78)
    print("NASA SMAP V4: CLASSICAL SVM VS QUANTUM-KERNEL SVM")
    print("=" * 78)
    print(f"Channels requested: {len(channels)}")
    print(f"Qubits / PCA components: {args.n_qubits}")
    print(f"Quantum feature map: {args.feature_map.upper()}")
    print(f"Reference size: up to {args.reference_size}")
    print(f"Calibration size: up to {args.calibration_size}")
    print(
        "Test sampling: up to "
        f"{args.max_test_per_class} normal + "
        f"{args.max_test_per_class} anomalous observations per channel"
    )
    print(f"Threshold quantile: {args.threshold_quantile}")
    print(f"One-Class SVM nu: {args.nu}")
    print()

    for channel_number, channel in enumerate(channels, start=1):
        rows = grouped_metadata.get_group(channel)
        # A channel receives the same random stream whether it is run alone
        # as a pilot or as part of the full experiment.
        rng = np.random.default_rng(
            args.seed + 1009 * channel_seed_index[channel]
        )

        print(f"[{channel_number}/{len(channels)}] Processing {channel}...")

        data, skip_reason = prepare_channel(
            channel=channel,
            rows=rows,
            rng=rng,
            n_qubits=args.n_qubits,
            reference_size=args.reference_size,
            calibration_size=args.calibration_size,
            max_test_per_class=args.max_test_per_class,
        )

        if data is None:
            print(f"  Skipped: {skip_reason}")
            skipped_channels.append({
                "channel": channel,
                "reason": skip_reason,
            })
            continue

        print(
            f"  Samples: reference={data['n_reference']}, "
            f"calibration={data['n_calibration']}, "
            f"test={data['n_evaluation']} "
            f"(normal={data['n_evaluation_normals']}, "
            f"anomaly={data['n_evaluation_anomalies']})"
        )
        # ---------------------------------------------------------
        # Diagnostic: distinct encoded feature vectors
        # ---------------------------------------------------------

        diagnostic_stages = {
            "Raw input": {
                "Reference": data["X_reference_raw_diag"],
                "Calibration": data["X_calibration_raw_diag"],
                "Evaluation": data["X_evaluation_raw_diag"],
            },
            "After PCA": {
                "Reference": data["X_reference_pca_diag"],
                "Calibration": data["X_calibration_pca_diag"],
                "Evaluation": data["X_evaluation_pca_diag"],
            },
            "After angle encoding": {
                "Reference": data["X_reference"],
                "Calibration": data["X_calibration"],
                "Evaluation": data["X_evaluation"],
            },
        }

        print("\n--- INPUT UNIQUENESS BY PREPROCESSING STAGE ---")

        for stage_name, split_sets in diagnostic_stages.items():
            print(f"\n{stage_name}")

            combined = np.vstack([
                split_sets["Reference"],
                split_sets["Calibration"],
                split_sets["Evaluation"],
            ])

            for split_name, values in split_sets.items():
                exact_unique = np.unique(values, axis=0).shape[0]

                print(
                    f"{split_name}: {len(values)} samples, "
                    f"{exact_unique} unique vectors"
                )

            print(
                "Combined unique vectors:",
                np.unique(combined, axis=0).shape[0],
            )

        print("\n--- FULL RAW CANDIDATE POOL UNIQUENESS ---")
        print(
            "Reference training pool:",
            data["n_unique_reference_pool_raw"],
        )
        print(
            "Calibration training pool:",
            data["n_unique_calibration_pool_raw"],
        )
        print(
            "Full test pool:",
            data["n_unique_test_pool_raw"],
        )
        print("\n--- CANDIDATE POOL SIZES ---")
        print("Reference pool rows:", data["n_valid_reference_region"])
        print("Calibration pool rows:", data["n_valid_calibration_region"])
        print("Full valid test rows:", data["n_valid_test"])

        # ---------------- Classical RBF One-Class SVM ----------------
        classical = run_classical_ocsvm(
            data, args.nu, args.threshold_quantile
        )
        classical_metrics = classical["metrics"]
        runtime_by_method[methods[0]] += classical["total_seconds"]

        per_channel_rows.append({
            "channel": channel,
            "method": methods[0],
            "n_reference": data["n_reference"],
            "n_calibration": data["n_calibration"],
            "n_test": data["n_evaluation"],
            "n_test_normals": data["n_evaluation_normals"],
            "n_test_anomalies": data["n_evaluation_anomalies"],
            "pca_explained_variance": data["pca_explained_variance"],
            "fit_seconds": classical["fit_seconds"],
            "score_seconds": classical["score_seconds"],
            "total_seconds": classical["total_seconds"],
            **{
                key: value
                for key, value in classical_metrics.items()
                if key != "predictions"
            },
        })


        # ---------------- Quantum-kernel One-Class SVM ----------------
        quantum = run_quantum_ocsvm(data, args)
        quantum_metrics = quantum["metrics"]
        runtime_by_method[methods[1]] += quantum["total_seconds"]
        feature_map_config = quantum["feature_map_config"]

        per_channel_rows.append({
            "channel": channel,
            "method": methods[1],
            "n_reference": data["n_reference"],
            "n_calibration": data["n_calibration"],
            "n_test": data["n_evaluation"],
            "n_test_normals": data["n_evaluation_normals"],
            "n_test_anomalies": data["n_evaluation_anomalies"],
            "pca_explained_variance": data["pca_explained_variance"],
            "fit_seconds": quantum["fit_seconds"],
            "score_seconds": quantum["score_seconds"],
            "total_seconds": quantum["total_seconds"],
            "state_cache_entries": quantum["cache_entries"],
            "state_cache_memory_mb": quantum["cache_memory_mb"],
            **{
                key: value
                for key, value in quantum_metrics.items()
                if key != "predictions"
            },
        })

        processed_channels.append(channel)
        total_available_test_timesteps += data["n_original_test"]
        total_available_test_anomalies += data["n_original_test_anomalies"]
        total_evaluation_samples += data["n_evaluation"]
        total_evaluation_anomalies += data["n_evaluation_anomalies"]

        print(
            f"  Classical F1={classical_metrics['f1']:.4f}, "
            f"runtime={classical['total_seconds']:.2f}s"
        )
        print(
            f"  Quantum   F1={quantum_metrics['f1']:.4f}, "
            f"runtime={quantum['total_seconds']:.2f}s, "
            f"cached states={quantum['cache_entries']}, "
            f"cache={quantum['cache_memory_mb']:.1f} MiB"
        )

        print("\n--- ANOMALY SCORE DIAGNOSTICS ---")

        for name, scores, threshold in [
            (
                "Classical",
                classical["test_scores"],
                classical["metrics"]["threshold"],
            ),
            (
                "Quantum",
                quantum["test_scores"],
                quantum["metrics"]["threshold"],
            ),
        ]:

            scores = np.asarray(scores)

            print(f"\n{name}")
            print("Unique scores:", np.unique(scores).size)
            print("Minimum:", scores.min())
            print("Median:", np.median(scores))
            print("Maximum:", scores.max())
            print("Threshold:", threshold)
            print("Predicted anomalies:", int((scores >= threshold).sum()))

        # Keep scores and predictions for later inspection and plots.
        # Keep the final test targets and original row indices for traceability.
        # The two methods are evaluated against exactly the same rows.
        classical_threshold = per_channel_rows[-2]["threshold"]
        quantum_threshold = per_channel_rows[-1]["threshold"]

        # Obtain predictions already returned by evaluate_scores.
        classical_pred = classical_metrics["predictions"]
        quantum_pred = quantum_metrics["predictions"]

        prediction_frames.append(pd.DataFrame({
            "channel": channel,
            "method": methods[0],
            "test_row_index": data["test_row_indices"],
            "y_true": data["y_evaluation"],
            "anomaly_score": classical["test_scores"],
            "threshold": classical_threshold,
            "y_pred": classical_pred,
        }))

        prediction_frames.append(pd.DataFrame({
            "channel": channel,
            "method": methods[1],
            "test_row_index": data["test_row_indices"],
            "y_true": data["y_evaluation"],
            "anomaly_score": quantum["test_scores"],
            "threshold": quantum_threshold,
            "y_pred": quantum_pred,
        }))

    if not per_channel_rows or not prediction_frames:
        raise RuntimeError(
            "No channels completed successfully. Check the dataset and "
            "the requested channel/sample settings."
        )

    per_channel_df = pd.DataFrame(per_channel_rows)
    predictions_df = pd.concat(prediction_frames, ignore_index=True)

    summary_df = aggregate_results(
        per_channel_df,
        predictions_df,
        methods,
        runtime_by_method,
    )

    # Output names are configurable so pilot runs cannot overwrite final runs.
    prefix = args.run_name
    metrics_path = RESULTS_DIR / f"{prefix}_metrics.csv"
    per_channel_path = RESULTS_DIR / f"{prefix}_per_channel.csv"
    predictions_path = RESULTS_DIR / f"{prefix}_predictions.csv"
    config_path = RESULTS_DIR / f"{prefix}_config.json"

    summary_df.to_csv(metrics_path, index=False)
    per_channel_df.to_csv(per_channel_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)

    config = {
        "experiment": (
            "NASA SMAP classical RBF One-Class SVM vs "
            "QMLib quantum-kernel One-Class SVM"
        ),
        "run_name": args.run_name,
        "dataset": "NASA SMAP",
        "seed": args.seed,
        "requested_channels": channels,
        "processed_channels": processed_channels,
        "skipped_channels": skipped_channels,
        "n_processed_channels": len(processed_channels),
        "full_test_timesteps_for_processed_channels": (
            total_available_test_timesteps
        ),
        "full_test_anomalies_for_processed_channels": (
            total_available_test_anomalies
        ),
        "evaluation_samples": total_evaluation_samples,
        "evaluation_anomalies": total_evaluation_anomalies,
        "n_qubits": args.n_qubits,
        "n_pca_components": args.n_qubits,
        "quantum_feature_map": feature_map_config,
        "quantum_backend": "PennyLane default.qubit",
        "classical_model": {
            "name": "sklearn OneClassSVM",
            "kernel": "rbf",
            "gamma": "scale",
            "nu": args.nu,
        },
        "quantum_model": {
            "name": "QMLib QuantumOneClassSVM",
            "kernel": "CachedFidelityKernel",
            "nu": args.nu,
        },
        "preprocessing": {
            "scaler": "StandardScaler",
            "pca": f"PCA(n_components={args.n_qubits}, svd_solver='full')",
            "fit_period": "first 65% of each training sequence",
            "angle_encoding": "2 * arctan(PCA output)",
            "same_encoded_inputs_for_both_models": True,
        },
        "training_protocol": {
            "reference_period": "first 65% of training sequence",
            "temporal_gap": "65%-75% of training sequence",
            "calibration_period": "last 25% of training sequence",
            "training_data_assumption": (
                "Training sequences are assumed to be predominantly normal."
            ),
            "reference_sample_limit_per_channel": args.reference_size,
            "calibration_sample_limit_per_channel": args.calibration_size,
        },
        "evaluation_protocol": {
            "sampling": "class-stratified within each channel",
            "max_normal_samples_per_channel": args.max_test_per_class,
            "max_anomalous_samples_per_channel": args.max_test_per_class,
            "threshold_quantile": args.threshold_quantile,
            "threshold_source": (
                "Separate normal calibration scores for each channel and model."
            ),
            "labels": {
                "0": "normal",
                "1": "anomaly",
            },
            "interval_convention": "zero-based inclusive endpoints",
            "warning": (
                "The class-stratified evaluation sample does not preserve "
                "the natural anomaly prevalence. Precision, F1 and AP describe "
                "this balanced evaluation design and should not be interpreted "
                "as deployment metrics at the original SMAP prevalence."
            ),
        },
        "runtime_seconds": runtime_by_method,
        "software_versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "pennylane": qml.__version__,
        },
        "notes": [
            "v1-v3 files and results are retained unchanged.",
            "Both methods use identical channel-specific reference, "
            "calibration and test observations.",
            "Scaler and PCA are fitted on training data only.",
            "Test labels are used only for evaluation and stratified sampling.",
            "Thresholds are not selected using test labels.",
            "Quantum state cache is cleared after each channel.",
            "No quantum advantage is inferred from these experiments.",
        ],
    }

    with config_path.open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)

    print("\n" + "=" * 78)
    print("V4 AGGREGATE RESULTS")
    print("=" * 78)
    display_columns = [
        "method",
        "channels_evaluated",
        "evaluation_samples",
        "pooled_precision",
        "pooled_recall",
        "pooled_f1",
        "macro_roc_auc",
        "macro_average_precision",
        "total_runtime_seconds",
    ]
    print(
        summary_df[display_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\nSaved outputs:")
    print(metrics_path)
    print(per_channel_path)
    print(predictions_path)
    print(config_path)

if __name__ == "__main__":
    main()
