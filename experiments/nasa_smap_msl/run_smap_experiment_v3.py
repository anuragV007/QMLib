
import ast
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import StandardScaler

from qmlib.feature_maps import ZFeatureMap, ZZFeatureMap
from qmlib.kernels import FidelityKernel


# =========================================================
# 1. Configuration
# =========================================================

SEED = 42
N_QUBITS = 2
N_REFERENCE = 32
MAX_CALIBRATION = 256
MAX_EVALUATION_PER_CHANNEL = 100
MAX_IF_TRAINING = 5000
THRESHOLD_QUANTILE = 0.99

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "NASA Data"

METADATA_PATH = DATA_ROOT / "labeled_anomalies.csv"
TRAIN_DIR = DATA_ROOT / "data" / "data" / "train"
TEST_DIR = DATA_ROOT / "data" / "data" / "test"

RESULTS_DIR = ROOT / "experiments" / "nasa_smap_msl" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

METHODS = [
    "ZFeatureMap",
    "ZZFeatureMap",
    "ClassicalNearestReference",
    "IsolationForest",
]


# =========================================================
# 2. Utility functions
# =========================================================

def parse_intervals(value):
    """Parse anomaly intervals from the metadata."""
    if value is None:
        return []

    if isinstance(value, float) and np.isnan(value):
        return []

    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return []

    intervals = []

    for interval in value:
        if len(interval) >= 2:
            start = int(interval[0])
            end = int(interval[1])

            if end >= start:
                intervals.append((start, end))

    return intervals


def make_labels(n_rows, intervals):
    """Create labels using zero-based, inclusive intervals."""
    labels = np.zeros(n_rows, dtype=np.int8)

    for start, end in intervals:
        start = max(0, start)
        end = min(n_rows - 1, end)

        if end >= start:
            labels[start:end + 1] = 1

    return labels


def sample_rows(X, n, rng):
    """Sample rows without replacement."""
    if len(X) <= n:
        return X.copy()

    indices = rng.choice(len(X), size=n, replace=False)
    return X[indices]


def compute_metrics(y_true, scores, threshold):
    """Evaluate anomaly scores using a calibration-only threshold."""
    predictions = (scores >= threshold).astype(np.int8)

    has_both_classes = (
        np.unique(y_true).size == 2
    )

    ap = np.nan

    if has_both_classes:
        ap = float(average_precision_score(y_true, scores))

    balanced_acc = np.nan

    if has_both_classes:
        balanced_acc = float(
            balanced_accuracy_score(y_true, predictions)
        )

    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": balanced_acc,
        "precision": float(
            precision_score(
                y_true, predictions, zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y_true, predictions, zero_division=0
            )
        ),
        "f1": float(
            f1_score(
                y_true, predictions, zero_division=0
            )
        ),
        "average_precision": ap,
        "threshold": float(threshold),
        "predicted_anomalies": int(predictions.sum()),
        "actual_anomalies": int(y_true.sum()),
    }


# =========================================================
# 3. Load and combine SMAP metadata
# =========================================================

if not METADATA_PATH.exists():
    raise FileNotFoundError(
        f"Metadata not found: {METADATA_PATH}"
    )

metadata = pd.read_csv(METADATA_PATH)

required_columns = {
    "chan_id",
    "spacecraft",
    "anomaly_sequences",
}

missing_columns = required_columns - set(metadata.columns)

if missing_columns:
    raise ValueError(
        f"Missing metadata columns: {sorted(missing_columns)}"
    )

smap_metadata = metadata[
    metadata["spacecraft"].astype(str).str.upper() == "SMAP"
].copy()

intervals_by_channel = {}

for _, row in smap_metadata.iterrows():
    channel = str(row["chan_id"])

    intervals_by_channel.setdefault(channel, []).extend(
        parse_intervals(row["anomaly_sequences"])
    )

channels = sorted(intervals_by_channel)

print(f"SMAP metadata channels: {len(channels)}")


# =========================================================
# 4. Per-channel train/calibration/test protocol
# =========================================================

channel_results = []
prediction_results = []

# Accumulate elapsed time separately for each method.
runtime_by_method = {name: 0.0 for name in METHODS}

total_test_timesteps = 0
total_test_anomalies = 0
total_evaluation_samples = 0
total_evaluation_anomalies = 0
processed_channels = []

for channel_number, channel in enumerate(channels):

    train_path = TRAIN_DIR / f"{channel}.npy"
    test_path = TEST_DIR / f"{channel}.npy"

    if not train_path.exists() or not test_path.exists():
        print(f"Skipping {channel}: missing train/test file")
        continue

    X_train_original = np.asarray(
        np.load(train_path), dtype=np.float64
    )

    X_test_original = np.asarray(
        np.load(test_path), dtype=np.float64
    )

    if X_train_original.ndim != 2 or X_test_original.ndim != 2:
        print(f"Skipping {channel}: expected 2D arrays")
        continue

    if X_train_original.shape[1] != X_test_original.shape[1]:
        print(f"Skipping {channel}: feature count mismatch")
        continue

    # Preserve the original test row indices so labels stay aligned.
    y_test_original = make_labels(
        len(X_test_original),
        intervals_by_channel[channel],
    )

    valid_train = np.isfinite(X_train_original).all(axis=1)
    valid_test = np.isfinite(X_test_original).all(axis=1)

    X_train = X_train_original[valid_train]
    X_test = X_test_original[valid_test]
    y_test = y_test_original[valid_test]

    if len(X_train) < 20 or len(X_test) == 0:
        print(f"Skipping {channel}: insufficient valid rows")
        continue

    # Training is assumed to be predominantly normal.
    # Use early 65% for fitting/reference candidates.
    reference_end = int(0.65 * len(X_train))

    # Leave a temporal gap, then use the final 25% for calibration.
    calibration_start = int(0.75 * len(X_train))

    X_reference_region = X_train[:reference_end]
    X_calibration_region = X_train[calibration_start:]

    if (
        len(X_reference_region) < 2
        or len(X_calibration_region) == 0
    ):
        print(f"Skipping {channel}: empty train partition")
        continue

    # Independent, reproducible random generator per channel.
    channel_rng = np.random.default_rng(
        SEED + 1009 * channel_number
    )

    # Select a fixed reference set for the quantum and
    # nearest-reference methods.
    X_reference_raw = sample_rows(
        X_reference_region,
        N_REFERENCE,
        channel_rng,
    )

    X_calibration_raw = sample_rows(
        X_calibration_region,
        MAX_CALIBRATION,
        channel_rng,
    )

    n_test = min(
        MAX_EVALUATION_PER_CHANNEL,
        len(X_test),
    )

    test_indices = channel_rng.choice(
        len(X_test),
        size=n_test,
        replace=False,
    )

    X_evaluation_raw = X_test[test_indices]
    y_evaluation = y_test[test_indices]

    # Fit scaler/PCA only on the early training partition.
    scaler = StandardScaler()
    scaler.fit(X_reference_region)

    X_train_scaled = scaler.transform(X_reference_region)
    X_reference_scaled = scaler.transform(X_reference_raw)
    X_calibration_scaled = scaler.transform(X_calibration_raw)
    X_evaluation_scaled = scaler.transform(X_evaluation_raw)

    if X_train_scaled.shape[1] < N_QUBITS:
        print(f"Skipping {channel}: too few input features")
        continue

    pca = PCA(n_components=N_QUBITS)
    pca.fit(X_train_scaled)

    X_train_pca = pca.transform(X_train_scaled)
    X_reference_pca = pca.transform(X_reference_scaled)
    X_calibration_pca = pca.transform(X_calibration_scaled)
    X_evaluation_pca = pca.transform(X_evaluation_scaled)

    # Deterministic bounded encoding for every method.
    def encode_angles(X):
        return 2.0 * np.arctan(X)

    X_train_encoded = encode_angles(X_train_pca)
    X_reference = encode_angles(X_reference_pca)
    X_calibration = encode_angles(X_calibration_pca)
    X_evaluation = encode_angles(X_evaluation_pca)

    # Isolation Forest uses the same encoded representation.
    if_training = sample_rows(
        X_train_encoded,
        min(MAX_IF_TRAINING, len(X_train_encoded)),
        channel_rng,
    )

    # -----------------------------------------------------
    # Prepare score arrays for each detector
    # -----------------------------------------------------

    scores_by_method = {}
    calibration_scores_by_method = {}

    # -----------------------------------------------------
    # Method 1: Quantum ZFeatureMap
    # -----------------------------------------------------

    start = time.perf_counter()

    z_kernel = FidelityKernel(
        feature_map=ZFeatureMap(
            n_qubits=N_QUBITS,
            reps=1,
        )
    )

    K_cal_z = z_kernel.compute(
        X_calibration, X_reference
    )

    K_test_z = z_kernel.compute(
        X_evaluation, X_reference
    )

    calibration_scores_by_method["ZFeatureMap"] = (
        1.0 - np.max(K_cal_z, axis=1)
    )

    scores_by_method["ZFeatureMap"] = (
        1.0 - np.max(K_test_z, axis=1)
    )

    runtime_by_method["ZFeatureMap"] += (
        time.perf_counter() - start
    )

    # -----------------------------------------------------
    # Method 2: Quantum ZZFeatureMap
    # -----------------------------------------------------

    start = time.perf_counter()

    zz_kernel = FidelityKernel(
        feature_map=ZZFeatureMap(
            n_qubits=N_QUBITS,
            reps=1,
        )
    )

    K_cal_zz = zz_kernel.compute(
        X_calibration, X_reference
    )

    K_test_zz = zz_kernel.compute(
        X_evaluation, X_reference
    )

    calibration_scores_by_method["ZZFeatureMap"] = (
        1.0 - np.max(K_cal_zz, axis=1)
    )

    scores_by_method["ZZFeatureMap"] = (
        1.0 - np.max(K_test_zz, axis=1)
    )

    runtime_by_method["ZZFeatureMap"] += (
        time.perf_counter() - start
    )

    # -----------------------------------------------------
    # Method 3: Classical nearest-reference distance
    # -----------------------------------------------------

    start = time.perf_counter()

    D_cal = pairwise_distances(
        X_calibration, X_reference, metric="euclidean"
    )

    D_test = pairwise_distances(
        X_evaluation, X_reference, metric="euclidean"
    )

    calibration_scores_by_method[
        "ClassicalNearestReference"
    ] = D_cal.min(axis=1)

    scores_by_method[
        "ClassicalNearestReference"
    ] = D_test.min(axis=1)

    runtime_by_method["ClassicalNearestReference"] += (
        time.perf_counter() - start
    )

    # -----------------------------------------------------
    # Method 4: Isolation Forest
    # -----------------------------------------------------

    start = time.perf_counter()

    isolation_forest = IsolationForest(
        n_estimators=100,
        max_samples="auto",
        contamination="auto",
        random_state=SEED + channel_number,
        n_jobs=1,
    )

    isolation_forest.fit(if_training)

    # Higher score means more anomalous.
    calibration_scores_by_method["IsolationForest"] = (
        -isolation_forest.decision_function(X_calibration)
    )

    scores_by_method["IsolationForest"] = (
        -isolation_forest.decision_function(X_evaluation)
    )

    runtime_by_method["IsolationForest"] += (
        time.perf_counter() - start
    )

    # -----------------------------------------------------
    # Threshold calibration and per-channel metrics
    # -----------------------------------------------------

    for method in METHODS:

        calibration_scores = (
            calibration_scores_by_method[method]
        )

        evaluation_scores = scores_by_method[method]

        # Threshold depends only on training calibration data.
        threshold = float(
            np.quantile(
                calibration_scores,
                THRESHOLD_QUANTILE,
            )
        )

        metrics = compute_metrics(
            y_evaluation,
            evaluation_scores,
            threshold,
        )

        channel_results.append({
            "channel": channel,
            "method": method,
            "n_reference": len(X_reference),
            "n_calibration": len(X_calibration),
            "n_test": len(y_evaluation),
            "pca_explained_variance": float(
                pca.explained_variance_ratio_.sum()
            ),
            **metrics,
        })

        predictions = (
            evaluation_scores >= threshold
        ).astype(np.int8)

        prediction_results.append(pd.DataFrame({
            "channel": channel,
            "method": method,
            "test_row_index": test_indices,
            "y_true": y_evaluation,
            "anomaly_score": evaluation_scores,
            "threshold": threshold,
            "y_pred": predictions,
        }))

    total_test_timesteps += len(X_test)
    total_test_anomalies += int(y_test.sum())
    total_evaluation_samples += len(y_evaluation)
    total_evaluation_anomalies += int(y_evaluation.sum())
    processed_channels.append(channel)

    print(
        f"Processed {channel}: "
        f"test={len(y_evaluation)}, "
        f"anomalies={int(y_evaluation.sum())}"
    )


# =========================================================
# 5. Aggregate results
# =========================================================

if not channel_results:
    raise RuntimeError(
        "No channels were processed. Check the dataset paths."
    )

per_channel_df = pd.DataFrame(channel_results)
predictions_df = pd.concat(
    prediction_results,
    ignore_index=True,
)

summary_rows = []

for method in METHODS:

    subset = per_channel_df[
        per_channel_df["method"] == method
    ].copy()

    method_predictions = predictions_df[
        predictions_df["method"] == method
    ]

    y_all = method_predictions["y_true"].to_numpy()
    pred_all = method_predictions["y_pred"].to_numpy()

    summary_rows.append({
        "method": method,
        "channels_evaluated": int(subset["channel"].nunique()),
        "evaluation_samples": int(len(y_all)),
        "actual_anomalies": int(y_all.sum()),
        "predicted_anomalies": int(pred_all.sum()),

        # Pooled classification metrics use the per-channel
        # calibration thresholds.
        "pooled_accuracy": float(
            accuracy_score(y_all, pred_all)
        ),
        "pooled_balanced_accuracy": float(
            balanced_accuracy_score(y_all, pred_all)
        ),
        "pooled_precision": float(
            precision_score(
                y_all, pred_all, zero_division=0
            )
        ),
        "pooled_recall": float(
            recall_score(
                y_all, pred_all, zero_division=0
            )
        ),
        "pooled_f1": float(
            f1_score(
                y_all, pred_all, zero_division=0
            )
        ),

        # Macro averages give each channel equal weight.
        # AP is averaged only over channels containing both classes.
        "macro_average_precision": float(
            subset["average_precision"].mean()
        ),
        "macro_balanced_accuracy": float(
            subset["balanced_accuracy"].mean()
        ),
        "macro_precision": float(
            subset["precision"].mean()
        ),
        "macro_recall": float(
            subset["recall"].mean()
        ),
        "macro_f1": float(
            subset["f1"].mean()
        ),
        "runtime_seconds": float(
            runtime_by_method[method]
        ),
    })

summary_df = pd.DataFrame(summary_rows)


# =========================================================
# 6. Save output files
# =========================================================

summary_path = RESULTS_DIR / "smap_experiment_v3_metrics.csv"
per_channel_path = (
    RESULTS_DIR / "smap_experiment_v3_per_channel.csv"
)
predictions_path = (
    RESULTS_DIR / "smap_experiment_v3_predictions.csv"
)
config_path = RESULTS_DIR / "smap_experiment_v3_config.json"

summary_df.to_csv(summary_path, index=False)
per_channel_df.to_csv(per_channel_path, index=False)
predictions_df.to_csv(predictions_path, index=False)

config = {
    "experiment": "SMAP channel-wise anomaly detection",
    "seed": SEED,
    "processed_channels": processed_channels,
    "n_processed_channels": len(processed_channels),
    "n_qubits": N_QUBITS,
    "feature_maps": ["ZFeatureMap", "ZZFeatureMap"],
    "classical_methods": [
        "ClassicalNearestReference",
        "IsolationForest",
    ],
    "angle_encoding": "2 * arctan(x)",
    "reference_sample_limit": N_REFERENCE,
    "calibration_sample_limit": MAX_CALIBRATION,
    "evaluation_sample_limit_per_channel": (
        MAX_EVALUATION_PER_CHANNEL
    ),
    "threshold_quantile": THRESHOLD_QUANTILE,
    "split": {
        "reference_and_preprocessing": "first 65% of training",
        "temporal_gap": "65%-75% of training",
        "calibration": "last 25% of training",
    },
    "test_timesteps_available": total_test_timesteps,
    "test_anomalies_available": total_test_anomalies,
    "evaluation_samples": total_evaluation_samples,
    "evaluation_anomalies": total_evaluation_anomalies,
    "notes": [
        "Each channel has its own scaler, PCA, reference set, "
        "calibration scores, and threshold.",
        "Scaler and PCA are fitted on the early training partition.",
        "Training data is assumed to be predominantly normal.",
        "Thresholds are computed from calibration data only.",
        "The same test rows are evaluated by all methods.",
        "Pooled average precision is omitted because raw score "
        "scales can differ across channels and methods.",
        "Macro average precision averages channel-level AP values "
        "for channels containing both classes.",
        "No quantum advantage is inferred from these experiments.",
    ],
}

with open(config_path, "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2)

print("\n" + "=" * 72)
print("V3 AGGREGATE RESULTS")
print("=" * 72)

display_columns = [
    "method",
    "channels_evaluated",
    "evaluation_samples",
    "pooled_precision",
    "pooled_recall",
    "pooled_f1",
    "macro_average_precision",
    "macro_balanced_accuracy",
    "runtime_seconds",
]

print(
    summary_df[display_columns].to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}",
    )
)

print("\nSaved results:")
print(summary_path)
print(per_channel_path)
print(predictions_path)
print(config_path)
