
import ast
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler

from qmlib.feature_maps import ZFeatureMap, ZZFeatureMap
from qmlib.kernels import FidelityKernel


# ---------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------

SEED = 42
N_QUBITS = 2
N_REFERENCE = 64
N_CALIBRATION = 512
N_EVALUATION = 1000
MAX_REFERENCE_CANDIDATES_PER_CHANNEL = 200
THRESHOLD_QUANTILE = 0.99

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "NASA Data"

METADATA_PATH = DATA_ROOT / "labeled_anomalies.csv"
TRAIN_DIR = DATA_ROOT / "data" / "data" / "train"
TEST_DIR = DATA_ROOT / "data" / "data" / "test"

RESULTS_DIR = ROOT / "experiments" / "nasa_smap_msl" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(SEED)


# ---------------------------------------------------------
# 2. Helpers
# ---------------------------------------------------------

def parse_anomaly_sequences(value):
    """Parse inclusive anomaly intervals from the metadata."""
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


def sample_rows(X, n, random_generator):
    """Sample up to n rows without replacement."""
    if len(X) <= n:
        return X.copy()

    indices = random_generator.choice(
        len(X), size=n, replace=False
    )
    return X[indices]


def anomaly_labels(n_rows, intervals):
    """Build zero-based labels; both interval endpoints are inclusive."""
    y = np.zeros(n_rows, dtype=np.int8)

    for start, end in intervals:
        start = max(0, start)
        end = min(n_rows - 1, end)

        if end >= start:
            y[start:end + 1] = 1

    return y


def evaluate_scores(y_true, scores, threshold):
    """Evaluate scores using a threshold fixed from calibration data."""
    y_pred = (scores >= threshold).astype(np.int8)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "precision": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, zero_division=0)
        ),
        "f1": float(
            f1_score(y_true, y_pred, zero_division=0)
        ),
        "average_precision": float(
            average_precision_score(y_true, scores)
        ),
        "threshold": float(threshold),
        "predicted_anomalies": int(y_pred.sum()),
        "actual_anomalies": int(y_true.sum()),
    }


# ---------------------------------------------------------
# 3. Load SMAP metadata and sequences
# ---------------------------------------------------------

if not METADATA_PATH.exists():
    raise FileNotFoundError(
        f"Metadata file not found: {METADATA_PATH}"
    )

metadata = pd.read_csv(METADATA_PATH)

required_columns = {
    "chan_id",
    "spacecraft",
    "anomaly_sequences",
}

missing = required_columns - set(metadata.columns)

if missing:
    raise ValueError(
        f"Metadata is missing required columns: {sorted(missing)}"
    )

smap_metadata = metadata[
    metadata["spacecraft"].astype(str).str.upper() == "SMAP"
].copy()

# Multiple metadata rows for a channel may describe separate events.
intervals_by_channel = {}

for _, row in smap_metadata.iterrows():
    channel = str(row["chan_id"])
    intervals = parse_anomaly_sequences(row["anomaly_sequences"])

    intervals_by_channel.setdefault(channel, []).extend(intervals)

reference_candidates = []
calibration_candidates = []
test_arrays = []
test_labels = []
loaded_channels = []

for channel in sorted(intervals_by_channel):
    train_path = TRAIN_DIR / f"{channel}.npy"
    test_path = TEST_DIR / f"{channel}.npy"

    if not train_path.exists() or not test_path.exists():
        print(f"Skipping {channel}: train/test file missing")
        continue

    X_train = np.asarray(np.load(train_path), dtype=np.float64)
    X_test = np.asarray(np.load(test_path), dtype=np.float64)

    if X_train.ndim != 2 or X_test.ndim != 2:
        print(f"Skipping {channel}: expected 2D arrays")
        continue

    if X_train.shape[1] != X_test.shape[1]:
        print(f"Skipping {channel}: train/test feature mismatch")
        continue

    # Keep row indices aligned with the anomaly labels.
    valid_train = np.isfinite(X_train).all(axis=1)
    valid_test = np.isfinite(X_test).all(axis=1)

    X_train = X_train[valid_train]
    X_test_original = np.asarray(np.load(test_path), dtype=np.float64)
    y_test_original = anomaly_labels(
        len(X_test_original),
        intervals_by_channel[channel],
    )

    X_test = X_test_original[valid_test]
    y_test_channel = y_test_original[valid_test]

    if len(X_train) < 10 or len(X_test) == 0:
        print(f"Skipping {channel}: insufficient valid rows")
        continue

    # Training data is assumed to be predominantly normal.
    # Reference candidates come from the first 65% of training.
    reference_end = int(0.65 * len(X_train))

    # Leave a 10% temporal gap; calibration uses the final 25%.
    calibration_start = int(0.75 * len(X_train))

    reference_region = X_train[:reference_end]
    calibration_region = X_train[calibration_start:]

    if len(reference_region) == 0 or len(calibration_region) == 0:
        print(f"Skipping {channel}: empty temporal partition")
        continue

    candidates = sample_rows(
        reference_region,
        MAX_REFERENCE_CANDIDATES_PER_CHANNEL,
        rng,
    )

    reference_candidates.append(candidates)

    # Bound the contribution of each channel to the calibration pool.
    calibration_candidates.append(
        sample_rows(calibration_region, 200, rng)
    )

    test_arrays.append(X_test)
    test_labels.append(y_test_channel)
    loaded_channels.append(channel)

if not loaded_channels:
    raise RuntimeError(
        "No SMAP channels were loaded. Check NASA Data paths and metadata."
    )

reference_candidate_pool = np.concatenate(reference_candidates)
calibration_pool = np.concatenate(calibration_candidates)
X_test_all = np.concatenate(test_arrays)
y_test_all = np.concatenate(test_labels)

# Select one shared set of samples for both feature maps.
X_reference_raw = sample_rows(
    reference_candidate_pool, N_REFERENCE, rng
)

X_calibration_raw = sample_rows(
    calibration_pool, N_CALIBRATION, rng
)

if len(X_calibration_raw) < 2:
    raise RuntimeError("Not enough calibration samples.")

n_evaluation = min(N_EVALUATION, len(X_test_all))
evaluation_indices = rng.choice(
    len(X_test_all),
    size=n_evaluation,
    replace=False,
)

X_evaluation_raw = X_test_all[evaluation_indices]
y_evaluation = y_test_all[evaluation_indices]

print(f"SMAP channels loaded: {len(loaded_channels)}")
print(f"Full test timesteps: {len(X_test_all):,}")
print(f"Full test anomalous timesteps: {int(y_test_all.sum()):,}")
print(f"Original features: {X_reference_raw.shape[1]}")
print(f"Reference samples: {len(X_reference_raw)}")
print(f"Calibration samples: {len(X_calibration_raw)}")
print(f"Evaluation samples: {len(X_evaluation_raw)}")
print(
    "Evaluation anomaly fraction: "
    f"{y_evaluation.mean():.4f}"
)


# ---------------------------------------------------------
# 4. Shared preprocessing: scaler and PCA fit on reference
# ---------------------------------------------------------

if X_reference_raw.shape[1] < N_QUBITS:
    raise ValueError(
        f"Need at least {N_QUBITS} input features."
    )

scaler = StandardScaler()
X_reference_scaled = scaler.fit_transform(X_reference_raw)

pca = PCA(n_components=N_QUBITS)
X_reference_pca = pca.fit_transform(X_reference_scaled)

X_calibration_pca = pca.transform(
    scaler.transform(X_calibration_raw)
)

X_evaluation_pca = pca.transform(
    scaler.transform(X_evaluation_raw)
)

print(
    "PCA explained variance ratio:",
    np.round(pca.explained_variance_ratio_, 4),
)


# ---------------------------------------------------------
# 5. Bounded angle encoding
# ---------------------------------------------------------

def encode_angles(X):
    """
    Map real-valued PCA features into (-pi, pi).

    Apply the same deterministic transformation to reference,
    calibration, and evaluation data.
    """
    return 2.0 * np.arctan(X)


X_reference = encode_angles(X_reference_pca)
X_calibration = encode_angles(X_calibration_pca)
X_evaluation = encode_angles(X_evaluation_pca)


# ---------------------------------------------------------
# 6. Quantum feature-map comparison
# ---------------------------------------------------------

feature_maps = {
    "ZFeatureMap": ZFeatureMap(n_qubits=N_QUBITS, reps=1),
    "ZZFeatureMap": ZZFeatureMap(n_qubits=N_QUBITS, reps=1),
}

metrics_rows = []
prediction_frames = []

for map_name, feature_map in feature_maps.items():
    print(f"\nRunning {map_name}...")
    start_time = time.perf_counter()

    kernel = FidelityKernel(feature_map=feature_map)

    # Similarity between calibration/evaluation samples and
    # the same reference set.
    K_calibration = kernel.compute(
        X_calibration, X_reference
    )
    K_evaluation = kernel.compute(
        X_evaluation, X_reference
    )

    # Novelty score: high score means low similarity to references.
    calibration_scores = 1.0 - np.max(
        K_calibration, axis=1
    )
    evaluation_scores = 1.0 - np.max(
        K_evaluation, axis=1
    )

    # Threshold comes only from the calibration scores.
    threshold = float(
        np.quantile(calibration_scores, THRESHOLD_QUANTILE)
    )

    metrics = evaluate_scores(
        y_evaluation,
        evaluation_scores,
        threshold,
    )

    elapsed = time.perf_counter() - start_time

    metrics_rows.append({
        "model": map_name,
        "angle_encoding": "2*arctan(x)",
        "n_qubits": N_QUBITS,
        "reps": 1,
        "threshold_quantile": THRESHOLD_QUANTILE,
        **metrics,
        "seconds": elapsed,
    })

    prediction_frames.append(pd.DataFrame({
        "model": map_name,
        "sample_index": evaluation_indices,
        "y_true": y_evaluation,
        "anomaly_score": evaluation_scores,
        "threshold": threshold,
        "y_pred": (
            evaluation_scores >= threshold
        ).astype(np.int8),
    }))

    print(f"Threshold: {threshold:.6f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Balanced accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1: {metrics['f1']:.4f}")
    print(f"Average precision: {metrics['average_precision']:.4f}")
    print(f"Predicted anomalies: {metrics['predicted_anomalies']}")
    print(f"Runtime: {elapsed:.3f} seconds")


# ---------------------------------------------------------
# 7. Save results without overwriting v1
# ---------------------------------------------------------

metrics_df = pd.DataFrame(metrics_rows)
predictions_df = pd.concat(
    prediction_frames, ignore_index=True
)

metrics_path = RESULTS_DIR / "smap_experiment_v2_metrics.csv"
predictions_path = RESULTS_DIR / "smap_experiment_v2_predictions.csv"
config_path = RESULTS_DIR / "smap_experiment_v2_config.json"

metrics_df.to_csv(metrics_path, index=False)
predictions_df.to_csv(predictions_path, index=False)

config = {
    "experiment": "SMAP bounded-angle Z vs ZZ feature maps",
    "seed": SEED,
    "channels": loaded_channels,
    "n_channels": len(loaded_channels),
    "n_qubits": N_QUBITS,
    "feature_maps": list(feature_maps.keys()),
    "angle_encoding": "2 * arctan(x)",
    "reference_samples": len(X_reference),
    "calibration_samples": len(X_calibration),
    "evaluation_samples": len(X_evaluation),
    "threshold_quantile": THRESHOLD_QUANTILE,
    "pca_explained_variance_ratio": (
        pca.explained_variance_ratio_.tolist()
    ),
    "evaluation_anomaly_fraction": float(y_evaluation.mean()),
    "notes": [
        "Preliminary pooled-channel experiment.",
        "Scaler and PCA fitted on reference samples only.",
        "Thresholds determined from calibration scores only.",
        "The same sampled reference, calibration, and evaluation "
        "rows are used for both feature maps.",
        "No quantum advantage is inferred from this experiment."
    ],
}

with open(config_path, "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2)

print("\n--- Comparison ---")
print(
    metrics_df[
        [
            "model",
            "accuracy",
            "balanced_accuracy",
            "precision",
            "recall",
            "f1",
            "average_precision",
            "predicted_anomalies",
            "seconds",
        ]
    ].to_string(index=False)
)

print("\nSaved:")
print(metrics_path)
print(predictions_path)
print(config_path)
