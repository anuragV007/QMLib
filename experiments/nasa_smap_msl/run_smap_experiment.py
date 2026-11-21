
import argparse
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
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from qmlib.feature_maps import ZFeatureMap
from qmlib.kernels import FidelityKernel
from qmlib.models.quantum_anomaly_detector import (
    QuantumAnomalyDetector,
)


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "NASA Data"
TRAIN_DIR = DATA_ROOT / "data" / "data" / "train"
TEST_DIR = DATA_ROOT / "data" / "data" / "test"
METADATA_FILE = DATA_ROOT / "labeled_anomalies.csv"

RESULTS_DIR = (
    ROOT / "experiments" / "nasa_smap_msl" / "results"
)


def parse_intervals(value):
    """Parse anomaly intervals from the NASA metadata CSV."""
    if pd.isna(value):
        return []

    parsed = ast.literal_eval(str(value))

    if not parsed:
        return []

    # Also support a single interval represented as [start, end].
    if (
        isinstance(parsed, (list, tuple))
        and len(parsed) == 2
        and all(np.isscalar(v) for v in parsed)
    ):
        return [(int(parsed[0]), int(parsed[1]))]

    intervals = []

    for interval in parsed:
        if len(interval) >= 2:
            intervals.append(
                (int(interval[0]), int(interval[1]))
            )

    return intervals


def load_smap_sequences():
    """
    Load SMAP train/test sequences and construct test labels.

    NASA anomaly intervals use zero-based inclusive endpoints.
    Duplicate metadata rows for a channel are combined.
    """
    metadata = pd.read_csv(METADATA_FILE)

    smap = metadata[
        metadata["spacecraft"].astype(str).str.upper() == "SMAP"
    ].copy()

    sequences = []
    missing = []

    for channel_id, group in smap.groupby("chan_id"):
        channel_id = str(channel_id)

        train_path = TRAIN_DIR / f"{channel_id}.npy"
        test_path = TEST_DIR / f"{channel_id}.npy"

        if not train_path.exists() or not test_path.exists():
            missing.append(channel_id)
            continue

        train_X = np.asarray(
            np.load(train_path), dtype=np.float64
        )
        test_X = np.asarray(
            np.load(test_path), dtype=np.float64
        )

        if train_X.ndim != 2 or test_X.ndim != 2:
            raise ValueError(
                f"{channel_id}: expected 2D train/test arrays."
            )

        if train_X.shape[1] != test_X.shape[1]:
            raise ValueError(
                f"{channel_id}: train/test feature counts differ."
            )

        if not (
            np.isfinite(train_X).all()
            and np.isfinite(test_X).all()
        ):
            raise ValueError(
                f"{channel_id}: non-finite values detected."
            )

        y_test = np.zeros(len(test_X), dtype=np.int8)

        for _, row in group.iterrows():
            for start, end in parse_intervals(
                row["anomaly_sequences"]
            ):
                if start < 0 or end < start:
                    raise ValueError(
                        f"{channel_id}: invalid interval "
                        f"({start}, {end})."
                    )

                if start >= len(y_test):
                    continue

                # Inclusive endpoint; clamp to array length.
                end = min(end, len(y_test) - 1)
                y_test[start:end + 1] = 1

        sequences.append(
            (channel_id, train_X, test_X, y_test)
        )

    if not sequences:
        raise RuntimeError(
            "No matching labelled SMAP train/test sequences found. "
            "Check NASA Data paths and channel filenames."
        )

    if missing:
        print("Skipped channels with missing files:", missing)

    return sequences


def sample_rows(X, maximum, rng):
    """Select at most 'maximum' rows without replacement."""
    if len(X) <= maximum:
        return X

    indices = rng.choice(len(X), size=maximum, replace=False)
    return X[indices]


def make_train_partitions(sequences, rng):
    """
    Construct temporally separated reference and calibration pools.

    The first 65% of each training sequence supplies reference/PCA
    candidates. The final 25% supplies threshold calibration.
    The intervening 10% is excluded as a temporal gap.
    """
    reference_parts = []
    calibration_parts = []

    for _, train_X, _, _ in sequences:
        n = len(train_X)

        reference_end = int(0.65 * n)
        calibration_start = int(0.75 * n)

        reference_segment = train_X[:reference_end]
        calibration_segment = train_X[calibration_start:]

        if len(reference_segment) == 0:
            continue

        # Bound PCA fitting cost while representing many channels.
        reference_parts.append(
            sample_rows(reference_segment, 200, rng)
        )

        if len(calibration_segment):
            calibration_parts.append(calibration_segment)

    if not reference_parts or not calibration_parts:
        raise RuntimeError(
            "Insufficient training data for reference/calibration."
        )

    reference_pool = np.vstack(reference_parts)
    calibration_pool = np.vstack(calibration_parts)

    return reference_pool, calibration_pool


def collect_test_data(sequences, maximum, rng):
    """
    Pool labelled test timesteps and sample uniformly over timesteps.

    This is a bounded smoke experiment, not a full-sequence evaluation.
    """
    all_X = np.vstack([item[2] for item in sequences])
    all_y = np.concatenate([item[3] for item in sequences])

    if len(all_X) != len(all_y):
        raise RuntimeError("Test samples and labels do not align.")

    indices = rng.choice(
        len(all_X),
        size=min(maximum, len(all_X)),
        replace=False,
    )

    return all_X[indices], all_y[indices]


def evaluate_scores(y_true, scores, threshold):
    predictions = (scores > threshold).astype(np.int8)

    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, predictions)
        ),
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
            f1_score(y_true, predictions, zero_division=0)
        ),
        "average_precision": float(
            average_precision_score(y_true, scores)
        ),
        "threshold": float(threshold),
        "predicted_anomalies": int(predictions.sum()),
        "actual_anomalies": int(y_true.sum()),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Bounded NASA SMAP quantum-kernel experiment."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reference-size", type=int, default=64)
    parser.add_argument("--calibration-size", type=int, default=512)
    parser.add_argument("--max-test-samples", type=int, default=1000)
    parser.add_argument("--threshold-quantile", type=float, default=0.99)
    args = parser.parse_args()

    if args.reference_size < 2:
        parser.error("--reference-size must be at least 2.")
    if args.calibration_size < 2:
        parser.error("--calibration-size must be at least 2.")
    if args.max_test_samples < 2:
        parser.error("--max-test-samples must be at least 2.")
    if not 0.0 < args.threshold_quantile < 1.0:
        parser.error("--threshold-quantile must be between 0 and 1.")

    rng = np.random.default_rng(args.seed)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading labelled SMAP sequences...")
    sequences = load_smap_sequences()

    total_test_steps = sum(len(item[2]) for item in sequences)
    total_anomalies = sum(int(item[3].sum()) for item in sequences)

    print(f"SMAP channels loaded: {len(sequences)}")
    print(f"Full test timesteps: {total_test_steps:,}")
    print(f"Labelled anomalous timesteps: {total_anomalies:,}")

    reference_pool, calibration_pool = make_train_partitions(
        sequences, rng
    )

    # The test set is sampled before transformation; no test fitting.
    test_raw, y_test = collect_test_data(
        sequences, args.max_test_samples, rng
    )

    # All preprocessing is fitted on the reference portion of training.
    scaler = StandardScaler()
    reference_scaled = scaler.fit_transform(reference_pool)

    pca = PCA(n_components=2, random_state=args.seed)
    reference_reduced = pca.fit_transform(reference_scaled)

    calibration_reduced = pca.transform(
        scaler.transform(
            sample_rows(
                calibration_pool,
                args.calibration_size,
                rng,
            )
        )
    )

    test_reduced = pca.transform(
        scaler.transform(test_raw)
    )

    # Use the same reference points for quantum and classical methods.
    reference = sample_rows(
        reference_reduced, args.reference_size, rng
    )

    print(f"Original features: {reference_pool.shape[1]}")
    print(f"Quantum features/qubits: {reference.shape[1]}")
    print(f"Reference samples: {len(reference)}")
    print(f"Calibration samples: {len(calibration_reduced)}")
    print(f"Evaluation samples: {len(test_reduced)}")
    print(f"Evaluation anomaly fraction: {y_test.mean():.4f}")
    print(
        "PCA explained variance ratio:",
        pca.explained_variance_ratio_.round(4).tolist(),
    )

    if len(np.unique(y_test)) < 2:
        raise RuntimeError(
            "The sampled test set contains only one class. "
            "Increase --max-test-samples and rerun."
        )

    metrics = []

    # ---------------------------------------------------------
    # Quantum-kernel novelty detector
    # ---------------------------------------------------------
    print("\nRunning quantum-kernel detector...")
    quantum_start = time.perf_counter()

    feature_map = ZFeatureMap(n_qubits=2, reps=1)
    kernel = FidelityKernel(feature_map=feature_map)

    detector = QuantumAnomalyDetector(
        kernel=kernel,
        threshold_quantile=args.threshold_quantile,
    )

    detector.fit(reference, calibration_reduced)
    quantum_scores = detector.score_samples(test_reduced)

    quantum_seconds = time.perf_counter() - quantum_start

    quantum_metrics = evaluate_scores(
        y_test, quantum_scores, detector.threshold_
    )
    quantum_metrics.update({
        "model": "Quantum fidelity kernel",
        "seconds": quantum_seconds,
    })
    metrics.append(quantum_metrics)

    # ---------------------------------------------------------
    # Classical nearest-reference-distance baseline
    # ---------------------------------------------------------
    print("Running classical nearest-neighbour baseline...")
    classical_start = time.perf_counter()

    nearest = NearestNeighbors(n_neighbors=1)
    nearest.fit(reference)

    calibration_distances = nearest.kneighbors(
        calibration_reduced, return_distance=True
    )[0][:, 0]

    classical_threshold = float(
        np.quantile(
            calibration_distances,
            args.threshold_quantile,
        )
    )

    test_distances = nearest.kneighbors(
        test_reduced, return_distance=True
    )[0][:, 0]

    classical_seconds = time.perf_counter() - classical_start

    classical_metrics = evaluate_scores(
        y_test, test_distances, classical_threshold
    )
    classical_metrics.update({
        "model": "Classical nearest-reference distance",
        "seconds": classical_seconds,
    })
    metrics.append(classical_metrics)

    # Save metrics, predictions, and experiment configuration.
    metrics_df = pd.DataFrame(metrics)
    metrics_path = RESULTS_DIR / "smap_experiment_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    predictions_df = pd.DataFrame({
        "y_true": y_test,
        "quantum_score": quantum_scores,
        "quantum_prediction": (
            quantum_scores > detector.threshold_
        ).astype(np.int8),
        "classical_score": test_distances,
        "classical_prediction": (
            test_distances > classical_threshold
        ).astype(np.int8),
    })

    predictions_path = (
        RESULTS_DIR / "smap_experiment_predictions.csv"
    )
    predictions_df.to_csv(predictions_path, index=False)

    config = {
        "dataset": "NASA SMAP",
        "seed": args.seed,
        "channels": len(sequences),
        "full_test_timesteps": total_test_steps,
        "full_test_anomalous_timesteps": total_anomalies,
        "reference_size": len(reference),
        "calibration_size": len(calibration_reduced),
        "evaluation_size": len(y_test),
        "threshold_quantile": args.threshold_quantile,
        "feature_map": feature_map.get_config(),
        "pca_explained_variance_ratio": (
            pca.explained_variance_ratio_.tolist()
        ),
        "note": (
            "Bounded random test sample; not a full-sequence "
            "benchmark. Training arrays are treated as normal."
        ),
    }

    config_path = RESULTS_DIR / "smap_experiment_config.json"
    with config_path.open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)

    print("\nResults:")
    print(
        metrics_df[
            [
                "model",
                "precision",
                "recall",
                "f1",
                "average_precision",
                "seconds",
            ]
        ].to_string(index=False)
    )

    print("\nSaved:")
    print(metrics_path)
    print(predictions_path)
    print(config_path)


if __name__ == "__main__":
    main()
