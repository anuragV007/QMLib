
import csv
import json
import time
from pathlib import Path

from sklearn.datasets import make_moons
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from qmlib.feature_maps import ZFeatureMap, ZZFeatureMap
from qmlib.kernels import FidelityKernel
from qmlib.models import QuantumClassifier
from qmlib.pipeline import QMLPipeline


def build_models():
    """Create classical and quantum models for comparison."""

    models = [
        (
            "Classical RBF SVM",
            "classical",
            Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", SVC(kernel="rbf", C=1.0)),
            ]),
        ),
        (
            "Logistic Regression",
            "classical",
            Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                )),
            ]),
        ),
        (
            "Random Forest",
            "classical",
            Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", RandomForestClassifier(
                    n_estimators=100,
                    random_state=42,
                )),
            ]),
        ),
    ]

    feature_maps = [
        ("Quantum Z", ZFeatureMap(n_qubits=2)),
        ("Quantum ZZ", ZZFeatureMap(n_qubits=2)),
    ]

    for name, feature_map in feature_maps:
        kernel = FidelityKernel(feature_map=feature_map)
        classifier = QuantumClassifier(kernel=kernel)

        model = QMLPipeline(
            steps=[("scaler", StandardScaler())],
            estimator=classifier,
        )

        models.append((name, "quantum_simulator", model))

    return models


def main():
    # Small dataset for checking the benchmark pipeline.
    X, y = make_moons(
        n_samples=80,
        noise=0.22,
        random_state=42,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for name, family, model in build_models():
        print(f"\nRunning: {name}")

        start = time.perf_counter()
        model.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - start

        start = time.perf_counter()
        predictions = model.predict(X_test)
        predict_seconds = time.perf_counter() - start

        row = {
            "model": name,
            "family": family,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "n_features": X_train.shape[1],
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision": float(
                precision_score(y_test, predictions, zero_division=0)
            ),
            "recall": float(
                recall_score(y_test, predictions, zero_division=0)
            ),
            "f1": float(
                f1_score(y_test, predictions, zero_division=0)
            ),
            "fit_seconds": round(fit_seconds, 6),
            "predict_seconds": round(predict_seconds, 6),
        }

        results.append(row)

        print(
            f"  Accuracy: {row['accuracy']:.3f} | "
            f"F1: {row['f1']:.3f} | "
            f"Fit: {fit_seconds:.3f}s | "
            f"Predict: {predict_seconds:.3f}s"
        )

    csv_path = output_dir / "smoke_benchmark.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(results[0].keys()),
        )
        writer.writeheader()
        writer.writerows(results)

    config = {
        "dataset": "sklearn.datasets.make_moons",
        "n_samples": 80,
        "noise": 0.22,
        "random_state": 42,
        "test_size": 0.25,
        "split_random_state": 42,
        "quantum_feature_maps": ["ZFeatureMap", "ZZFeatureMap"],
        "quantum_qubits": 2,
        "quantum_execution": "Local simulator through QMLib backend",
        "purpose": (
            "Smoke test for benchmarking infrastructure; "
            "not evidence of quantum advantage."
        ),
    }

    config_path = output_dir / "smoke_benchmark_config.json"

    with config_path.open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)

    print("\nBenchmark completed.")
    print(f"Results: {csv_path}")
    print(f"Configuration: {config_path}")


if __name__ == "__main__":
    main()
