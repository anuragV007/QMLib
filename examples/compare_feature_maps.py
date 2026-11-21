"""Compare interchangeable QMLib feature maps on a small binary task."""

import numpy as np
from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from qmlib.feature_maps import ZFeatureMap, ZZFeatureMap, PauliFeatureMap
from qmlib.kernels import FidelityKernel
from qmlib.models import QuantumClassifier


def run_experiment(feature_map, X_train, X_test, y_train, y_test):
    """Run the same classifier workflow with a supplied feature map."""
    kernel = FidelityKernel(feature_map=feature_map)
    model = QuantumClassifier(kernel=kernel, C=1.0)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    return {
        "accuracy": accuracy_score(y_test, predictions),
        "f1": f1_score(y_test, predictions, zero_division=0),
    }


def main():
    X, y = make_moons(n_samples=80, noise=0.20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    feature_maps = {
        "ZFeatureMap": ZFeatureMap(n_qubits=2, reps=2),
        "ZZFeatureMap": ZZFeatureMap(n_qubits=2, reps=2),
        "PauliFeatureMap": PauliFeatureMap(
            n_qubits=2, reps=2, paulis=["X", "Z"]
        ),
    }

    print("QMLib feature-map interchangeability example")
    for name, feature_map in feature_maps.items():
        results = run_experiment(
            feature_map, X_train, X_test, y_train, y_test
        )
        print(
            f"{name:20s} "
            f"accuracy={results['accuracy']:.3f} "
            f"f1={results['f1']:.3f}"
        )


if __name__ == "__main__":
    main()
