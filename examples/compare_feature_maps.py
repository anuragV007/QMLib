import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

from qmlib.feature_maps import (
    ZFeatureMap,
    ZZFeatureMap,
)

from qmlib.kernels import FidelityKernel
from qmlib.models import QuantumClassifier


def run_experiment(feature_map):
    iris = load_iris()

    X = iris.data
    y = iris.target

    mask = y < 2

    X = X[mask]
    y = y[mask]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    kernel = FidelityKernel(
        feature_map=feature_map
    )

    model = QuantumClassifier(
        feature_map=feature_map,
        kernel=kernel
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    return {
        "accuracy": accuracy_score(
            y_test,
            predictions
        ),
        "f1": f1_score(
            y_test,
            predictions
        ),
    }


experiments = {
    "ZFeatureMap": ZFeatureMap(
        n_qubits=4,
        reps=2
    ),

    "ZZFeatureMap": ZZFeatureMap(
        n_qubits=4,
        reps=2
    ),
}


print("\nQMLib Feature Map Comparison")
print("============================")

for name, feature_map in experiments.items():

    print(f"\nRunning: {name}")

    results = run_experiment(
        feature_map
    )

    print(
        f"Accuracy: {results['accuracy']:.4f}"
    )

    print(
        f"F1:       {results['f1']:.4f}"
    )