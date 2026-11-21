"""Minimal quantum-kernel regression example."""

import numpy as np
from qmlib.feature_maps import ZZFeatureMap
from qmlib.kernels import FidelityKernel
from qmlib.models import QuantumKernelRidge


def main():
    rng = np.random.default_rng(7)
    X = rng.uniform(-1.0, 1.0, size=(16, 2))
    y = np.sin(X[:, 0]) + 0.5 * X[:, 1]

    feature_map = ZZFeatureMap(n_qubits=2, reps=1)
    kernel = FidelityKernel(feature_map=feature_map)
    model = QuantumKernelRidge(kernel=kernel, alpha=0.05)

    model.fit(X, y)
    predictions = model.predict(X)

    print("Training R²:", round(model.score(X, y), 4))
    print("First five predictions:", np.round(predictions[:5], 4))


if __name__ == "__main__":
    main()
