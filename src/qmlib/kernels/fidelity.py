import numpy as np
import pennylane as qml

from .base import QuantumKernel


class FidelityKernel(QuantumKernel):
    """
    Quantum fidelity kernel.

    K(x, y) = |<phi(x)|phi(y)>|^2
    """

    def __init__(self, feature_map):
        self.feature_map = feature_map

        self.device = qml.device(
            "default.qubit",
            wires=feature_map.n_qubits
        )

    def _state(self, x):
        @qml.qnode(self.device)
        def circuit():
            self.feature_map.circuit(x)
            return qml.state()

        return circuit()

    def compute(self, X, Y=None):
        X = np.asarray(X)

        if Y is None:
            Y = X
        else:
            Y = np.asarray(Y)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")

        if Y.ndim != 2:
            raise ValueError("Y must be a 2D array.")

        if X.shape[1] != self.feature_map.n_qubits:
            raise ValueError(
                f"X has {X.shape[1]} features, "
                f"but the feature map expects "
                f"{self.feature_map.n_qubits}."
            )

        if Y.shape[1] != self.feature_map.n_qubits:
            raise ValueError(
                f"Y has {Y.shape[1]} features, "
                f"but the feature map expects "
                f"{self.feature_map.n_qubits}."
            )

        states_X = [self._state(x) for x in X]
        states_Y = [self._state(y) for y in Y]

        kernel_matrix = np.zeros(
            (len(X), len(Y)),
            dtype=float
        )

        for i, state_x in enumerate(states_X):
            for j, state_y in enumerate(states_Y):
                overlap = np.vdot(state_x, state_y)

                kernel_matrix[i, j] = (
                    np.abs(overlap) ** 2
                )

        return kernel_matrix