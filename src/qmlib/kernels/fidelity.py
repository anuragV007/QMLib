import numpy as np
import pennylane as qml

from .base import QuantumKernel
from qmlib.backends import PennyLaneBackend


class FidelityKernel(QuantumKernel):
    """
    Quantum fidelity kernel.

    K(x, y) = |<phi(x) | phi(y)>|^2

    Parameters
    ----------
    feature_map : FeatureMap
        Feature map used to encode the data.

    backend : Backend, optional
        Quantum execution backend.
    """

    def __init__(
        self,
        feature_map,
        backend=None
    ):

        self.feature_map = feature_map

        if backend is None:
            backend = PennyLaneBackend()

        self.backend = backend

        self.device = self.backend.create_device(
            feature_map.n_qubits
        )

    def _state(self, x):
        """
        Generate the quantum state for one sample.
        """

        @qml.qnode(self.device)
        def circuit():

            self.feature_map.circuit(x)

            return qml.state()

        return circuit()

    def compute(self, X, Y=None):
        """
        Compute the fidelity kernel matrix.

        If Y is None:

            K = K(X, X)

        Otherwise:

            K = K(X, Y)
        """

        X = np.asarray(X)

        if Y is None:
            Y = X
        else:
            Y = np.asarray(Y)

        if X.ndim != 2:
            raise ValueError(
                "X must be a 2D array."
            )

        if Y.ndim != 2:
            raise ValueError(
                "Y must be a 2D array."
            )

        expected_features = self.feature_map.n_qubits

        if X.shape[1] != expected_features:
            raise ValueError(
                f"X has {X.shape[1]} features, "
                f"but the feature map expects "
                f"{expected_features}."
            )

        if Y.shape[1] != expected_features:
            raise ValueError(
                f"Y has {Y.shape[1]} features, "
                f"but the feature map expects "
                f"{expected_features}."
            )

        states_X = [
            self._state(x)
            for x in X
        ]

        states_Y = [
            self._state(y)
            for y in Y
        ]

        kernel_matrix = np.zeros(
            (len(X), len(Y)),
            dtype=float
        )

        for i, state_x in enumerate(states_X):

            for j, state_y in enumerate(states_Y):

                overlap = np.vdot(
                    state_x,
                    state_y
                )

                kernel_matrix[i, j] = (
                    np.abs(overlap) ** 2
                )

        return kernel_matrix