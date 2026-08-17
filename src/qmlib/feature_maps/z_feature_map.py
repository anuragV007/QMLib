import pennylane as qml

from .base import FeatureMap


class ZFeatureMap(FeatureMap):
    """Single-qubit Z-based quantum feature map."""

    def circuit(self, x):
        if len(x) != self.n_qubits:
            raise ValueError(
                f"Expected {self.n_qubits} features, got {len(x)}."
            )

        for _ in range(self.reps):
            for i, value in enumerate(x):
                qml.Hadamard(wires=i)
                qml.RZ(value, wires=i)