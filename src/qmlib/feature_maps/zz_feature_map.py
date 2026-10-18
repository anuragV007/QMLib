import pennylane as qml

from .base import FeatureMap


class ZZFeatureMap(FeatureMap):
    """
    ZZ-based quantum feature map.

    In addition to single-qubit Z rotations, this feature map
    introduces pairwise interactions between neighboring features.
    """

    def circuit(self, x):

        if len(x) != self.n_qubits:
            raise ValueError(
                f"Expected {self.n_qubits} features, "
                f"got {len(x)}."
            )

        for _ in range(self.reps):

            for i, value in enumerate(x):

                qml.Hadamard(
                    wires=i
                )

                qml.RZ(
                    value,
                    wires=i
                )

            for i in range(self.n_qubits - 1):

                qml.CNOT(
                    wires=[i, i + 1]
                )

                qml.RZ(
                    x[i] * x[i + 1],
                    wires=i + 1
                )

                qml.CNOT(
                    wires=[i, i + 1]
                )

    def get_config(self):

        config = super().get_config()

        config["name"] = "ZZFeatureMap"

        return config