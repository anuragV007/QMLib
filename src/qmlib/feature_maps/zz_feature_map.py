import pennylane as qml

from .base import FeatureMap


class ZZFeatureMap(FeatureMap):
    """
    Z-based feature map with pairwise ZZ interactions.
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

            for i in range(
                self.n_qubits - 1
            ):

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

        config.update({
            "name": "ZZFeatureMap"
        })

        return config