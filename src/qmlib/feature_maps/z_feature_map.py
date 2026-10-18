import pennylane as qml

from .base import FeatureMap


class ZFeatureMap(FeatureMap):
    """
    Z-based quantum feature map.

    Each classical feature is encoded using a Hadamard
    gate followed by an RZ rotation.
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

    def get_config(self):

        config = super().get_config()

        config["name"] = "ZFeatureMap"

        return config