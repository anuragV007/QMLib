import pennylane as qml

from .base import FeatureMap


class PauliFeatureMap(FeatureMap):
    """
    Pauli-rotation based quantum feature map.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.

    reps : int
        Number of repetitions.

    paulis : list[str]
        Pauli rotations to apply.

        Supported:
        - "X"
        - "Y"
        - "Z"
    """

    def __init__(
        self,
        n_qubits: int,
        reps: int = 1,
        paulis=None
    ):

        super().__init__(
            n_qubits=n_qubits,
            reps=reps
        )

        if paulis is None:
            paulis = ["Z"]

        if not isinstance(paulis, list):
            raise TypeError(
                "paulis must be a list."
            )

        supported = {"X", "Y", "Z"}

        for pauli in paulis:

            if pauli not in supported:
                raise ValueError(
                    f"Unsupported Pauli operator: {pauli}. "
                    f"Supported operators are {supported}."
                )

        if len(paulis) == 0:
            raise ValueError(
                "paulis must contain at least one operator."
            )

        self.paulis = paulis

    def circuit(self, x):

        if len(x) != self.n_qubits:
            raise ValueError(
                f"Expected {self.n_qubits} features, "
                f"got {len(x)}."
            )

        for _ in range(self.reps):

            for i, value in enumerate(x):

                for pauli in self.paulis:

                    if pauli == "X":

                        qml.RX(
                            value,
                            wires=i
                        )

                    elif pauli == "Y":

                        qml.RY(
                            value,
                            wires=i
                        )

                    elif pauli == "Z":

                        qml.RZ(
                            value,
                            wires=i
                        )

    def get_config(self):

        config = super().get_config()

        config.update({
            "name": "PauliFeatureMap",
            "paulis": self.paulis,
        })

        return config