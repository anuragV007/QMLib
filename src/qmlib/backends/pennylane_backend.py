import pennylane as qml

from .base import Backend


class PennyLaneBackend(Backend):
    """
    PennyLane execution backend.

    Parameters
    ----------
    device_name : str
        PennyLane device name.
    """

    def __init__(
        self,
        device_name: str = "default.qubit"
    ):

        if not isinstance(device_name, str):
            raise TypeError(
                "device_name must be a string."
            )

        self.device_name = device_name

    def create_device(
        self,
        n_qubits: int
    ):

        if not isinstance(n_qubits, int):
            raise TypeError(
                "n_qubits must be an integer."
            )

        if n_qubits <= 0:
            raise ValueError(
                "n_qubits must be greater than 0."
            )

        return qml.device(
            self.device_name,
            wires=n_qubits
        )