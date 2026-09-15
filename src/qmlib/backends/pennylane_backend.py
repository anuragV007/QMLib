import pennylane as qml

from .base import Backend


class PennyLaneBackend(Backend):
    """
    PennyLane-based quantum execution backend.

    Parameters
    ----------
    device_name : str
        Name of the PennyLane device.
    """

    def __init__(self, device_name="default.qubit"):
        self.device_name = device_name

    def create_device(self, n_qubits: int):
        """
        Create a PennyLane device with the requested
        number of qubits.
        """

        return qml.device(
            self.device_name,
            wires=n_qubits
        )