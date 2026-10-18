from abc import ABC, abstractmethod


class Backend(ABC):
    """
    Base interface for quantum execution backends.

    A backend is responsible for providing the execution
    device used by QML components.
    """

    @abstractmethod
    def create_device(self, n_qubits: int):
        """
        Create a quantum device.

        Parameters
        ----------
        n_qubits : int
            Number of qubits required.

        Returns
        -------
        Quantum device
        """
        raise NotImplementedError