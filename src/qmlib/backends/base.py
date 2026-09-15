from abc import ABC, abstractmethod


class Backend(ABC):
    """
    Base interface for quantum execution backends.
    """

    @abstractmethod
    def create_device(self, n_qubits: int):
        """
        Create and return a quantum device.

        Parameters
        ----------
        n_qubits : int
            Number of qubits required.

        Returns
        -------
        Quantum device
        """
        raise NotImplementedError