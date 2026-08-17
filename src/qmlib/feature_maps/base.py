from abc import ABC, abstractmethod


class FeatureMap(ABC):
    """Base interface for quantum feature maps."""

    def __init__(self, n_qubits: int, reps: int = 1):
        self.n_qubits = n_qubits
        self.reps = reps

    @abstractmethod
    def circuit(self, x):
        """Encode classical data x into a quantum state."""
        raise NotImplementedError