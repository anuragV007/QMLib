from abc import ABC, abstractmethod


class FeatureMap(ABC):
    """
    Base interface for quantum feature maps.
    """

    def __init__(
        self,
        n_qubits: int,
        reps: int = 1
    ):

        if not isinstance(n_qubits, int):
            raise TypeError(
                "n_qubits must be an integer."
            )

        if n_qubits <= 0:
            raise ValueError(
                "n_qubits must be greater than 0."
            )

        if not isinstance(reps, int):
            raise TypeError(
                "reps must be an integer."
            )

        if reps <= 0:
            raise ValueError(
                "reps must be greater than 0."
            )

        self.n_qubits = n_qubits
        self.reps = reps

    @abstractmethod
    def circuit(self, x):
        """
        Encode classical data x into a quantum state.
        """
        raise NotImplementedError

    def get_config(self):
        """
        Return the configuration of the feature map.
        """

        return {
            "n_qubits": self.n_qubits,
            "reps": self.reps,
        }