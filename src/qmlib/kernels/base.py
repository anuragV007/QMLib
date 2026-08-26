from abc import ABC, abstractmethod


class QuantumKernel(ABC):
    """Base interface for quantum kernels."""

    @abstractmethod
    def compute(self, X, Y=None):
        """
        Compute the quantum kernel matrix.

        Parameters
        ----------
        X : array-like
            First dataset.
        Y : array-like, optional
            Second dataset.

        Returns
        -------
        numpy.ndarray
            Kernel matrix.
        """
        raise NotImplementedError