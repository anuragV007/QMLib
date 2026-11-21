"""Similarity-based quantum novelty detector with calibration."""

import numpy as np
from sklearn.base import BaseEstimator


class QuantumAnomalyDetector(BaseEstimator):
    """
    Novelty detector using a quantum similarity kernel.

    `fit(reference_X, calibration_X)` is intentionally a two-set API:
    reference_X defines normal behaviour, while a separate normal calibration
    set determines the anomaly threshold. It is not a conventional
    supervised `fit(X, y)` estimator.

    Larger `score_samples` values indicate more anomalous observations.
    `predict` returns 1 for anomaly and 0 for normal.
    """

    def __init__(self, kernel, threshold_quantile=0.99):
        self.kernel = kernel
        self.threshold_quantile = threshold_quantile

    @staticmethod
    def _validate_X(X, name="X"):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"{name} must be a 2D array.")
        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError(f"{name} must not be empty.")
        if not np.isfinite(X).all():
            raise ValueError(f"{name} contains NaN or infinite values.")
        return X

    def _check_fitted(self):
        if not getattr(self, "is_fitted_", False):
            raise RuntimeError("Detector is not fitted. Call fit() first.")

    def score_samples(self, X):
        """Return 1 - maximum similarity to the normal reference set."""
        self._check_fitted()
        X = self._validate_X(X)

        if X.shape[1] != self.reference_X_.shape[1]:
            raise ValueError(
                "X and reference samples must have the same number of features."
            )

        similarities = np.asarray(
            self.kernel.compute(X, self.reference_X_), dtype=float
        )
        expected = (len(X), len(self.reference_X_))
        if similarities.shape != expected:
            raise ValueError(
                f"Kernel returned shape {similarities.shape}; expected {expected}."
            )
        if not np.isfinite(similarities).all():
            raise ValueError("Kernel returned non-finite similarities.")

        similarities = np.clip(similarities, 0.0, 1.0)
        return 1.0 - np.max(similarities, axis=1)

    def fit(self, reference_X, calibration_X):
        """Store normal references and calibrate the threshold on normal data."""
        if not 0.0 < self.threshold_quantile < 1.0:
            raise ValueError("threshold_quantile must be between 0 and 1.")

        reference_X = self._validate_X(reference_X, "reference_X")
        calibration_X = self._validate_X(calibration_X, "calibration_X")

        if reference_X.shape[1] != calibration_X.shape[1]:
            raise ValueError(
                "Reference and calibration data must have the same feature count."
            )
        if not callable(getattr(self.kernel, "compute", None)):
            raise TypeError("kernel must implement compute(X, Y=None).")

        self.reference_X_ = reference_X.copy()
        self.is_fitted_ = True
        self.calibration_scores_ = self.score_samples(calibration_X)
        self.threshold_ = float(
            np.quantile(self.calibration_scores_, self.threshold_quantile)
        )
        return self

    def predict(self, X):
        """Return 1 for anomaly and 0 for normal."""
        scores = self.score_samples(X)
        return (scores > self.threshold_).astype(int)

    def get_config(self):
        """Return detector configuration."""
        return {
            "model": self.__class__.__name__,
            "kernel": type(self.kernel).__name__,
            "threshold_quantile": float(self.threshold_quantile),
            "threshold": (
                None if not getattr(self, "is_fitted_", False)
                else float(self.threshold_)
            ),
            "n_reference_samples": (
                None if not getattr(self, "is_fitted_", False)
                else int(len(self.reference_X_))
            ),
            "is_fitted": bool(getattr(self, "is_fitted_", False)),
        }
