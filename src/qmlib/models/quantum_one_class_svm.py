"""One-Class SVM using a precomputed QMLib quantum kernel."""

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.svm import OneClassSVM


def _as_finite_scalar(value, name):
    """Convert a scalar hyperparameter to float and reject invalid values."""
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise ValueError(f"{name} must be a finite numeric scalar.")
    try:
        numeric_value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite numeric scalar.") from exc
    if not np.isfinite(numeric_value):
        raise ValueError(f"{name} must be finite.")
    return numeric_value


def _validate_parameters(nu, tol, cache_size):
    """Validate One-Class SVM hyperparameters."""
    numeric_nu = _as_finite_scalar(nu, "nu")
    numeric_tol = _as_finite_scalar(tol, "tol")
    numeric_cache_size = _as_finite_scalar(cache_size, "cache_size")

    if not 0 < numeric_nu <= 1:
        raise ValueError("nu must be in (0, 1].")
    if numeric_tol <= 0:
        raise ValueError("tol must be positive.")
    if numeric_cache_size <= 0:
        raise ValueError("cache_size must be positive.")


class QuantumOneClassSVM(BaseEstimator):
    """
    One-Class SVM with a quantum kernel.

    Training uses normal reference samples only.
    Prediction follows scikit-learn's One-Class SVM convention:
      +1 = normal
      -1 = anomaly

    `anomaly_score` returns the negated decision function, so larger values
    indicate more anomalous observations.
    """

    def __init__(self, kernel, nu=0.05, tol=1e-3, cache_size=200):
        _validate_parameters(nu, tol, cache_size)
        self.kernel = kernel
        self.nu = nu
        self.tol = tol
        self.cache_size = cache_size
        self.is_fitted_ = False

    @staticmethod
    def _validate_X(X, name="X"):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"{name} must be a 2D array.")
        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError(f"{name} must not be empty.")
        if not np.isfinite(X).all():
            raise ValueError(f"{name} contains non-finite values.")
        return X

    def fit(self, X):
        """Fit the One-Class SVM using normal reference observations."""
        # Revalidate in case set_params/direct assignment changed a parameter.
        _validate_parameters(self.nu, self.tol, self.cache_size)
        X = self._validate_X(X)
        if len(X) < 2:
            raise ValueError("At least two reference samples are required.")
        if not callable(getattr(self.kernel, "compute", None)):
            raise TypeError("kernel must implement compute(X, Y=None).")

        self.is_fitted_ = False
        gram = np.asarray(self.kernel.compute(X, X), dtype=float)
        if gram.shape != (len(X), len(X)):
            raise ValueError(
                f"Training kernel has shape {gram.shape}; "
                f"expected {(len(X), len(X))}."
            )
        if not np.isfinite(gram).all():
            raise ValueError("Training kernel contains non-finite values.")

        self.model_ = OneClassSVM(
            kernel="precomputed",
            nu=float(self.nu),
            tol=float(self.tol),
            cache_size=float(self.cache_size),
        )
        self.model_.fit(gram)

        self.X_train_ = X.copy()
        self.n_features_in_ = X.shape[1]
        self.is_fitted_ = True
        return self

    def _cross_kernel(self, X):
        if not self.is_fitted_:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        X = self._validate_X(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("Feature dimension mismatch.")

        cross = np.asarray(self.kernel.compute(X, self.X_train_), dtype=float)
        expected = (len(X), len(self.X_train_))
        if cross.shape != expected:
            raise ValueError(
                f"Prediction kernel has shape {cross.shape}; expected {expected}."
            )
        if not np.isfinite(cross).all():
            raise ValueError("Prediction kernel contains non-finite values.")
        return cross

    def decision_function(self, X):
        """Return signed normality scores (higher means more normal)."""
        cross = self._cross_kernel(X)
        return self.model_.decision_function(cross).ravel()

    def score_samples(self, X):
        """Alias for the signed One-Class SVM normality score."""
        cross = self._cross_kernel(X)
        return self.model_.score_samples(cross).ravel()

    def anomaly_score(self, X):
        """Return anomaly scores (higher means more anomalous)."""
        return -self.decision_function(X)

    def predict(self, X):
        """Return +1 for normal and -1 for anomaly."""
        cross = self._cross_kernel(X)
        return self.model_.predict(cross)

    def get_config(self):
        """Return estimator configuration."""
        return {
            "model": self.__class__.__name__,
            "kernel": type(self.kernel).__name__,
            "nu": float(self.nu),
            "tol": float(self.tol),
            "cache_size": float(self.cache_size),
            "n_reference_samples": (
                None if not self.is_fitted_ else int(len(self.X_train_))
            ),
            "is_fitted": bool(self.is_fitted_),
        }
