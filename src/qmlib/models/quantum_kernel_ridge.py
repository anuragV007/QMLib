"""
Quantum-kernel ridge regression.

The quantum kernel builds the Gram matrix. Scikit-learn solves the regularized
kernel regression system. Centering the kernel and target provides an intercept.
"""

import numpy as np
from sklearn.base import RegressorMixin, BaseEstimator
from sklearn.kernel_ridge import KernelRidge


def _validate_positive_finite_parameter(value, name):
    """Validate a positive, finite scalar numeric hyperparameter."""
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise ValueError(f"{name} must be a finite number greater than 0.")
    try:
        numeric_value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"{name} must be a finite number greater than 0."
        ) from exc
    if not np.isfinite(numeric_value) or numeric_value <= 0:
        raise ValueError(f"{name} must be a finite number greater than 0.")
    return numeric_value


class QuantumKernelRidge(RegressorMixin, BaseEstimator):
    """Kernel ridge regressor backed by a QMLib quantum kernel."""

    def __init__(self, kernel, alpha=1e-2):
        # Validate at construction time (as required by the QMLib API/tests),
        # while retaining the original parameter value for sklearn cloning.
        _validate_positive_finite_parameter(alpha, "alpha")
        self.kernel = kernel
        self.alpha = alpha
        self.is_fitted_ = False

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

    def _compute_kernel(self, X, Y, expected_shape, name):
        matrix = np.asarray(self.kernel.compute(X, Y), dtype=float)
        if matrix.shape != expected_shape:
            raise ValueError(
                f"{name} kernel has shape {matrix.shape}; "
                f"expected {expected_shape}."
            )
        if not np.isfinite(matrix).all():
            raise ValueError(f"{name} kernel contains non-finite values.")
        return matrix

    def fit(self, X, y):
        """Fit using the centered quantum training Gram matrix."""
        # Revalidate because public estimator parameters may have been changed
        # after construction using set_params or direct assignment.
        alpha = _validate_positive_finite_parameter(self.alpha, "alpha")
        X = self._validate_X(X)
        y = np.asarray(y, dtype=float)

        if y.ndim == 2 and y.shape[1] == 1:
            y = y.ravel()
        if y.ndim not in (1, 2):
            raise ValueError("y must be a 1D or 2D numeric array.")
        if len(X) != len(y):
            raise ValueError("X and y must contain the same number of samples.")
        if len(X) < 2:
            raise ValueError("At least two training samples are required.")
        if not np.isfinite(y).all():
            raise ValueError("y contains NaN or infinite values.")
        if not callable(getattr(self.kernel, "compute", None)):
            raise TypeError("kernel must implement compute(X, Y=None).")

        self.is_fitted_ = False
        gram = self._compute_kernel(X, X, (len(X), len(X)), "Training")
        gram = 0.5 * (gram + gram.T)

        self.kernel_column_mean_ = gram.mean(axis=0, keepdims=True)
        self.kernel_grand_mean_ = float(gram.mean())
        centered_gram = (
            gram
            - gram.mean(axis=1, keepdims=True)
            - self.kernel_column_mean_
            + self.kernel_grand_mean_
        )

        self.y_mean_ = np.mean(y, axis=0, keepdims=True)
        self.model_ = KernelRidge(alpha=alpha, kernel="precomputed")
        self.model_.fit(centered_gram, y - self.y_mean_)

        self.X_train_ = X.copy()
        self.n_features_in_ = X.shape[1]
        self.is_fitted_ = True
        return self

    def predict(self, X):
        """Predict target values for new samples."""
        if not self.is_fitted_:
            raise RuntimeError("QuantumKernelRidge is not fitted. Call fit() first.")

        X = self._validate_X(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has a different feature count from training data.")

        cross = self._compute_kernel(
            X, self.X_train_, (len(X), len(self.X_train_)), "Prediction"
        )
        centered_cross = (
            cross
            - cross.mean(axis=1, keepdims=True)
            - self.kernel_column_mean_
            + self.kernel_grand_mean_
        )
        return self.model_.predict(centered_cross) + self.y_mean_

    def get_config(self):
        """Return the estimator configuration."""
        return {
            "model": self.__class__.__name__,
            "kernel": type(self.kernel).__name__,
            "alpha": float(self.alpha),
            "n_train_samples": (
                None if not self.is_fitted_ else int(len(self.X_train_))
            ),
            "is_fitted": bool(self.is_fitted_),
        }
