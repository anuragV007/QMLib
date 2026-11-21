"""Quantum-kernel SVM classifier."""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC


class QuantumClassifier(ClassifierMixin, BaseEstimator):
    """
    SVC classifier using a QMLib precomputed quantum kernel.

    The quantum kernel produces sample similarities; the final SVM
    optimization is classical.
    """

    def __init__(self, kernel, C=1.0):
        self.kernel = kernel
        self.C = C

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

    def fit(self, X, y):
        """Fit the quantum kernel SVM."""
        X = self._validate_X(X)
        y = np.asarray(y)

        if y.ndim != 1:
            raise ValueError("y must be a 1D array.")
        if len(X) != len(y):
            raise ValueError("X and y must contain the same number of samples.")
        if len(np.unique(y)) < 2:
            raise ValueError("Classification requires at least two classes.")
        if not np.isscalar(self.C) or self.C <= 0:
            raise ValueError("C must be greater than 0.")
        if not callable(getattr(self.kernel, "compute", None)):
            raise TypeError("kernel must implement compute(X, Y=None).")

        gram = np.asarray(self.kernel.compute(X, X), dtype=float)
        if gram.shape != (len(X), len(X)):
            raise ValueError("Training kernel matrix has an invalid shape.")
        if not np.isfinite(gram).all():
            raise ValueError("Training kernel matrix contains non-finite values.")

        self.classifier_ = SVC(C=self.C, kernel="precomputed")
        self.classifier_.fit(gram, y)
        self.X_train_ = X.copy()
        self.classes_ = self.classifier_.classes_
        self.n_features_in_ = X.shape[1]
        self.is_fitted_ = True
        return self

    def predict(self, X):
        """Predict class labels."""
        self._check_fitted()
        X = self._validate_X(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has a different feature count from training data.")

        cross = np.asarray(
            self.kernel.compute(X, self.X_train_), dtype=float
        )
        if cross.shape != (len(X), len(self.X_train_)):
            raise ValueError("Prediction kernel matrix has an invalid shape.")
        if not np.isfinite(cross).all():
            raise ValueError("Prediction kernel matrix contains non-finite values.")
        return self.classifier_.predict(cross)

    def decision_function(self, X):
        """Return SVM decision scores."""
        self._check_fitted()
        X = self._validate_X(X)
        cross = np.asarray(self.kernel.compute(X, self.X_train_), dtype=float)
        if cross.shape != (len(X), len(self.X_train_)):
            raise ValueError("Prediction kernel matrix has an invalid shape.")
        return self.classifier_.decision_function(cross)

    def score(self, X, y):
        """Return classification accuracy."""
        return accuracy_score(y, self.predict(X))

    def get_config(self):
        """Return a JSON-serializable estimator summary."""
        return {
            "model": self.__class__.__name__,
            "kernel": type(self.kernel).__name__,
            "C": float(self.C),
            "n_train_samples": (
                None if not getattr(self, "is_fitted_", False)
                else int(len(self.X_train_))
            ),
            "is_fitted": bool(getattr(self, "is_fitted_", False)),
        }

    def _check_fitted(self):
        if not getattr(self, "is_fitted_", False):
            raise RuntimeError(
                "QuantumClassifier has not been fitted. Call fit() first."
            )
