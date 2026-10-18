
from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np


class QMLPipeline:
    """
    Compose preprocessing transformers with a final QML estimator.

    Intermediate transformers must implement fit() and transform().
    The final estimator must implement fit() and predict().

    Parameters
    ----------
    steps : list of (str, transformer) tuples
        Ordered preprocessing steps.
    estimator : object
        Final estimator, such as QuantumClassifier.
    """

    def __init__(
        self,
        steps: List[Tuple[str, Any]],
        estimator: Any,
    ):
        self.steps = list(steps)
        self.estimator = estimator
        self.is_fitted_ = False

        self._validate_configuration()

    def _validate_configuration(self):
        names = []

        for step in self.steps:
            if not isinstance(step, tuple) or len(step) != 2:
                raise ValueError(
                    "Each step must be a (name, transformer) tuple."
                )

            name, transformer = step

            if not isinstance(name, str) or not name:
                raise ValueError(
                    "Every pipeline step needs a non-empty name."
                )

            names.append(name)

            if not callable(getattr(transformer, "fit", None)):
                raise TypeError(
                    f"Transformer '{name}' must implement fit()."
                )

            if not callable(getattr(transformer, "transform", None)):
                raise TypeError(
                    f"Transformer '{name}' must implement transform()."
                )

        if len(names) != len(set(names)):
            raise ValueError("Pipeline step names must be unique.")

        if not callable(getattr(self.estimator, "fit", None)):
            raise TypeError("Estimator must implement fit().")

        if not callable(getattr(self.estimator, "predict", None)):
            raise TypeError("Estimator must implement predict().")

    @staticmethod
    def _validate_X(X):
        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")

        if X.shape[0] == 0:
            raise ValueError("X must contain at least one sample.")

        if X.shape[1] == 0:
            raise ValueError("X must contain at least one feature.")

        if not np.isfinite(X).all():
            raise ValueError(
                "X contains NaN or infinite values. "
                "Handle them during preprocessing."
            )

        return X

    def _fit_transform_steps(self, X):
        for name, transformer in self.steps:
            transformer.fit(X)
            X = np.asarray(transformer.transform(X), dtype=float)
            X = self._validate_X(X)

        return X

    def _transform_steps(self, X):
        X = self._validate_X(X)

        for name, transformer in self.steps:
            X = np.asarray(transformer.transform(X), dtype=float)
            X = self._validate_X(X)

        return X

    def fit(self, X, y):
        """Fit preprocessing steps and the final estimator."""
        X = self._validate_X(X)
        y = np.asarray(y)

        if y.ndim != 1:
            raise ValueError("y must be a 1D array.")

        if len(X) != len(y):
            raise ValueError(
                "X and y must contain the same number of samples."
            )

        if len(np.unique(y)) < 2:
            raise ValueError(
                "Classification requires at least two classes."
            )

        X_transformed = self._fit_transform_steps(X)

        self.estimator.fit(X_transformed, y)
        self.is_fitted_ = True

        return self

    def predict(self, X):
        """Transform samples and return predictions."""
        self._check_is_fitted()

        X_transformed = self._transform_steps(X)
        return self.estimator.predict(X_transformed)

    def score(self, X, y):
        """Return the estimator's score on transformed samples."""
        self._check_is_fitted()

        if not callable(getattr(self.estimator, "score", None)):
            raise AttributeError(
                "The estimator does not implement score()."
            )

        X_transformed = self._transform_steps(X)
        return self.estimator.score(X_transformed, y)

    def get_config(self):
        """Return a basic description of the pipeline configuration."""
        return {
            "pipeline_class": self.__class__.__name__,
            "steps": [
                {
                    "name": name,
                    "class": transformer.__class__.__name__,
                }
                for name, transformer in self.steps
            ],
            "estimator": self.estimator.__class__.__name__,
        }

    def _check_is_fitted(self):
        if not self.is_fitted_:
            raise RuntimeError(
                "This pipeline has not been fitted. Call fit() first."
            )
