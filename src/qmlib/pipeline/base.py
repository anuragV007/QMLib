"""Composable preprocessing and QML estimator pipeline."""

from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np
from sklearn.base import BaseEstimator


class QMLPipeline(BaseEstimator):
    """
    Compose preprocessing transformers with a final estimator.

    Each transformer must implement ``fit(X)`` and ``transform(X)``.
    The final estimator must implement ``fit(X, y)`` and ``predict(X)``.
    Its ``score`` method is delegated to, so both classifiers and regressors
    can be used without hard-coded task assumptions.

    Parameters
    ----------
    steps:
        Ordered (name, transformer) pairs.
    estimator:
        Final estimator, e.g. QuantumClassifier or QuantumKernelRidge.
    """

    def __init__(self, steps: List[Tuple[str, Any]], estimator: Any):
        self.steps = steps
        self.estimator = estimator
        self.is_fitted_ = False
        self._validate_configuration()

    @property
    def named_steps(self):
        """Return preprocessing transformers by their step names."""
        return dict(self.steps)

    def _validate_configuration(self):
        if not isinstance(self.steps, (list, tuple)):
            raise TypeError("steps must be a list or tuple.")

        names = []
        for step in self.steps:
            if not isinstance(step, tuple) or len(step) != 2:
                raise ValueError("Each step must be a (name, transformer) tuple.")

            name, transformer = step
            if not isinstance(name, str) or not name:
                raise ValueError("Every pipeline step needs a non-empty name.")
            if name == "estimator":
                raise ValueError("'estimator' is reserved and cannot be a step name.")
            if not callable(getattr(transformer, "fit", None)):
                raise TypeError(f"Transformer '{name}' must implement fit().")
            if not callable(getattr(transformer, "transform", None)):
                raise TypeError(f"Transformer '{name}' must implement transform().")
            names.append(name)

        if len(names) != len(set(names)):
            raise ValueError("Pipeline step names must be unique.")

        if not callable(getattr(self.estimator, "fit", None)):
            raise TypeError("estimator must implement fit().")
        if not callable(getattr(self.estimator, "predict", None)):
            raise TypeError("estimator must implement predict().")

    @staticmethod
    def _validate_X(X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")
        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("X must contain samples and features.")
        if not np.isfinite(X).all():
            raise ValueError("X contains NaN or infinite values.")
        return X

    def _fit_transform_steps(self, X):
        for name, transformer in self.steps:
            transformer.fit(X)
            transformed = self._validate_X(transformer.transform(X))
            if len(transformed) != len(X):
                raise ValueError(
                    f"Transformer '{name}' changed the number of samples."
                )
            X = transformed
        return X

    def _transform_steps(self, X):
        X = self._validate_X(X)
        for name, transformer in self.steps:
            transformed = self._validate_X(transformer.transform(X))
            if len(transformed) != len(X):
                raise ValueError(
                    f"Transformer '{name}' changed the number of samples."
                )
            X = transformed
        return X

    def fit(self, X, y):
        """Fit each transformer in order, then fit the final estimator."""
        X = self._validate_X(X)
        y = np.asarray(y)
        if y.ndim == 0:
            raise ValueError("y must contain one target per sample.")
        if len(X) != len(y):
            raise ValueError("X and y must contain the same number of samples.")

        self.is_fitted_ = False
        transformed = self._fit_transform_steps(X)
        self.estimator.fit(transformed, y)
        self.is_fitted_ = True
        return self

    def predict(self, X):
        """Transform X and delegate prediction to the final estimator."""
        self._check_is_fitted()
        return self.estimator.predict(self._transform_steps(X))

    def score(self, X, y):
        """Delegate scoring to the final estimator (accuracy, R², etc.)."""
        self._check_is_fitted()
        scorer = getattr(self.estimator, "score", None)
        if not callable(scorer):
            raise AttributeError("The final estimator does not implement score().")
        return scorer(self._transform_steps(X), y)

    def decision_function(self, X):
        """Delegate decision_function when supported by the estimator."""
        self._check_is_fitted()
        method = getattr(self.estimator, "decision_function", None)
        if not callable(method):
            raise AttributeError("The final estimator has no decision_function().")
        return method(self._transform_steps(X))

    def predict_proba(self, X):
        """Delegate predict_proba when supported by the estimator."""
        self._check_is_fitted()
        method = getattr(self.estimator, "predict_proba", None)
        if not callable(method):
            raise AttributeError("The final estimator has no predict_proba().")
        return method(self._transform_steps(X))

    def get_params(self, deep=True):
        """Expose pipeline and nested component parameters."""
        params = {"steps": self.steps, "estimator": self.estimator}
        if deep:
            for name, component in self.steps:
                params[name] = component
                getter = getattr(component, "get_params", None)
                if callable(getter):
                    params.update({
                        f"{name}__{key}": value
                        for key, value in getter(deep=True).items()
                    })

            getter = getattr(self.estimator, "get_params", None)
            if callable(getter):
                params.update({
                    f"estimator__{key}": value
                    for key, value in getter(deep=True).items()
                })
        return params

    def set_params(self, **params):
        """Set pipeline components or nested estimator/transformer parameters."""
        if not params:
            return self

        if "steps" in params:
            self.steps = params.pop("steps")
        if "estimator" in params:
            self.estimator = params.pop("estimator")

        step_map = dict(self.steps)
        for key, value in params.items():
            if "__" not in key:
                if key in step_map:
                    step_map[key] = value
                else:
                    raise ValueError(f"Unknown pipeline parameter: {key}")
                continue

            component_name, sub_name = key.split("__", 1)
            if component_name == "estimator":
                component = self.estimator
            elif component_name in step_map:
                component = step_map[component_name]
            else:
                raise ValueError(f"Unknown pipeline component: {component_name}")

            setter = getattr(component, "set_params", None)
            if callable(setter):
                setter(**{sub_name: value})
            elif hasattr(component, sub_name):
                setattr(component, sub_name, value)
            else:
                raise ValueError(
                    f"Component '{component_name}' has no parameter '{sub_name}'."
                )

        self.steps = [
            (name, step_map[name]) for name, _ in self.steps
        ]
        self._validate_configuration()
        self.is_fitted_ = False
        return self

    def get_config(self):
        """Return a JSON-serializable summary of the pipeline structure."""
        return {
            "pipeline_class": self.__class__.__name__,
            "steps": [
                {"name": name, "class": transformer.__class__.__name__}
                for name, transformer in self.steps
            ],
            "estimator": self.estimator.__class__.__name__,
        }

    def _check_is_fitted(self):
        if not self.is_fitted_:
            raise RuntimeError(
                "This pipeline has not been fitted. Call fit() first."
            )
