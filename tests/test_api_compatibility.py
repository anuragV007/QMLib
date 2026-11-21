import numpy as np
import pytest
from sklearn.base import clone

from qmlib.models import QuantumClassifier, QuantumKernelRidge
from qmlib.pipeline import QMLPipeline


class LinearKernel:
    def compute(self, X, Y=None):
        X = np.asarray(X, dtype=float)
        Y = X if Y is None else np.asarray(Y, dtype=float)
        return X @ Y.T


class Centerer:
    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float) - self.mean_


def test_quantum_kernel_ridge_supports_regression_api():
    X = np.linspace(-1, 1, 12).reshape(-1, 1)
    y = 2.0 * X.ravel() + 0.5

    model = QuantumKernelRidge(kernel=LinearKernel(), alpha=1e-8)
    model.fit(X, y)

    prediction = model.predict(X)
    assert prediction.shape == y.shape
    assert model.score(X, y) > 0.999
    assert clone(model).get_params()["alpha"] == pytest.approx(1e-8)


def test_pipeline_supports_continuous_regression_targets():
    X = np.linspace(-1, 1, 12).reshape(-1, 1)
    y = 3.0 * X.ravel() - 2.0

    regressor = QuantumKernelRidge(
        kernel=LinearKernel(),
        alpha=1e-8,
    )
    pipeline = QMLPipeline(
        steps=[("center", Centerer())],
        estimator=regressor,
    )

    pipeline.fit(X, y)
    assert pipeline.predict(X).shape == y.shape
    assert pipeline.score(X, y) > 0.999


def test_pipeline_nested_parameter_access():
    regressor = QuantumKernelRidge(
        kernel=LinearKernel(),
        alpha=0.1,
    )
    pipeline = QMLPipeline(steps=[], estimator=regressor)

    assert pipeline.get_params()["estimator__alpha"] == pytest.approx(0.1)
    pipeline.set_params(estimator__alpha=0.2)
    assert pipeline.estimator.alpha == pytest.approx(0.2)


def test_pipeline_does_not_require_two_target_classes():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([1.0, 1.0, 1.0, 1.0])

    pipeline = QMLPipeline(
        steps=[],
        estimator=QuantumKernelRidge(kernel=LinearKernel(), alpha=0.1),
    )
    pipeline.fit(X, y)
    assert pipeline.predict(X).shape == y.shape


def test_quantum_classifier_uses_same_kernel_interface():
    X = np.array([
        [-1.0, -1.0],
        [-0.8, -0.9],
        [0.8, 0.9],
        [1.0, 1.0],
    ])
    y = np.array([0, 0, 1, 1])

    model = QuantumClassifier(kernel=LinearKernel(), C=1.0)
    model.fit(X, y)
    assert model.predict(X).shape == y.shape
    assert model.score(X, y) > 0.9


def test_one_class_svm_parameters_are_cloneable():
    from qmlib.models import QuantumOneClassSVM

    model = QuantumOneClassSVM(
        kernel=LinearKernel(), nu=0.1, tol=1e-4, cache_size=100
    )
    cloned = clone(model)
    assert cloned.get_params()["nu"] == pytest.approx(0.1)
    assert cloned.get_params()["tol"] == pytest.approx(1e-4)
    assert cloned.get_params()["cache_size"] == 100


def test_anomaly_detector_parameters_are_cloneable():
    from qmlib.models import QuantumAnomalyDetector

    model = QuantumAnomalyDetector(
        kernel=LinearKernel(), threshold_quantile=0.95
    )
    cloned = clone(model)
    assert cloned.get_params()["threshold_quantile"] == pytest.approx(0.95)
