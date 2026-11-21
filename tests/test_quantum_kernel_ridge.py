import numpy as np
import pytest

from qmlib.models import QuantumKernelRidge


class LinearKernel:
    def compute(self, X, Y=None):
        X = np.asarray(X, dtype=float)
        Y = X if Y is None else np.asarray(Y, dtype=float)
        return X @ Y.T


def test_quantum_kernel_ridge_fit_predict():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = 2.0 * X.ravel() + 1.0

    model = QuantumKernelRidge(LinearKernel(), alpha=1e-8)
    model.fit(X, y)
    predictions = model.predict(X)

    assert predictions.shape == y.shape
    assert np.mean(np.abs(predictions - y)) < 1e-5
    assert model.get_config()["is_fitted"] is True


def test_quantum_kernel_ridge_predict_before_fit_raises():
    model = QuantumKernelRidge(LinearKernel())
    with pytest.raises(RuntimeError):
        model.predict(np.array([[1.0]]))


def test_quantum_kernel_ridge_rejects_mismatched_rows():
    model = QuantumKernelRidge(LinearKernel())
    with pytest.raises(ValueError):
        model.fit(np.array([[0.0], [1.0]]), np.array([0.0]))


def test_quantum_kernel_ridge_rejects_nonpositive_alpha():
    with pytest.raises(ValueError):
        QuantumKernelRidge(LinearKernel(), alpha=0.0)
