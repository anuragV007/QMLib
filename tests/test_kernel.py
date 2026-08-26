import numpy as np
import pytest

from qmlib.feature_maps import ZFeatureMap
from qmlib.kernels import FidelityKernel


def test_fidelity_kernel_shape():
    feature_map = ZFeatureMap(n_qubits=2)

    kernel = FidelityKernel(feature_map)

    X = np.array([
        [0.1, 0.2],
        [0.3, 0.4],
        [0.5, 0.6],
    ])

    K = kernel.compute(X)

    assert K.shape == (3, 3)


def test_fidelity_kernel_diagonal():
    feature_map = ZFeatureMap(n_qubits=2)

    kernel = FidelityKernel(feature_map)

    X = np.array([
        [0.1, 0.2],
        [0.3, 0.4],
    ])

    K = kernel.compute(X)

    assert np.allclose(
        np.diag(K),
        1.0,
        atol=1e-6
    )


def test_fidelity_kernel_is_symmetric():
    feature_map = ZFeatureMap(n_qubits=2)

    kernel = FidelityKernel(feature_map)

    X = np.array([
        [0.1, 0.2],
        [0.3, 0.4],
    ])

    K = kernel.compute(X)

    assert np.allclose(
        K,
        K.T,
        atol=1e-6
    )


def test_fidelity_kernel_rejects_wrong_dimensions():
    feature_map = ZFeatureMap(n_qubits=2)

    kernel = FidelityKernel(feature_map)

    X = np.array([
        [0.1, 0.2, 0.3],
    ])

    with pytest.raises(ValueError):
        kernel.compute(X)