import numpy as np

from qmlib.feature_maps import ZFeatureMap
from qmlib.kernels import FidelityKernel
from qmlib.backends import PennyLaneBackend

def test_fidelity_kernel_shape():

    feature_map = ZFeatureMap(
        n_qubits=2
    )

    kernel = FidelityKernel(
        feature_map
    )

    X = np.array([
        [0.1, 0.2],
        [0.3, 0.4],
        [0.5, 0.6],
    ])

    K = kernel.compute(X)

    assert K.shape == (3, 3)


def test_fidelity_kernel_diagonal():

    feature_map = ZFeatureMap(
        n_qubits=2
    )

    kernel = FidelityKernel(
        feature_map
    )

    X = np.array([
        [0.1, 0.2],
        [0.3, 0.4],
    ])

    K = kernel.compute(X)

    assert np.allclose(
        np.diag(K),
        1.0
    )


def test_fidelity_kernel_symmetric():

    feature_map = ZFeatureMap(
        n_qubits=2
    )

    kernel = FidelityKernel(
        feature_map
    )

    X = np.array([
        [0.1, 0.2],
        [0.3, 0.4],
        [0.5, 0.6],
    ])

    K = kernel.compute(X)

    assert np.allclose(
        K,
        K.T
    )


def test_cross_kernel_shape():

    feature_map = ZFeatureMap(
        n_qubits=2
    )

    kernel = FidelityKernel(
        feature_map
    )

    X_train = np.array([
        [0.1, 0.2],
        [0.3, 0.4],
        [0.5, 0.6],
    ])

    X_test = np.array([
        [0.7, 0.8],
        [0.9, 1.0],
    ])

    K = kernel.compute(
        X_test,
        X_train
    )

    assert K.shape == (2, 3)

def test_kernel_uses_supplied_backend():

    feature_map = ZFeatureMap(
        n_qubits=2
    )

    backend = PennyLaneBackend(
        device_name="default.qubit"
    )

    kernel = FidelityKernel(
        feature_map=feature_map,
        backend=backend
    )

    assert kernel.backend is backend
    assert kernel.device is not None