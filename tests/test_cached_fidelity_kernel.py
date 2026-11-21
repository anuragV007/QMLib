import numpy as np

from qmlib.feature_maps import ZFeatureMap
from qmlib.kernels import (
    FidelityKernel,
    CachedFidelityKernel,
)


def test_cached_kernel_matches_regular_kernel():

    X = np.array([
        [0.1, 0.2],
        [0.3, 0.4],
        [0.5, 0.6],
    ])

    feature_map = ZFeatureMap(n_qubits=2)

    regular = FidelityKernel(feature_map)
    cached = CachedFidelityKernel(feature_map)

    K_regular = regular.compute(X)
    K_cached = cached.compute(X)

    assert np.allclose(K_regular, K_cached)


def test_cached_kernel_reuses_statevectors():

    X = np.array([
        [0.1, 0.2],
        [0.3, 0.4],
        [0.5, 0.6],
    ])

    feature_map = ZFeatureMap(n_qubits=2)
    kernel = CachedFidelityKernel(feature_map)

    kernel.compute(X, X)

    initial_cache_size = len(kernel._state_cache)

    assert initial_cache_size == 3

    # Reuse previously computed samples.
    kernel.compute(X[:1], X)

    assert len(kernel._state_cache) == initial_cache_size


def test_cached_kernel_clear_cache():

    X = np.array([
        [0.1, 0.2],
        [0.3, 0.4],
    ])

    kernel = CachedFidelityKernel(
        ZFeatureMap(n_qubits=2)
    )

    kernel.compute(X)

    assert len(kernel._state_cache) > 0

    kernel.clear_cache()

    assert len(kernel._state_cache) == 0