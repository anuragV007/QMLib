import numpy as np
import pytest
import pennylane as qml

from qmlib.feature_maps import (
    ZFeatureMap,
    ZZFeatureMap,
)


def test_z_feature_map_accepts_correct_dimension():
    feature_map = ZFeatureMap(n_qubits=4)

    x = np.array([0.1, 0.2, 0.3, 0.4])

    # Should not raise an exception
    feature_map.circuit(x)


def test_z_feature_map_rejects_wrong_dimension():
    feature_map = ZFeatureMap(n_qubits=4)

    x = np.array([0.1, 0.2, 0.3])

    with pytest.raises(ValueError):
        feature_map.circuit(x)


def test_zz_feature_map_accepts_correct_dimension():
    feature_map = ZZFeatureMap(n_qubits=4)

    x = np.array([0.1, 0.2, 0.3, 0.4])

    feature_map.circuit(x)


def test_zz_feature_map_rejects_wrong_dimension():
    feature_map = ZZFeatureMap(n_qubits=4)

    x = np.array([0.1, 0.2, 0.3])

    with pytest.raises(ValueError):
        feature_map.circuit(x)


def test_z_feature_map_builds_quantum_operations():
    dev = qml.device("default.qubit", wires=4)

    feature_map = ZFeatureMap(n_qubits=4)

    @qml.qnode(dev)
    def circuit():
        feature_map.circuit(
            np.array([0.1, 0.2, 0.3, 0.4])
        )
        return qml.state()

    state = circuit()

    assert state.shape == (16,)


def test_zz_feature_map_builds_quantum_operations():
    dev = qml.device("default.qubit", wires=4)

    feature_map = ZZFeatureMap(n_qubits=4)

    @qml.qnode(dev)
    def circuit():
        feature_map.circuit(
            np.array([0.1, 0.2, 0.3, 0.4])
        )
        return qml.state()

    state = circuit()

    assert state.shape == (16,)