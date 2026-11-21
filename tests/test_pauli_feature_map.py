import numpy as np
import pytest
import pennylane as qml

from qmlib.feature_maps import PauliFeatureMap


def test_default_pauli_configuration():

    feature_map = PauliFeatureMap(n_qubits=2)

    assert feature_map.paulis == ["Z"]


def test_pauli_x_y_z_circuit():

    device = qml.device("default.qubit", wires=2)

    feature_map = PauliFeatureMap(
        n_qubits=2,
        reps=2,
        paulis=["X", "Y", "Z"]
    )

    @qml.qnode(device)
    def circuit():

        feature_map.circuit(
            np.array([0.1, 0.2])
        )

        return qml.state()

    state = circuit()

    assert state.shape == (4,)


def test_pauli_rejects_non_list():

    with pytest.raises(TypeError):
        PauliFeatureMap(
            n_qubits=2,
            paulis="XYZ"
        )


def test_pauli_rejects_unsupported_operator():

    with pytest.raises(ValueError):
        PauliFeatureMap(
            n_qubits=2,
            paulis=["A"]
        )


def test_pauli_rejects_empty_list():

    with pytest.raises(ValueError):
        PauliFeatureMap(
            n_qubits=2,
            paulis=[]
        )


def test_pauli_rejects_wrong_input_dimensions():

    feature_map = PauliFeatureMap(n_qubits=2)

    with pytest.raises(ValueError):
        feature_map.circuit(
            np.array([0.1, 0.2, 0.3])
        )