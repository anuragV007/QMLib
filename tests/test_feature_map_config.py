from qmlib.feature_maps import (
    ZFeatureMap,
    ZZFeatureMap,
    PauliFeatureMap,
)


def test_z_feature_map_config():

    feature_map = ZFeatureMap(
        n_qubits=4,
        reps=2
    )

    config = feature_map.get_config()

    assert config["name"] == "ZFeatureMap"
    assert config["n_qubits"] == 4
    assert config["reps"] == 2


def test_zz_feature_map_config():

    feature_map = ZZFeatureMap(
        n_qubits=4,
        reps=2
    )

    config = feature_map.get_config()

    assert config["name"] == "ZZFeatureMap"


def test_pauli_feature_map_config():

    feature_map = PauliFeatureMap(
        n_qubits=4,
        reps=2,
        paulis=["X", "Z"]
    )

    config = feature_map.get_config()

    assert config["name"] == "PauliFeatureMap"
    assert config["paulis"] == ["X", "Z"]