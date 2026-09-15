from qmlib.backends import PennyLaneBackend


def test_pennylane_backend_creates_device():

    backend = PennyLaneBackend()

    device = backend.create_device(
        n_qubits=4
    )

    assert device is not None