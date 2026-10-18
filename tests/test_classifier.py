import numpy as np

from qmlib.feature_maps import ZFeatureMap
from qmlib.kernels import FidelityKernel
from qmlib.models import QuantumClassifier


def test_quantum_classifier():

    X = np.array([
        [0.1, 0.1],
        [0.2, 0.2],
        [1.0, 1.0],
        [1.1, 1.1],
    ])

    y = np.array([
        0,
        0,
        1,
        1,
    ])

    feature_map = ZFeatureMap(
        n_qubits=2
    )

    kernel = FidelityKernel(
        feature_map
    )

    model = QuantumClassifier(
        kernel
    )

    model.fit(
        X,
        y
    )

    predictions = model.predict(X)

    assert len(predictions) == len(y)

    assert model.is_fitted_