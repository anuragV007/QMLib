import numpy as np

from qmlib.feature_maps import ZFeatureMap
from qmlib.kernels import FidelityKernel
from qmlib.models import QuantumOneClassSVM


def test_quantum_one_class_svm_with_real_fidelity_kernel():

    X_train = np.array([
        [0.10, 0.20],
        [0.15, 0.25],
        [0.20, 0.15],
        [0.25, 0.20],
        [0.12, 0.18],
        [0.18, 0.22],
    ])

    X_test = np.array([
        [0.14, 0.19],
        [1.50, 1.70],
    ])

    feature_map = ZFeatureMap(
        n_qubits=2,
        reps=1
    )

    kernel = FidelityKernel(
        feature_map=feature_map
    )

    model = QuantumOneClassSVM(
        kernel=kernel,
        nu=0.05
    )

    model.fit(X_train)

    scores = model.anomaly_score(X_test)
    predictions = model.predict(X_test)

    assert model.is_fitted_

    assert scores.shape == (2,)
    assert predictions.shape == (2,)

    assert np.isfinite(scores).all()

    assert set(predictions).issubset({-1, 1})

    assert model.get_config()["model"] == "QuantumOneClassSVM"