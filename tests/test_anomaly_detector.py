
import numpy as np
import pytest

from qmlib.models.quantum_anomaly_detector import (
    QuantumAnomalyDetector,
)


class DummyKernel:
    """Simple similarity kernel for fast unit tests."""

    def compute(self, X, Y=None):
        X = np.asarray(X, dtype=float)
        Y = X if Y is None else np.asarray(Y, dtype=float)

        distances = np.sum(
            (X[:, None, :] - Y[None, :, :]) ** 2,
            axis=2,
        )

        return np.exp(-distances)


def test_detector_fits_and_predicts():
    reference = np.array([
        [0.0, 0.0],
        [0.1, 0.1],
        [-0.1, 0.0],
    ])

    calibration = np.array([
        [0.02, 0.01],
        [0.05, -0.02],
        [-0.03, 0.02],
        [0.01, -0.04],
    ])

    detector = QuantumAnomalyDetector(
        kernel=DummyKernel(),
        threshold_quantile=0.95,
    )

    detector.fit(reference, calibration)

    predictions = detector.predict(
        np.array([[0.0, 0.0], [5.0, 5.0]])
    )

    assert predictions.shape == (2,)
    assert predictions[0] == 0
    assert predictions[1] == 1
    assert detector.threshold_ is not None


def test_detector_rejects_non_finite_values():
    detector = QuantumAnomalyDetector(DummyKernel())

    with pytest.raises(ValueError):
        detector.fit(
            np.array([[0.0, np.nan]]),
            np.array([[0.0, 0.0]]),
        )


def test_detector_requires_fit_before_prediction():
    detector = QuantumAnomalyDetector(DummyKernel())

    with pytest.raises(RuntimeError):
        detector.predict(np.array([[0.0, 0.0]]))


def test_detector_rejects_wrong_feature_count():
    detector = QuantumAnomalyDetector(DummyKernel())

    detector.fit(
        np.array([[0.0, 0.0], [0.1, 0.1]]),
        np.array([[0.02, 0.02], [0.03, 0.03]]),
    )

    with pytest.raises(ValueError):
        detector.score_samples(np.array([[0.0, 0.0, 0.0]]))
