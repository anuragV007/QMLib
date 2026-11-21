import numpy as np
import pytest

from qmlib.models import QuantumOneClassSVM


class DummyKernel:
    """RBF similarity kernel for fast model testing."""

    def compute(self, X, Y=None):

        X = np.asarray(X, dtype=float)
        Y = X if Y is None else np.asarray(Y, dtype=float)

        distances = np.sum(
            (X[:, None, :] - Y[None, :, :]) ** 2,
            axis=2
        )

        return np.exp(-distances)


class WrongShapeKernel:
    def compute(self, X, Y=None):
        return np.zeros((1, 1))


class InvalidKernel:
    def compute(self, X, Y=None):
        return np.full((len(X), len(Y)), np.nan)


def make_training_data():

    return np.array([
        [-0.2, 0.0],
        [0.0, 0.1],
        [0.2, -0.1],
        [0.1, 0.2],
        [-0.1, -0.2],
        [0.05, -0.05],
    ])


def test_model_initialization():

    model = QuantumOneClassSVM(
        kernel=DummyKernel(),
        nu=0.05
    )

    assert model.nu == 0.05
    assert not model.is_fitted_


@pytest.mark.parametrize("nu", [0, -0.1, 1.1])
def test_invalid_nu(nu):

    with pytest.raises(ValueError):
        QuantumOneClassSVM(
            kernel=DummyKernel(),
            nu=nu
        )


@pytest.mark.parametrize("tol", [0, -1])
def test_invalid_tolerance(tol):

    with pytest.raises(ValueError):
        QuantumOneClassSVM(
            kernel=DummyKernel(),
            tol=tol
        )


def test_fit_returns_model():

    X = make_training_data()

    model = QuantumOneClassSVM(
        kernel=DummyKernel()
    )

    result = model.fit(X)

    assert result is model
    assert model.is_fitted_
    assert model.X_train_.shape == X.shape


def test_fit_requires_two_samples():

    model = QuantumOneClassSVM(
        kernel=DummyKernel()
    )

    with pytest.raises(ValueError):
        model.fit(np.array([[0.1, 0.2]]))


def test_predict_returns_sklearn_labels():

    X = make_training_data()

    model = QuantumOneClassSVM(
        kernel=DummyKernel(),
        nu=0.1
    )

    model.fit(X)

    predictions = model.predict(
        np.array([
            [0.0, 0.0],
            [5.0, 5.0],
        ])
    )

    assert predictions.shape == (2,)
    assert set(predictions).issubset({-1, 1})


def test_anomaly_score_is_higher_for_distant_sample():

    X = make_training_data()

    model = QuantumOneClassSVM(
        kernel=DummyKernel(),
        nu=0.1
    )

    model.fit(X)

    scores = model.anomaly_score(
        np.array([
            [0.0, 0.0],
            [5.0, 5.0],
        ])
    )

    assert scores[1] > scores[0]


def test_decision_function_before_fit():

    model = QuantumOneClassSVM(
        kernel=DummyKernel()
    )

    with pytest.raises(RuntimeError):
        model.decision_function(
            np.array([[0.0, 0.0]])
        )


def test_predict_before_fit():

    model = QuantumOneClassSVM(
        kernel=DummyKernel()
    )

    with pytest.raises(RuntimeError):
        model.predict(
            np.array([[0.0, 0.0]])
        )


def test_wrong_feature_dimension():

    model = QuantumOneClassSVM(
        kernel=DummyKernel()
    )

    model.fit(make_training_data())

    with pytest.raises(ValueError):
        model.predict(
            np.array([[0.1, 0.2, 0.3]])
        )


def test_non_finite_input():

    model = QuantumOneClassSVM(
        kernel=DummyKernel()
    )

    with pytest.raises(ValueError):
        model.fit(
            np.array([
                [0.1, np.nan],
                [0.2, 0.3],
            ])
        )


def test_wrong_kernel_shape():

    model = QuantumOneClassSVM(
        kernel=WrongShapeKernel()
    )

    with pytest.raises(ValueError):
        model.fit(make_training_data())


def test_non_finite_kernel():

    model = QuantumOneClassSVM(
        kernel=InvalidKernel()
    )

    with pytest.raises(ValueError):
        model.fit(make_training_data())


def test_get_config():

    model = QuantumOneClassSVM(
        kernel=DummyKernel(),
        nu=0.05
    )

    config = model.get_config()

    assert config["model"] == "QuantumOneClassSVM"
    assert config["kernel"] == "DummyKernel"
    assert config["nu"] == 0.05
    assert config["n_reference_samples"] is None

    model.fit(make_training_data())

    config = model.get_config()

    assert config["n_reference_samples"] == 6