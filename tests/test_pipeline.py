
import numpy as np
import pytest

from qmlib.pipeline import QMLPipeline


class DummyTransformer:
    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        return self

    def transform(self, X):
        return np.asarray(X) - self.mean_


class DummyEstimator:
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.fitted_ = True
        return self

    def predict(self, X):
        if not getattr(self, "fitted_", False):
            raise RuntimeError("Estimator is not fitted.")

        return np.zeros(len(X), dtype=int)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)


def make_pipeline():
    return QMLPipeline(
        steps=[("center", DummyTransformer())],
        estimator=DummyEstimator(),
    )


def test_fit_returns_pipeline():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = np.array([0, 1])

    pipeline = make_pipeline()

    assert pipeline.fit(X, y) is pipeline
    assert pipeline.is_fitted_


def test_predict_returns_one_prediction_per_sample():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = np.array([0, 1])

    pipeline = make_pipeline().fit(X, y)

    predictions = pipeline.predict(X)

    assert predictions.shape == (2,)
    assert np.array_equal(predictions, [0, 0])


def test_score():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = np.array([0, 0])

    # Use two classes for training, then test on one class.
    X_train = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
    ])
    y_train = np.array([0, 1, 0])

    pipeline = make_pipeline().fit(X_train, y_train)

    assert pipeline.score(X, y) == 1.0


def test_predict_before_fit_raises_error():
    pipeline = make_pipeline()

    with pytest.raises(RuntimeError, match="has not been fitted"):
        pipeline.predict(np.array([[1.0, 2.0]]))


def test_duplicate_step_names_are_rejected():
    with pytest.raises(ValueError, match="unique"):
        QMLPipeline(
            steps=[
                ("transform", DummyTransformer()),
                ("transform", DummyTransformer()),
            ],
            estimator=DummyEstimator(),
        )


def test_mismatched_sample_counts_are_rejected():
    pipeline = make_pipeline()

    with pytest.raises(ValueError, match="same number"):
        pipeline.fit(
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            np.array([0]),
        )


def test_single_class_training_data_is_rejected():
    pipeline = make_pipeline()

    with pytest.raises(ValueError, match="at least two classes"):
        pipeline.fit(
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            np.array([0, 0]),
        )


def test_get_config():
    config = make_pipeline().get_config()

    assert config["pipeline_class"] == "QMLPipeline"
    assert config["steps"][0]["name"] == "center"
    assert config["estimator"] == "DummyEstimator"
