import numpy as np
import pytest

from qmlib.pipeline import QMLPipeline


class DummyTransformer:
    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float) - self.mean_


class DummyEstimator:
    def fit(self, X, y):
        self.target_mean_ = float(np.mean(y))
        self.fitted_ = True
        return self

    def predict(self, X):
        if not getattr(self, "fitted_", False):
            raise RuntimeError("Estimator is not fitted.")
        return np.full(len(X), self.target_mean_)

    def score(self, X, y):
        return -float(np.mean((self.predict(X) - np.asarray(y)) ** 2))


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


def test_score_delegates_to_estimator():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    y = np.array([1.0, 2.0, 3.0])
    pipeline = make_pipeline().fit(X, y)
    assert pipeline.score(X, y) <= 0.0


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


def test_single_class_target_is_not_rejected_by_generic_pipeline():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    y = np.array([1.0, 1.0, 1.0])
    pipeline = make_pipeline().fit(X, y)
    assert pipeline.predict(X).shape == (3,)


def test_get_config():
    config = make_pipeline().get_config()
    assert config["pipeline_class"] == "QMLPipeline"
    assert config["steps"][0]["name"] == "center"
    assert config["estimator"] == "DummyEstimator"
