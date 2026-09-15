import numpy as np
from sklearn.svm import SVC


class QuantumClassifier:
    """
    Quantum-kernel based classifier.

    The classifier uses a quantum kernel to construct
    the kernel matrix and a classical SVM as the
    decision model.

    Parameters
    ----------
    kernel : QuantumKernel
        Quantum kernel used to compute similarities.

    C : float
        SVM regularization parameter.
    """

    def __init__(
        self,
        kernel,
        C=1.0
    ):

        self.kernel = kernel

        self.classifier = SVC(
            C=C,
            kernel="precomputed"
        )

        self.X_train_ = None
        self.is_fitted_ = False

    def fit(self, X, y):
        """
        Fit the quantum-kernel classifier.
        """

        X = np.asarray(X)

        K_train = self.kernel.compute(X)

        self.classifier.fit(
            K_train,
            y
        )

        self.X_train_ = X
        self.is_fitted_ = True

        return self

    def predict(self, X):
        """
        Predict labels for new samples.
        """

        self._check_fitted()

        X = np.asarray(X)

        K_test = self.kernel.compute(
            X,
            self.X_train_
        )

        return self.classifier.predict(
            K_test
        )

    def score(self, X, y):
        """
        Compute classification accuracy.
        """

        predictions = self.predict(X)

        return np.mean(
            predictions == np.asarray(y)
        )

    def _check_fitted(self):

        if not self.is_fitted_:

            raise RuntimeError(
                "QuantumClassifier has not been fitted. "
                "Call fit() before predict()."
            )