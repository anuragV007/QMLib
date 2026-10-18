import numpy as np

from sklearn.svm import SVC


class QuantumClassifier:
    """
    Quantum-kernel based classifier.

    The quantum kernel is used to construct a kernel matrix,
    while a classical SVM performs the final classification.

    Parameters
    ----------
    kernel : QuantumKernel
        Quantum kernel used by the classifier.

    C : float
        SVM regularization parameter.
    """

    def __init__(
        self,
        kernel,
        C: float = 1.0
    ):

        if C <= 0:
            raise ValueError(
                "C must be greater than 0."
            )

        self.kernel = kernel

        self.classifier = SVC(
            C=C,
            kernel="precomputed"
        )

        self.X_train_ = None
        self.is_fitted_ = False

    def fit(self, X, y):
        """
        Fit the classifier.
        """

        X = np.asarray(X)

        if X.ndim != 2:
            raise ValueError(
                "X must be a 2D array."
            )

        if len(X) != len(y):
            raise ValueError(
                "X and y must contain the same "
                "number of samples."
            )

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
        Return classification accuracy.
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