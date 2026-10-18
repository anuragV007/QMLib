
import numpy as np
from sklearn.preprocessing import StandardScaler

from qmlib.feature_maps import ZFeatureMap
from qmlib.kernels import FidelityKernel
from qmlib.models import QuantumClassifier
from qmlib.pipeline import QMLPipeline


X_train = np.array([
    [-1.0, -1.0],
    [-0.8, -0.9],
    [0.9, 1.0],
    [1.0, 0.8],
])

y_train = np.array([0, 0, 1, 1])

X_test = np.array([
    [-0.9, -0.8],
    [0.8, 0.9],
])

y_test = np.array([0, 1])

feature_map = ZFeatureMap(n_qubits=2)
kernel = FidelityKernel(feature_map=feature_map)
classifier = QuantumClassifier(kernel=kernel)

pipeline = QMLPipeline(
    steps=[("scaler", StandardScaler())],
    estimator=classifier,
)

pipeline.fit(X_train, y_train)

predictions = pipeline.predict(X_test)

print("Configuration:", pipeline.get_config())
print("Predictions:", predictions)
print("Accuracy:", pipeline.score(X_test, y_test))
