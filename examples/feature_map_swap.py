import numpy as np

from qmlib.feature_maps import (
    ZFeatureMap,
    ZZFeatureMap,
    PauliFeatureMap,
)

from qmlib.kernels import FidelityKernel
from qmlib.models import QuantumClassifier


X = np.array([
    [0.1, 0.2],
    [0.2, 0.1],
    [1.0, 1.1],
    [1.1, 1.0],
])

y = np.array([
    0,
    0,
    1,
    1,
])


# Change ONLY this component.
feature_map = ZZFeatureMap(
    n_qubits=2,
    reps=1
)

kernel = FidelityKernel(
    feature_map=feature_map
)

model = QuantumClassifier(
    kernel=kernel
)

model.fit(
    X,
    y
)

predictions = model.predict(X)

print("Feature map:")
print(feature_map.get_config())

print("\nPredictions:")
print(predictions)

print("\nAccuracy:")
print(model.score(X, y))