import numpy as np

from qmlib.feature_maps import (
    ZFeatureMap,
    ZZFeatureMap,
    PauliFeatureMap
)

from qmlib.kernels import FidelityKernel
from qmlib.models import QuantumClassifier


# --------------------------------------------------
# Example dataset
# --------------------------------------------------

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


# --------------------------------------------------
# Choose feature map
# --------------------------------------------------

feature_map = ZZFeatureMap(
    n_qubits=2,
    reps=1
)


# --------------------------------------------------
# Build quantum kernel
# --------------------------------------------------

kernel = FidelityKernel(
    feature_map=feature_map
)


# --------------------------------------------------
# Build classifier
# --------------------------------------------------

model = QuantumClassifier(
    kernel=kernel
)


# --------------------------------------------------
# Train
# --------------------------------------------------

model.fit(
    X,
    y
)


# --------------------------------------------------
# Predict
# --------------------------------------------------

predictions = model.predict(X)


print("Feature Map:")
print(
    feature_map.get_config()
)

print()

print("Predictions:")
print(predictions)

print()

print("Accuracy:")
print(
    model.score(X, y)
)