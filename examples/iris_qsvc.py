import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
)

from qmlib.feature_maps import ZZFeatureMap
from qmlib.kernels import FidelityKernel
from qmlib.models import QuantumClassifier


# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

iris = load_iris()

X = iris.data
y = iris.target


# --------------------------------------------------
# 2. Use two classes for a simple binary problem
# --------------------------------------------------

mask = y < 2

X = X[mask]
y = y[mask]


# --------------------------------------------------
# 3. Scale features
# --------------------------------------------------

scaler = StandardScaler()

X = scaler.fit_transform(X)


# --------------------------------------------------
# 4. Train/test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# 5. Choose quantum feature map
# --------------------------------------------------

feature_map = ZZFeatureMap(
    n_qubits=4,
    reps=2
)


# --------------------------------------------------
# 6. Create quantum kernel
# --------------------------------------------------

kernel = FidelityKernel(
    feature_map=feature_map
)


# --------------------------------------------------
# 7. Create classifier
# --------------------------------------------------

model = QuantumClassifier(
    feature_map=feature_map,
    kernel=kernel,
    C=1.0
)


# --------------------------------------------------
# 8. Train
# --------------------------------------------------

print("Training quantum classifier...")

model.fit(
    X_train,
    y_train
)


# --------------------------------------------------
# 9. Evaluate
# --------------------------------------------------

predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

f1 = f1_score(
    y_test,
    predictions
)


print("\nResults")
print("-------")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")