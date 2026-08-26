# QMLib

A modular and extensible Quantum Machine Learning library.

> **Status:** Early development — v0.1.0

## Vision

QMLib aims to provide a modular interface for constructing
Quantum Machine Learning pipelines using interchangeable
quantum components.

The long-term goal is to support multiple QML paradigms
through a common and composable API.

## Current Features

### Feature Maps

- ZFeatureMap
- ZZFeatureMap

### Quantum Kernels

- Fidelity Quantum Kernel

### Models

- Quantum Kernel Classifier

## Example

```python
from qmlib.feature_maps import ZZFeatureMap
from qmlib.kernels import FidelityKernel
from qmlib.models import QuantumClassifier

feature_map = ZZFeatureMap(
    n_qubits=4,
    reps=2
)

kernel = FidelityKernel(
    feature_map=feature_map
)

model = QuantumClassifier(
    feature_map=feature_map,
    kernel=kernel
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)