# QMLib

**A modular and extensible Quantum Machine Learning library.**

> **Status:** Early development — v0.2.0

QMLib is a Python library for constructing Quantum Machine Learning (QML) workflows using modular and interchangeable quantum components.

The project is inspired by the design philosophy of libraries such as scikit-learn, with an emphasis on:

* modularity
* reusable components
* interchangeable quantum methods
* consistent APIs
* backend abstraction
* reproducible experimentation

## Vision

QMLib aims to provide a common framework for constructing Quantum Machine Learning pipelines in which individual components can be replaced without rewriting the complete workflow.

The long-term goal is to support multiple QML paradigms, including:

* quantum kernel methods
* variational quantum algorithms
* quantum neural networks
* quantum convolutional neural networks

through common and composable interfaces.

---

# Architecture

QMLib is organized around the following conceptual architecture:

```text
                    QMLib
                      │
       ┌──────────────┼──────────────┐
       │              │              │
       ▼              ▼              ▼
 Feature Maps      Kernels         Models
       │              │              │
       └──────────────┼──────────────┘
                      │
                      ▼
                   Backend
                      │
                      ▼
               Quantum Runtime
```

The architecture separates:

* data encoding
* quantum similarity computation
* machine learning models
* quantum execution

This allows individual components to evolve independently.

---

# Current Components

## Feature Maps

Currently available:

* `ZFeatureMap`
* `ZZFeatureMap`
* `PauliFeatureMap`

Example:

```python
from qmlib.feature_maps import ZZFeatureMap

feature_map = ZZFeatureMap(
    n_qubits=4,
    reps=2
)
```

---

## Quantum Kernels

Currently available:

* `FidelityKernel`

The fidelity kernel is defined as:

$$
K(x,y)=|\langle\phi(x)|\phi(y)\rangle|^2
$$

Example:

```python
from qmlib.kernels import FidelityKernel

kernel = FidelityKernel(
    feature_map=feature_map
)
```

---

## Models

Currently available:

* `QuantumClassifier`

The quantum classifier uses the quantum kernel to construct a kernel matrix and a classical SVM as the decision model.

Example:

```python
from qmlib.models import QuantumClassifier

model = QuantumClassifier(
    kernel=kernel
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)
```

---

# Backend Abstraction

QMLib separates quantum algorithms from the execution backend.

Currently available:

* `PennyLaneBackend`

Example:

```python
from qmlib.backends import PennyLaneBackend

backend = PennyLaneBackend()
```

The backend can then be supplied to a quantum kernel:

```python
kernel = FidelityKernel(
    feature_map=feature_map,
    backend=backend
)
```

This architecture is designed to allow additional backends in future versions without changing the higher-level QML algorithm.

---

# Plug-and-Play Feature Maps

One of the main design goals of QMLib is component interchangeability.

For example:

```python
feature_map = ZFeatureMap(
    n_qubits=4
)
```

can be replaced by:

```python
feature_map = ZZFeatureMap(
    n_qubits=4
)
```

or:

```python
feature_map = PauliFeatureMap(
    n_qubits=4,
    paulis=["X", "Z"]
)
```

while keeping the same kernel and classifier:

```python
kernel = FidelityKernel(
    feature_map=feature_map
)

model = QuantumClassifier(
    kernel=kernel
)

model.fit(X_train, y_train)
```

This enables controlled comparison of different quantum feature representations.

---

# Example

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
    kernel=kernel
)

model.fit(
    X_train,
    y_train
)

predictions = model.predict(
    X_test
)
```

---

# Design Principles

QMLib follows several architectural principles:

### 1. Modularity

Each major QML component is represented independently.

### 2. Interchangeability

Different implementations can be substituted through common interfaces.

### 3. Separation of concerns

Feature maps, kernels, models and execution backends have separate responsibilities.

### 4. Reproducibility

Components expose configuration information to support experiment tracking and reproducibility.

### 5. Extensibility

New QML algorithms should be implementable without modifying unrelated components.

---

# Current Project Status

QMLib is currently under active development.

### Implemented

* Feature-map abstraction
* Z feature map
* ZZ feature map
* Pauli feature map
* Quantum-kernel abstraction
* Fidelity quantum kernel
* Quantum-kernel classifier
* Backend abstraction
* PennyLane backend
* Initial automated testing

### Planned

* Additional quantum kernels
* Additional execution backends
* Variational quantum classifiers
* Quantum neural networks
* Quantum convolutional neural networks
* Pipeline abstraction
* Benchmarking framework
* Hardware execution
* Performance and scalability analysis

---

# Research Direction

The library is being developed as part of an M.Tech thesis investigating the design of modular and extensible software abstractions for Quantum Machine Learning.

The research will evaluate:

* component interchangeability
* model performance
* execution time
* circuit complexity
* scalability
* backend portability
* reproducibility

using representative QML datasets and workloads.

---

# Development

Clone the repository and install the package in editable mode:

```bash
git clone https://github.com/anuragV007/QMLib.git
cd QMLib

python -m venv .venv

# Windows
.venv\Scripts\activate

pip install -e .
```

Run the test suite:

```bash
pytest
```

---

# License

This project is currently under development.
