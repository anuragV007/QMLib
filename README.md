# QMLib

**A modular Python library for composing Quantum Machine Learning workflows.**

> **Release target:** v0.4.0

QMLib separates quantum data encoding, quantum-kernel computation, classical learning algorithms and pipeline composition. It is designed to make quantum components reusable and interchangeable without rewriting the complete machine-learning workflow.

The current release focuses on **quantum-kernel methods** using PennyLane.

## What is implemented?

| Layer | Components |
|---|---|
| Feature maps | `ZFeatureMap`, `ZZFeatureMap`, `PauliFeatureMap` |
| Quantum kernels | `FidelityKernel`, `CachedFidelityKernel` |
| Models | `QuantumClassifier`, `QuantumOneClassSVM`, `QuantumAnomalyDetector`, `QuantumKernelRidge` |
| Backend | `PennyLaneBackend` |
| Composition | `QMLPipeline` |

QMLib is a research prototype. The current backend implementation uses PennyLane devices; it is not yet a provider-independent abstraction across quantum SDKs.

## Architecture

```text
Classical feature vectors
          |
          v
    Feature maps
  Z / ZZ / Pauli
          |
          v
    Quantum kernels
 Fidelity / Cached Fidelity
          |
          v
     QML estimators
 Classifier / One-Class SVM /
 Anomaly Detector / Kernel Ridge
          |
          v
      QMLPipeline
          |
          v
  PennyLane execution device
```

The kernel-based models are hybrid: quantum circuits generate the kernel similarities, while classical scikit-learn estimators perform the final SVM or kernel-ridge optimization.

## Installation

Python 3.9 or later is required.

```powershell
git clone https://github.com/anuragV007/QMLib.git
cd QMLib

python -m venv .venv
.venv\Scripts\activate

python -m pip install -e ".[dev]"
```

## Quick start: quantum classification

```python
from qmlib.feature_maps import ZZFeatureMap
from qmlib.kernels import FidelityKernel
from qmlib.models import QuantumClassifier

feature_map = ZZFeatureMap(n_qubits=2, reps=2)
kernel = FidelityKernel(feature_map=feature_map)
model = QuantumClassifier(kernel=kernel, C=1.0)

model.fit(X_train, y_train)
predictions = model.predict(X_test)
print("Accuracy:", model.score(X_test, y_test))
```

## Swap a feature map without changing the model

Only the feature-map construction changes:

```python
from qmlib.feature_maps import ZFeatureMap
# Alternatives: ZZFeatureMap, PauliFeatureMap

feature_map = ZFeatureMap(n_qubits=2, reps=2)
kernel = FidelityKernel(feature_map=feature_map)
model = QuantumClassifier(kernel=kernel)
```

The same kernel and model interfaces can be retained.

A runnable comparison is available in `examples/compare_feature_maps.py`.

## Quick start: quantum-kernel regression

```python
from qmlib.feature_maps import ZZFeatureMap
from qmlib.kernels import FidelityKernel
from qmlib.models import QuantumKernelRidge

feature_map = ZZFeatureMap(n_qubits=2, reps=1)
kernel = FidelityKernel(feature_map=feature_map)
regressor = QuantumKernelRidge(kernel=kernel, alpha=0.01)

regressor.fit(X_train, y_train)
predictions = regressor.predict(X_test)
print("R²:", regressor.score(X_test, y_test))
```

A complete example is available in `examples/quantum_kernel_ridge_regression.py`.

## Pipeline composition

`QMLPipeline` composes classical preprocessing transformers with a final estimator. It supports both classification and regression by delegating `score()` to the final estimator.

```python
from sklearn.preprocessing import StandardScaler
from qmlib.feature_maps import ZZFeatureMap
from qmlib.kernels import FidelityKernel
from qmlib.models import QuantumClassifier
from qmlib.pipeline import QMLPipeline

pipeline = QMLPipeline(
    steps=[("scaler", StandardScaler())],
    estimator=QuantumClassifier(
        kernel=FidelityKernel(ZZFeatureMap(n_qubits=2))
    ),
)

pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
```

Intermediate transformers must implement `fit(X)` and `transform(X)`. The final estimator must implement `fit(X, y)` and `predict(X)`. For scoring, it should implement `score(X, y)`.

## Feature maps

- **ZFeatureMap:** Hadamard preparation followed by data-dependent Z rotations.
- **ZZFeatureMap:** Z rotations with nearest-neighbour entangling ZZ-style interactions.
- **PauliFeatureMap:** Configurable single-qubit X, Y and Z data-dependent rotations.

The current `PauliFeatureMap` is a basic configurable Pauli-rotation implementation; it should not be confused with every generalized Pauli evolution circuit used in other frameworks.

## Quantum fidelity kernel

For encoded quantum states \(\lvert\phi(x)\rangle\), QMLib computes:

\[
K(x,y)=|\langle \phi(x)\mid\phi(y)\rangle|^2
\]

`CachedFidelityKernel` caches statevectors for repeated input vectors. Call `clear_cache()` to release cached states when an experiment is complete.

## Tests and examples

Run the complete test suite:

```powershell
python -m pytest -q
```

Run the feature-map interchangeability example:

```powershell
python .\examples\compare_feature_maps.py
```

Run the regression example:

```powershell
python .\examples\quantum_kernel_ridge_regression.py
```

## Current limitations

- The main execution path is PennyLane-based; cross-SDK backend portability is a future extension.
- Current kernel computation is statevector-based and can become expensive as qubit count or training-set size increases.
- Quantum-kernel models use classical SVM or kernel-ridge solvers after computing the quantum Gram matrix.
- No quantum advantage is claimed. Classical baselines and resource measurements are required for every empirical comparison.
- NASA SMAP/MSL experiments are research examples, not part of the core library API.

## Research project

QMLib is developed as part of an M.Tech thesis on modular and extensible Quantum Machine Learning software. The project evaluates component interchangeability, correctness, reproducibility, model performance and execution cost.

## Repository structure

```text
src/qmlib/
  backends/
  feature_maps/
  kernels/
  models/
  pipeline/

examples/
tests/
benchmarks/
experiments/
```

## Versioning

The v0.4.0 release target consolidates quantum-kernel classification, one-class anomaly detection, kernel regression, cached fidelity evaluation and generic pipeline composition.


## Novelty detectors

`QuantumOneClassSVM` follows scikit-learn's One-Class SVM label convention
(`+1` normal, `-1` anomaly) and also provides `anomaly_score(X)`.

`QuantumAnomalyDetector` is a separate similarity-threshold detector. Its
training method requires two normal sets:

```python
from qmlib.models import QuantumAnomalyDetector

detector = QuantumAnomalyDetector(
    kernel=kernel,
    threshold_quantile=0.99,
)
detector.fit(reference_normal, calibration_normal)
anomaly_scores = detector.score_samples(X_test)
labels = detector.predict(X_test)  # 1 anomaly, 0 normal
```

The detector's two-input `fit(reference_X, calibration_X)` method is
intentionally different from the `fit(X, y)` interface used by classifiers
and regressors. It is not currently a drop-in final estimator for
`QMLPipeline`.
