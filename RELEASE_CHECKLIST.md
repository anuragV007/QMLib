# QMLib v0.4.0 release checklist

Do not tag or publish the release until all applicable checks pass.

## Local environment

- [ ] Confirm the working tree and branch.
- [ ] Back up local uncommitted work.
- [ ] Merge the changes in this patch without deleting local anomaly model files.
- [ ] Confirm `src/qmlib/models/quantum_anomaly_detector.py` exists.
- [ ] Confirm `src/qmlib/models/quantum_one_class_svm.py` exists.
- [ ] Confirm public exports import successfully.

## Validation

```powershell
python -m pip install -e ".[dev]"
python -m py_compile .\src\qmlib\models\quantum_classifier.py
python -m py_compile .\src\qmlib\models\quantum_kernel_ridge.py
python -m py_compile .\src\qmlib\pipeline\base.py
python -m pytest -q
python -m coverage erase
python -m coverage run -m pytest -q
python -m coverage report -m
```

- [ ] Run `examples/compare_feature_maps.py`.
- [ ] Run `examples/quantum_kernel_ridge_regression.py`.
- [ ] Run one small NASA forecasting smoke test.
- [ ] Verify all experiment outputs and config files are reproducible.

## Repository hygiene

- [ ] Do not commit `.venv/`, `__pycache__/`, `.pytest_cache/`, coverage output, `NASA Data/`, or large generated prediction arrays.
- [ ] Keep small aggregate metrics/configuration files only when intentionally versioned.
- [ ] Check `git status --short` before staging.
- [ ] Inspect `git diff --check` and `git diff --stat`.
- [ ] Commit in logical groups.

## Release metadata

- [ ] Confirm `pyproject.toml` and `qmlib.__version__` both report `0.4.0`.
- [ ] Confirm README describes only implemented functionality.
- [ ] Add an explicit license only after choosing and checking the intended license.
- [ ] Push the verified main branch.
- [ ] Create the `v0.4.0` tag only after successful validation.
