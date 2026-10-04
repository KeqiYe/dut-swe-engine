# Contributing

Use Python 3.10 or later. From a checkout, create and activate a virtual
environment, then install the development dependencies:

```sh
python -m pip install -e '.[dev]'
python -m pytest -q
```

The suite checks the NumPy and Warp CPU backends; CUDA tests are skipped when
no NVIDIA device is available. No viewer or MATLAB installation is required.
For numerical changes, report the backend, precision, mesh, initial state,
time step, and duration. Add a regression that checks a physical invariant,
an independent reference, convergence, or rejected-step state preservation.

Known stability limits are documented in `validation/REPORT.md` and the
README. Please preserve those limits in documentation until new evidence
supports changing them. A small energy drift alone is not evidence that a
deformed shoreline mesh is accurate.

Submit changes through a pull request with a short description and relevant
test results. Contributions are accepted under the repository's MIT license.
