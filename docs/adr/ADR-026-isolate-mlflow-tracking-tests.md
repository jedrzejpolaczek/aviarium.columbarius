# ADR-026: Isolate MLflow Tracking Tests Into a Separate pytest Process

**Date:** 2026-07-06
**Status:** Accepted

## Context

`tests/ml/training/test_tracking.py` exercises `src/ml/training/tracking.py`
against a fresh SQLite-backed MLflow tracking store per test (each test
points `MLFLOW_TRACKING_URI` at its own `tmp_path` database via the
`mlflow_tmp` autouse fixture).

Run in isolation, all 18 tests in this file pass reliably. Run as part of
the full suite (`uv run pytest`, ~1050 other tests), the process crashes
with a fatal (non-pytest-reported) error partway through this file, inside
`mlflow.tracking.fluent.set_experiment`. The crash does not reproduce when
the file is run alone, which rules out a defect in the tests' own logic —
it is a resource/state accumulation issue in MLflow's internal tracking
store registry across a very large number of distinct SQLite tracking URIs
in a single process. Root-causing the exact internal MLflow behaviour is
not warranted for a locally-run test suite.

## Decision

Run `tests/ml/training/test_tracking.py` as a second, separate `pytest`
invocation, excluded from the main suite via `--ignore`:

```
uv run pytest --ignore=tests/ml/training/test_tracking.py
uv run pytest tests/ml/training/test_tracking.py
```

This applies to `Makefile`'s `test` target, `.github/workflows/ci.yml`, and
`scripts/pre-push` — anywhere the full suite is invoked.

## Consequences

- `make test` / CI / pre-push now run two `pytest` processes instead of
  one. Total wall-clock time increases slightly (~2s for a second pytest
  startup) but the suite no longer crashes.
- New MLflow-heavy test files should either live in this same file (so
  they're covered by the isolation) or get evaluated individually if they
  show the same instability under the full suite.
- If this recurs (e.g. a new module needs the same treatment), consider
  filing the resource-accumulation behaviour upstream with MLflow rather
  than repeating this workaround per-file.
