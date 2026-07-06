# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.1.1] - 2026-07-06

### Fixed
- Isolated MLflow-dependent tests (`tests/ml/training/test_tracking.py`) into a separate pytest process — running them inside the full suite caused a fatal crash (ADR-026).

### Added
- Dependabot configuration for `uv`, `npm` (frontend), and GitHub Actions.
- README "Results" section linking the per-phase analysis write-ups in `notebooks/`.
- `scripts/check_and_retrain.py` — scheduled drift/MAPE check with conditional retraining, replacing the need to retrain unconditionally.
- `scripts/rollback_model.py` — manual production model rollback via MLflow Registry alias.
- `docs/runbooks/model-incidents.md` — operational runbook for prediction-service incidents.

## [0.1.0] - 2026-07-06

### Added
- Bronze/Silver/Gold medallion pipeline for MtG card price data
- LightGBM model training with walk-forward cross-validation
- FastAPI price prediction endpoint
- Docker Compose setup for API and web UI
