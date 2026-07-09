# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- `DuckDBRepository` (ADR-029) — shared connection creation plus `get_tables`/`query_df`; migrated `app/main.py`, `app/dependencies.py`, `health.py`, `train_model.py`, and `check_and_retrain.py` off ad-hoc connection handling.
- ADR-027 (TF-IDF card embeddings), ADR-028 (SHAP interpretability), ADR-030 (shared idiom conventions — indexes intentional cross-module repetition so future audits don't re-flag it).
- Vitest test infrastructure for the frontend (smoke tests for `App.tsx`, `api.ts`).
- Test coverage for previously untested paths: `rollback_model.py`, `check_and_retrain.py`/`retrain()` against a real MLflow store, `run_pipeline`/`check_health`/`kill_proc` scripts, `FormatStaple`/`TournamentResult` dataclasses, source registry, tournament HTML parsing, app startup lifespan.

### Changed
- Large cross-module deduplication pass: extracted shared helpers for JSON config loading, Bronze/Silver/Gold guard clauses, HTTP fetch-with-retry, EUR/percent formatting (`formatEur`, `formatPercent`), rolling-window CTE fragments, legality-transition LAG CTEs, and router guards (`require_model`/`require_match`), removing dozens of duplicated implementations.
- Consolidated frontend formatting/label logic (`cardMeta`, `fmtReturn` → `formatPercent`) into single sources of truth.
- Split large functions into named helpers (`lifespan` startup steps, `check_and_retrain` precondition/retrain execution, `get_underpriced_cards` inference/response-building).
- Corrected numerous docstrings and comments found stale against current code (`silver/storage.py`, `underpriced.py`, `health.py`, ADR-014, ADR-029) and fixed the stale "File Structure" section in README.

### Fixed
- `health.py`'s `/health` endpoint reading `app.state.db`, which no longer existed after the repository migration.
- mypy strict-mode errors from a missing explicit `duckdb` re-export.
- `daily_update` not catching `StorageWriteError` on price snapshots the way `populate` does.
- Windows `PermissionError` in `format_staples` HTML cleanup.
- `rollback_model.py` and `train_model.py` logging inconsistently with the rest of the scripts (stdlib `logging` / console-only instead of `src.logger`).
- Pinned `shap` lower bound and relaxed the `pandas` pin to `>=2.3.3` to resolve a dependency resolver conflict; documented the Intel-Mac constraint.
- Pinned `vitest` to a Vite-5-compatible version and centralized `jest-dom` setup.

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
