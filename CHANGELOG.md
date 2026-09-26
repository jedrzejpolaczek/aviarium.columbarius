# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- `.env.example` — annotated template for every environment variable the application reads (`MODEL_RUN_ID`, `GOLD_DB_PATH`, `MLFLOW_TRACKING_URI`, `ADMIN_TOKEN`, `CORS_ORIGINS`, `ALERT_WEBHOOK_URL`, `HEARTBEAT_URL`), documenting the degraded behaviour when each is left unset.
- `docker/docker-compose.staging.yml` — full standalone staging environment (API 8100, frontend 3100, own `data-staging/` Gold copy and `logs-staging/`), runnable alongside production. Staging deliberately copies the Gold database rather than sharing the production file, so a staging failure cannot take production down; MLflow stores are shared read-only since artifacts are immutable.
- README "Measured results" section with the per-tier MAE comparison (LightGBM wins Tier 1, loses Tier 2 by ~7%, ties Tier 3), marked as a single-split estimate taken at `gold_snapshot_date = 2026-07-02` and explicitly flagged as pending re-measurement under cross-validation.
- README explanation of *why* AGPL-3.0 was chosen and what its network clause means for anyone hosting this as a service.
- `ML_FINDINGS.md` T8 section filled with the Optuna hyperparameters actually logged in MLflow (`tuned_lightgbm_nb04`, 2026-07-10), including why its 85% MAPE is expected and why it is not directly comparable to the per-tier test MAE in T6.
- `DuckDBRepository` (ADR-029) — shared connection creation plus `get_tables`/`query_df`; migrated `app/main.py`, `app/dependencies.py`, `health.py`, `train_model.py`, and `check_and_retrain.py` off ad-hoc connection handling.
- ADR-027 (TF-IDF card embeddings), ADR-028 (SHAP interpretability), ADR-030 (shared idiom conventions — indexes intentional cross-module repetition so future audits don't re-flag it).
- Vitest test infrastructure for the frontend (smoke tests for `App.tsx`, `api.ts`).
- Test coverage for previously untested paths: `rollback_model.py`, `check_and_retrain.py`/`retrain()` against a real MLflow store, `run_pipeline`/`check_health`/`kill_proc` scripts, `FormatStaple`/`TournamentResult` dataclasses, source registry, tournament HTML parsing, app startup lifespan.
- Production-readiness hardening (ADR-031, ADR-032, ADR-033), following a full 5-audit maturity review:
  - `ALERT_WEBHOOK_URL` — optional Slack/Discord/Mattermost-compatible webhook channel on `send_alert`, alongside the existing JSONL log and desktop notification (ADR-031).
  - `HEARTBEAT_URL` — dead-man's-switch ping in `check_and_retrain.py`, fired on every run (success or `/fail`) so a scheduled task that silently stops running is itself detectable (ADR-031).
  - Global API exception handler (`register_exception_handlers` in `app/main.py`) — any unhandled exception now logs, alerts, and returns a structured `{"detail": "Internal server error."}` (500) instead of a bare crash (ADR-033).
  - `POST /admin/reload-model` (`app/routers/admin.py`) — token-protected (`X-Admin-Token`/`ADMIN_TOKEN`, constant-time comparison) hot model reload, so `scripts/rollback_model.py`'s registry-alias swap takes effect without a container restart (ADR-032).
  - Backup restorability verification in `scripts/backup_data.py` — each copied DuckDB file is opened read-only and checked for at least one table immediately after copying; a corrupt/empty copy raises `BackupVerificationError` and discards the whole snapshot.
  - Log rotation and grouped pruning in `src/logger.py` — `RotatingFileHandler` (10MB/5 backups) plus count-based pruning of old timestamped log groups (base file + `.log.N` rotation siblings pruned together, `keep_last_logs=90`).
  - `mem_limit`/`cpus` caps on both `docker-compose.yml` services (api: 2g/2.0, frontend: 256m/0.5).

- `src/monitoring/serving_check.py` — compares `MODEL_RUN_ID` (what the API loads) against the MLflow Registry's `production` alias (what promotion comparisons and rollbacks act on) at the start of every `check_and_retrain.py` run, alerting on a mismatch. The two had diverged: the API was serving run `9c1ec7de…` (version 3) while the alias sat on version 2, run `c46d4787…`, so every automatic promotion decision was being measured against a model nobody was serving. The check only reports — which pointer is correct is an operator decision, documented as §2b in `docs/runbooks/model-incidents.md`.

### Changed
- Large cross-module deduplication pass: extracted shared helpers for JSON config loading, Bronze/Silver/Gold guard clauses, HTTP fetch-with-retry, EUR/percent formatting (`formatEur`, `formatPercent`), rolling-window CTE fragments, legality-transition LAG CTEs, and router guards (`require_model`/`require_match`), removing dozens of duplicated implementations.
- Consolidated frontend formatting/label logic (`cardMeta`, `fmtReturn` → `formatPercent`) into single sources of truth.
- Split large functions into named helpers (`lifespan` startup steps, `check_and_retrain` precondition/retrain execution, `get_underpriced_cards` inference/response-building).
- Corrected numerous docstrings and comments found stale against current code (`silver/storage.py`, `underpriced.py`, `health.py`, ADR-014, ADR-029) and fixed the stale "File Structure" section in README.
- `docs/architecture/c3/monitoring.md` corrected to match ADR-020: drift detection is logged but not wired into `should_retrain` (was previously described as one of "three independent signals" feeding the decision equally); fixed broken `ADR-020` links (`020-model-retraining-strategy.md` → `ADR-020-monitoring-and-retraining-architecture.md`).

### Fixed
- Silent-success reporting across the ETL entry point, after a 36-day Scryfall ingestion outage (2026-07-29 → 2026-09-02) was recorded as a successful run every day and found two months late by accident:
  - `run_health_checks` raised `SystemExit(1)` on any FAIL. `SystemExit` derives from `BaseException`, so it bypassed `run_pipeline.py`'s `except Exception`, skipping both `send_alert` and `_write_status` — leaving the *previous* run's `"result": "success"` in `logs/last_pipeline_status.json`. It now returns its results; `scripts/check_health.py` owns the exit code and `daily_pipeline` records a degraded run.
  - Bronze had no freshness check at all (`_check_table_has_rows` counts cumulative tables, so it reported 535 655 healthy rows every day of the outage), and `bronze_scryfall_prices_history` — the table whose absence *was* the outage — was not checked at all. Both price-history tables now get per-source freshness checks.
  - A Bronze source that failed to download was logged and skipped with no trace in the run's outcome. Skipping stays (one dead source must not abort the tiers the survivors can refresh), but `daily_bronze_pipeline` now reports which sources produced zero records.
  - `run_pipeline.py` gained a third outcome, `"degraded"` — alerts, records what failed, exits 1 — plus the `HEARTBEAT_URL` dead-man's-switch that `check_and_retrain.py` already had.
- `prepare_training_data` trained and evaluated on NULL targets despite its docstring promising otherwise. `build_target` returns `LN(1+eur_t7) - LN(1+eur_t0)`, NULL whenever `eur` is missing on either end (~16% of the catalogue has no Cardmarket EUR price). `evaluate_per_tier` aggregates with `np.mean`, and `assign_tier` maps a NaN price to Tier 1, so Tier 1 MAE/MAPE came back NaN for every fold and every model. Measured at `2026-07-01`: 93 386 rows, 14 694 NaN, 78 692 usable — the figure README and `ML_FINDINGS.md` already quote, because the notebooks filtered and CV did not. Tier 1 now reports MAE 0.0214 / 0.0244.
- `_compare_and_promote` treated a missing `cv_mape_tier1` on the incumbent as `+inf`, so every candidate compared favourably and automatic retraining would promote unconditionally — the metric is written only by `log_cv_results` and exists on no run in `mlflow.db`. Promotion now requires a finite value on both sides; the empty-registry bootstrap case is checked first so a first-ever retrain still seeds the registry.
- MLflow artifact paths resolved against whatever the working directory happened to be. `artifact_location` is stored as the relative `mlruns/<id>` — correctly so, since the host and the container need it to mean two different absolute paths — but nothing anchored the cwd, so notebook training wrote into `notebooks/ml_models/mlruns/` while the project-root `mlflow.db` described it as `mlruns/<id>`, and `docker-compose.yml` was pointed at the notebook tree to compensate. `setup_experiment` now refuses to run outside the project root, the notebooks `chdir` there in their setup cell, the six real logged models were copied into `mlruns/1/models/`, and both compose files mount `../mlruns`. No `mlflow.db` migration needed.
- The test suite writing into production artefacts: all 190 records in `logs/alerts.jsonl` came from pytest (`send_alert`'s `alerts_log_path` default was bound at import time and could not be redirected), timestamped pipeline logs were landing in `logs/`, and MLflow test fixtures redirected only the tracking URI — leaving logged models to accumulate in the real `./mlruns` (21 directories in one suite run).
- README's stale "Known gap" warning claiming the latest MLflow runs showed suspicious `mae_test = 0.0`. That degenerate result belonged to the 32-snapshot runs of 2026-07-05/06; the 36-snapshot re-runs of 2026-07-09/10 report real metrics (LightGBM test MAE 0.0539 vs naive 0.0559). The warning had outlived the problem and was understating the project's actual results.
- `.gitignore` not covering per-environment secret files or the staging directories: `.env.*` is now ignored (with `.env.example` explicitly re-included), along with `data-staging/` and `logs-staging/` — the latter would otherwise have made a multi-GB Gold copy committable.
- `ML_FINDINGS.md` recorded `walk_forward_cv_nb03`'s `InsufficientDataError` as a standing limitation. It was correct when logged (2026-07-10) but not since: `generate_folds` gates on calendar span rather than snapshot count, so CV unlocked on 2026-07-15, five days after the last failed attempt. The Gold layer now spans 67 snapshots (2026-05-26 → 2026-09-24) and generates 13 folds; every metric in T6/T8 is a single-split estimate roughly three months out of date.
- The "13 folds" figure above, in `ML_FINDINGS.md` and in `docs/architecture/c4/lightgbm-model.md`, overstated what actually runs. `generate_folds` validates calendar span, not data availability: `walk_forward_cv` silently `continue`s past any fold whose validation window holds no snapshot, or whose snapshot has no exact `t+7` partner for `build_target`. Measured by running it: **13 generated, 2 executed** (folds 1 and 2; folds 5–9 have no validation snapshot at all). Nothing in the output distinguishes the two numbers. Documented as a known gap rather than fixed — counting usable folds would put the honest total below `generate_folds`' own minimum of 3 and block CV entirely, which only gets resolved once Gold covers more than 67 of the 181 days already in Bronze.
- `health.py`'s `/health` endpoint reading `app.state.db`, which no longer existed after the repository migration.
- mypy strict-mode errors from a missing explicit `duckdb` re-export.
- `daily_update` not catching `StorageWriteError` on price snapshots the way `populate` does.
- Windows `PermissionError` in `format_staples` HTML cleanup.
- `rollback_model.py` and `train_model.py` logging inconsistently with the rest of the scripts (stdlib `logging` / console-only instead of `src.logger`).
- Pinned `shap` lower bound and relaxed the `pandas` pin to `>=2.3.3` to resolve a dependency resolver conflict; documented the Intel-Mac constraint.
- Pinned `vitest` to a Vite-5-compatible version and centralized `jest-dom` setup.
- README's ADR table was missing ADR-027 through ADR-030 (present as files, never added to the index).

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
