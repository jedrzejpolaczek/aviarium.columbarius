# Runbook: Model & Prediction-Service Incidents

Operational reference for the incidents this project can currently
detect or cause. Pairs with [ADR-020](../adr/ADR-020-monitoring-and-retraining-architecture.md)
(monitoring architecture) and [ADR-018](../adr/ADR-018-tier-based-model-selection.md)
(tiered pricing).

## 1. `/predict` and `/underpriced` return 503

**Symptom:** `GET /health` returns `{"status": "degraded", "model_loaded": false, ...}`.

**Cause:** `MODEL_RUN_ID` is unset, or `load_model_from_mlflow` failed at
startup (see `app/main.py` lifespan step 5 — failures are caught and logged
at WARNING, not fatal).

**Fix:**
1. Check the API container logs for `Model load failed (<run_id>): <exc>`.
2. Confirm the run_id exists and has a `model` artefact:
   `uv run python -c "import mlflow; mlflow.set_tracking_uri('sqlite:///mlflow.db'); print(mlflow.get_run('<run_id>'))"`
3. If the run_id is wrong or the artefact is missing, set `MODEL_RUN_ID` to
   a known-good run (see Section 2, step 1 for how to list available runs/versions)
   and restart the container: `docker compose -f docker/docker-compose.yml up -d --build api`.

## 1b. `/health` returns 503 with `"features_loaded": false`

**Symptom:** `GET /health` returns `{"status": "degraded", "features_loaded": false, ...}`.
Unlike Section 1 (model load failed but features are fine), here `/cards`,
`/predict`, `/similar`, and `/underpriced` **all** return 503 — there is no
feature matrix to serve from.

**Cause:** Building `X_all`/`X_all_t` or the `CardSimilarityIndex` raised at
startup (see `app/main.py` lifespan — the exception is caught and logged at
ERROR, not fatal, and triggers a desktop alert via `src.monitoring.alerts`).
Common causes: a Gold schema change that the feature pipeline doesn't
expect yet, or corrupt/partial Gold tables from an interrupted ETL run.

**Fix:**
1. Check the API container logs for `Feature matrix / similarity index
   build failed — starting in degraded mode: <exc>`.
2. Check `logs/alerts.jsonl` for a corresponding `"API startup degraded"`
   entry around the same timestamp.
3. Re-run `make pipeline` to rebuild Gold from a clean state, then restart
   the API container: `docker compose -f docker/docker-compose.yml up -d --build api`.
4. If the error persists after a clean ETL run, it is a real code/schema
   bug in the feature pipeline (`src/ml/features/pipeline.py`) — not fixed
   by re-running the pipeline.

## 2. `logs/last_check_status.json` shows `"result": "retrained"` and predictions look wrong afterwards

**Symptom:** `scripts/check_and_retrain.py` (scheduled daily — see README
"Monitoring & Scheduled Retraining") retrained and auto-promoted a new
model, but predictions regressed.

**Cause:** `retrain()` only compares CV Tier-1 MAPE before promoting
(`src/monitoring/retraining._compare_and_promote`) — it does not catch
regressions that only show up on live traffic.

**Fix — roll back to the previous version:**
1. Find the previous production run:
   ```bash
   uv run python -c "
   import mlflow
   mlflow.set_tracking_uri('sqlite:///mlflow.db')
   client = mlflow.tracking.MlflowClient()
   for v in client.search_model_versions(\"name='mtg_price_model'\"):
       print(v.version, v.run_id, v.aliases, v.creation_timestamp)
   "
   ```
2. Roll back to the previous known-good version:
   `uv run python -m scripts.rollback_model --version <previous_version>`
3. Update `MODEL_RUN_ID` in `docker/.env` to that version's `run_id` and
   restart the API container.

## 3. `logs/last_check_status.json` shows `"result": "error"`

**Cause:** one of three reasons in the `"reason"` field:
- `"gold_db_missing"` — the ETL pipeline hasn't run yet.
- `"no_snapshot"` — `gold_price_features` is empty.
- `"retrain_failed"` — a retrain trigger fired, but `retrain()` itself
  raised an exception (CV, feature build, LightGBM fit, or MLflow logging
  failure). The status file's `"error"` field holds the exception message.

**Fix:**
- For `gold_db_missing` / `no_snapshot`: run `make pipeline`, confirm
  `data/gold/cards.duckdb` exists and `gold_price_features` is populated,
  then re-run `make monitor`.
- For `retrain_failed`: read the `"error"` field in
  `logs/last_check_status.json` and the full traceback in the day's
  `logs/pipeline_*.log` (or console output if run interactively) to
  diagnose the underlying failure — this is not fixed by re-running the
  pipeline.

## 4. `logs/last_pipeline_status.json` shows `"result": "error"`

**Symptom:** The daily `scripts/run_pipeline.py` run failed; a desktop
alert (see `src.monitoring.alerts`) and a JSON status file were written.

**Fix:**
1. Read the `"error"` field in `logs/last_pipeline_status.json` and the
   full traceback in the day's `logs/pipeline_*.log`.
2. Common causes: a source API (Scryfall/MTGJson) returned a persistent
   error after exhausting the 5 retry attempts (ADR-014), or a schema
   change broke a Pydantic validator.
3. Fix the underlying issue, then re-run `make pipeline` manually to
   confirm before waiting for the next scheduled run.

## 5. Desktop alert / `logs/alerts.jsonl` entry titled "Backup failed"

**Symptom:** `scripts/backup_data.py` (scheduled daily via `make backup`,
after `make pipeline && make monitor` — see README "Monitoring & Scheduled
Retraining") raised, and a "Backup failed" alert was recorded.

**Cause:** either none of the backup sources exist yet (fresh checkout,
ETL hasn't run — `run_backup` raises `FileNotFoundError`), or a copy failed
partway through (disk full, permission error on `--backup-dir` — raises
`OSError`; the partially-written snapshot directory is cleaned up
automatically, so a failed run never leaves a corrupt backup behind).

**Fix:**
1. Read the alert message in `logs/alerts.jsonl` (or the container/console
   logs) for the exact exception.
2. If no sources exist: run `make pipeline` and `make train` first, then
   retry `make backup`.
3. If a copy failed: check free disk space and write permissions on
   `--backup-dir` (default `backups/` at the project root), then retry.
4. This is not model-serving-critical — a failed backup does not affect
   `/predict` or retraining — but should be fixed before the next scheduled
   run so backup coverage doesn't have a gap.

## Alerting

`scripts/run_pipeline.py`, `scripts/check_and_retrain.py`,
`scripts/backup_data.py`, and the API's `lifespan` degraded-mode path all
call `src.monitoring.alerts.send_alert` on failure, which (1) appends a
durable record to `logs/alerts.jsonl` and
(2) best-effort shows a desktop notification via `plyer` — no external
account or credentials required. The desktop notification only appears if
the machine is logged in and unlocked when the scheduled task runs; the
JSONL log is the reliable source of truth and does not depend on that.
In the Docker deployment, `docker/docker-compose.yml` mounts `logs/` from
the host into the container (`../logs:/app/logs`) and the Dockerfile
pre-creates that directory owned by the non-root `app` user, so the
API container's alerts land in the same `logs/alerts.jsonl` on the host
as the scheduled scripts' — check there first, container logs second
(the desktop notification itself never fires inside a headless container).
Set `HEARTBEAT_URL` (a healthchecks.io-style ping URL) to detect the
scheduled task silently not running at all — `check_and_retrain.py` pings
it on every run, success or failure, so a missing ping (not just a
`result: error` status) is itself the alert.

There is still no remote paging (Slack/email/PagerDuty) — `logs/alerts.jsonl`,
`logs/last_check_status.json`, `logs/last_pipeline_status.json`, and the
container logs are the full observability surface today. Adding real
remote paging requires credentials (webhook URL, SMTP, etc.) this project
does not currently have configured.
