# Runbook: Model & Prediction-Service Incidents

Operational reference for the three incidents this project can currently
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
   a known-good run (see Section 3 below for how to find one) and restart
   the container: `docker compose -f docker/docker-compose.yml up -d --build api`.

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

**Cause:** either `"reason": "gold_db_missing"` (ETL pipeline hasn't run
yet) or `"reason": "no_snapshot"` (`gold_price_features` is empty).

**Fix:** run `make pipeline`, confirm `data/gold/cards.duckdb` exists and
`gold_price_features` is populated, then re-run `make monitor`.

## Known limitation

There is no automated alerting (Slack/email/PagerDuty) wired to any of
the above — `logs/last_check_status.json` and the container logs must be
checked manually or by whatever external tooling is set up to watch them.
Adding real paging requires credentials (webhook URL, SMTP, etc.) this
project does not currently have configured; treat manual log/status
checking as the interim process until that changes.
