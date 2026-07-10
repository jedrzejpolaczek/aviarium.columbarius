# C3 — Monitoring Components

The Monitoring subsystem tracks model performance through two signals that feed the retraining decision — prediction accuracy (MAPE) and business events (bans/unbans) — plus a third, data-drift signal that is computed and logged but **not currently wired into that decision** (ADR-020 explicitly defers this; the infrastructure exists for future integration). These signals feed into the RetrainingOrchestrator (`should_retrain` / `retrain` in `src/monitoring/retraining.py`), which decides when to retrain and promote updated models to production.

```mermaid
C4Component
  title C3 — Monitoring System Components

  Container(gold_db, "Gold DB", "DuckDB", "Persistent data warehouse for predictions, events, and price features")
  System(mlflow, "MLflow Server", "Model registry and deployment tracking")
  Container(data_pipeline, "Data Pipeline", "Triggers monitoring after daily update")

  Container_Boundary(monitoring, "Monitoring System") {
    Component(mape_tracker, "MapeTracker", "Performance Monitor", "Logs predictions to gold_predictions, computes rolling MAPE over sliding window")
    Component(event_trigger, "EventTrigger", "Event Detector", "Detects ban/unban events from gold_events table")
    Component(drift_detector, "DriftDetector", "Drift Monitor", "Runs Evidently KS-test on price distribution")
    Component(orchestrator, "RetrainingOrchestrator", "Decision Engine", "Aggregates signals and triggers retraining pipeline")
  }

  Rel(data_pipeline, mape_tracker, "After daily update, logs new predictions", "triggers")
  Rel(mape_tracker, gold_db, "Writes to gold_predictions, reads rolling window", "reads/writes")
  Rel(event_trigger, gold_db, "Reads gold_events for ban/unban detection", "reads")
  Rel(drift_detector, gold_db, "Reads price distribution from gold_price_features", "reads")
  Rel(data_pipeline, orchestrator, "Triggers monitoring check after daily_pipeline", "triggers")
  Rel(orchestrator, mape_tracker, "Checks rolling MAPE threshold", "queries")
  Rel(orchestrator, event_trigger, "Checks for ban/unban events", "queries")
  Rel(orchestrator, drift_detector, "Checks for price drift", "queries")
  Rel(orchestrator, mlflow, "Promotes retrained model to production alias", "writes")
```

## Components

| Component | Responsibility | ADR |
|-----------|---|---|
| **MapeTracker** | Logs predictions from the data pipeline to `gold_predictions` table and computes rolling MAPE over a sliding window to track prediction accuracy decay | [ADR-020](../../adr/ADR-020-monitoring-and-retraining-architecture.md) |
| **EventTrigger** | Monitors the `gold_events` table for ban/unban events that signal market disruptions requiring model retraining | [ADR-020](../../adr/ADR-020-monitoring-and-retraining-architecture.md) |
| **DriftDetector** | Runs Evidently statistical tests (KS-test) on the price feature distribution in `gold_price_features` to detect data drift. Logged as a leading indicator; **not currently consumed by `should_retrain`** — see ADR-020's "Why Evidently for Drift Detection" section. | [ADR-020](../../adr/ADR-020-monitoring-and-retraining-architecture.md) |
| **RetrainingOrchestrator** | Aggregates signals from the three monitors, evaluates the `should_retrain` decision logic, orchestrates the retraining pipeline, and promotes successful models in MLflow | [ADR-020](../../adr/ADR-020-monitoring-and-retraining-architecture.md) |

## Retraining Decision Logic

`should_retrain` is triggered when **either** of two signals indicates a problem (checked in this order — event first, since it's the immediate trigger; MAPE as the fallback for gradual drift):

- **Event Signal**: Ban/unban event detected in `gold_events` on the same day (market disruption) — immediate trigger.
- **MAPE Threshold**: Rolling MAPE exceeds the configured threshold for 3 consecutive days (model accuracy degraded).

The **Drift Signal** (Evidently KS-test on the price distribution) is computed and logged for ops review but does **not** currently participate in this decision — see ADR-020.

When `should_retrain == True`, `scripts/check_and_retrain.py` (run on a daily schedule, per README's "Monitoring & Scheduled Retraining") orchestrates:
1. Retrains on the latest trainable Gold snapshot.
2. Evaluates the new model against holdout performance thresholds.
3. If performance is acceptable, promotes the model to the `production` alias in MLflow.
4. Writes the outcome to `logs/last_check_status.json` and, on any error branch, calls `src.monitoring.alerts.send_alert` (ADR-031) — durable JSONL log, best-effort desktop notification, and an optional webhook if `ALERT_WEBHOOK_URL` is configured.
5. Pings `HEARTBEAT_URL` (ADR-031) on every run, success or failure — a dead-man's-switch so a scheduled task that silently stops running at all (not just one that errors) is itself detectable.
