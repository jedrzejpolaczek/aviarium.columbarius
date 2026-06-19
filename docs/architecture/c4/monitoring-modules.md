# C4 — Monitoring Modules Code

The monitoring system detects prediction drift and ban events, triggering automatic retraining decisions via a priority-based logic that first checks for ban events (immediate retrain) and then evaluates rolling MAPE degradation over consecutive days.

```mermaid
classDiagram
  class mape_tracker {
    <<module>>
    save_predictions(conn, predictions_df, model_run_id, snapshot_date) None
    compute_rolling_mape(conn) float
    is_mape_alert(conn) bool
  }

  class event_trigger {
    <<module>>
    has_ban_event_today(conn, check_date) bool
    get_todays_events(conn) list
  }

  class drift {
    <<module>>
    fetch_prices_for_period(conn, start_date, end_date) DataFrame
    compute_drift_report(conn) dict
  }

  class retraining {
    <<module>>
    should_retrain(conn) tuple~bool, str~
    retrain(conn) str
    promote_to_production(run_id) None
  }

  retraining ..> mape_tracker : calls compute_rolling_mape,is_mape_alert
  retraining ..> event_trigger : calls has_ban_event_today,get_todays_events
```

## Module Responsibilities

| Module | Responsibility |
|--------|-----------------|
| `mape_tracker` | Persists predictions to gold_predictions and computes rolling 7-day MAPE against actual prices; emits MAPE alerts when threshold breached for consecutive days |
| `event_trigger` | Queries gold_events table to detect ban/unban events occurring on a specific date |
| `drift` | Fetches historical EUR prices and runs Evidently KS-test comparing reference (last 30 days) vs current (last 7 days) price distributions |
| `retraining` | Orchestrates retraining decisions, executes LightGBM model training via walk-forward CV, logs to MLflow, and promotes winning models to production |

## Retraining Decision Logic

The `should_retrain()` function checks two independent signals in priority order:

1. **Ban/unban event today** (queries gold_events) → immediate retrain signal
2. **MAPE > 30% for 3 consecutive days** (via `compute_rolling_mape()` and `is_mape_alert()`) → drift-induced retrain signal

If either signal fires, `retrain()` is invoked to build a fresh LightGBM model using walk-forward cross-validation on the snapshot and training on full historical data. The new model is logged to MLflow with a new `run_id`. 

`promote_to_production(run_id)` only promotes a model if its CV MAPE is better than the currently registered Production model in MLflow Registry, ensuring quality gates are enforced.

## Data Sources

| Module | Reads From | Purpose |
|--------|-----------|---------|
| `mape_tracker` | gold_predictions, gold_price_features | Retrieve past predictions and 7-day actual prices for MAPE computation |
| `event_trigger` | gold_events | Detect ban/unban events occurring today |
| `drift` | gold_price_features | Fetch historical EUR prices for distribution analysis |
| `retraining` | gold_events, gold_predictions, gold_price_features | Aggregate signals for retraining decision; source training data for model rebuild |
