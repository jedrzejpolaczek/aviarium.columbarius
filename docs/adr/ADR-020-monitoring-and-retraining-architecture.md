# ADR-020: Monitoring and Automated Retraining Architecture

## Context

A deployed ML model degrades over time.  In the MTG price prediction domain
there are two distinct degradation modes:

1. **Gradual drift** — the statistical properties of card prices shift slowly
   as the meta-game evolves.  A model trained three months ago may have
   slightly wrong feature importances.  Standard MAPE monitoring detects this.

2. **Abrupt shock** — a card banned in a competitive format drops 50–90% in
   value within 24 hours.  MAPE-only monitoring would miss this for up to
   three days (the consecutive-days window), during which every Tier 1 or 2
   prediction for that card is materially wrong.

Two monitoring approaches were considered:

**Option A — MAPE-only**: compute rolling MAPE daily; alert after N
consecutive days above threshold.

**Option B — Dual-signal**: combine MAPE with an event table
(``gold_events``) that records ban/unban announcements.  An event triggers
immediate retraining; MAPE remains the fallback for gradual drift.

## Decision

Use **Option B — Dual-signal monitoring** with four modules:

| Module             | Responsibility                                        |
|--------------------|-------------------------------------------------------|
| `mape_tracker.py`  | Write predictions to `gold_predictions`; compute rolling MAPE |
| `event_trigger.py` | Query `gold_events`; detect same-day bans/unbans       |
| `drift.py`         | Evidently report on `eur`/`log_eur` distribution shift |
| `retraining.py`    | Orchestrate: check signals → retrain → promote         |

`should_retrain` checks event signal first (immediate trigger), then MAPE
threshold (3-day streak), then returns ``False`` if neither is met.

## Prediction Lineage via ``gold_predictions``

Every price prediction is persisted with its ``model_run_id`` before being
served.  Seven days later the same rows are joined against
``gold_price_features`` to compute actual MAPE.  This gives full lineage:

```
alert date → prediction snapshot_date → model_run_id → MLflow run → training data snapshot
```

Without this table, MAPE tracking is impossible because predictions are
ephemeral (returned via HTTP and not stored).

## Why Consecutive-Days Threshold (3 days), Not Single-Day

A single day of high MAPE may reflect a genuine model failure or a buyout
spike (one player buying thousands of copies of a card) that reverses the
next day.  Requiring three consecutive days above 30% filters out most
single-day volatility events while still catching real degradation within
a week.  The threshold values (30%, 3 days) are initial defaults and should
be recalibrated after 90 days of production data.

## Why Evidently for Drift Detection

Evidently computes per-column statistical tests (Kolmogorov-Smirnov for
continuous features) and aggregates into an overall ``dataset_drift`` boolean.
Alternatives considered:

- **PSI (Population Stability Index)**: requires binning — sensitive to bin
  choice, non-trivial for highly skewed price distributions.
- **Custom KS test**: would replicate what Evidently already provides with
  a report, HTML output, and column-level drill-down.
- **MAPE alone**: MAPE is a lagging indicator (waits 7 days for actuals);
  Evidently drift fires immediately when the distribution shifts.

Evidently drift is a *leading indicator* that fires before MAPE rises,
giving the team an earlier warning.  It is not currently wired into the
``should_retrain`` decision (only logged) but the infrastructure is in place
for future integration.

## MLflow Registry Promotion Strategy

Three-model scenario after each retrain:

1. **New run** — logged to Staging (registered but not aliased).
2. **Compare** — new model's CV MAPE Tier 1 vs current Production's
   `cv_mape_tier1` metric.
3. **Promote** — if new ≤ current, set ``production`` alias; old version moves
   to Archived automatically.  If new > current, leave Production unchanged
   and log a warning.

Using the MLflow 2.x alias API (``set_registered_model_alias``) instead of
the deprecated stage-transition API.  A ``production`` alias is the single
source of truth for which model the API server loads.  Rollback is a single
alias reassignment, not a code deployment.

## Consequences

### Positive
- Format ban detected same day (not 3+ days later via MAPE).
- Full prediction lineage enables post-mortem analysis of any alert.
- MLflow Registry provides instant rollback without code changes.
- Evidently drift report is a usable artefact for ops review (HTML + dict).

### Negative
- ``gold_events`` must be populated by the ETL pipeline;
  if `build_events()` is broken, event-triggered retrain silently degrades
  to MAPE-only.
- Evidently adds a dependency (``evidently>=0.4.0``); the Evidently API has
  changed across major versions — the ``DataDriftPreset`` path is pinned to
  the 0.4+ dict structure.
- ``retrain()`` duplicates the feature preparation logic from ``app/main.py``
  and notebook 02.  This is intentional (the modules are independent by ADR-016)
  but is a maintenance surface if the feature set changes.
