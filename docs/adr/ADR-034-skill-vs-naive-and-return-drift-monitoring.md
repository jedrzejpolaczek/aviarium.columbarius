# ADR-034: Monitoring by Skill Against Naive and by Return Drift

**Date:** 2026-09-30
**Status:** Accepted — supersedes the MAPE and drift parts of [ADR-020](ADR-020-monitoring-and-retraining-architecture.md)

## Context

ADR-020 designed three retrain signals: ban/unban events, a MAPE alert and a
data-drift report. The 2026-09-29 investigation (ADR-020, Amendment) found that
only the ban signal ever worked:

- **MAPE was never fed.** Nothing called `save_predictions`, and nothing had
  since the first commit. Even if it had been:
  - `gold_predictions` was not in `GoldStorage._KNOWN_GOLD_TABLES`, so the nightly
    rebuild would have dropped it;
  - the `CURRENT_DATE - 7` window could never hold the three matured dates the
    alert needed;
  - the 30% price-level threshold sat far above the model's real error of 9–12%.
    That error is dominated by sub-€1 cards, where one cent is 10%.
- **Drift was never called.** On real history it scored 0.001–0.010 against its
  0.1 threshold, including across the July 2026 price-feed switch. The price
  *level* distribution of ~80 000 cards barely moves in months.

The same week, honest walk-forward CV showed LightGBM tying or losing to the
naive "price will not change" forecast. A monitor that cannot tell a model worse
than naive from a good one is watching the wrong quantity.

## Decision

1. **Record what the served model predicts, every day.**
   `scripts/check_and_retrain.py` scores all cards on the latest Gold snapshot
   with the served model, following the API's own feature path. It stores the
   predicted 7-day log return in a separate DuckDB file
   (`MONITORING_DB_PATH`, default `data/monitoring/monitoring.duckdb`). A separate
   file because Gold is rebuilt nightly and opened read-only by this job and by
   the API. The file is backed up, since this history cannot be recomputed.

2. **Measure skill against naive, not error size.** Once a prediction date's
   t+7 prices exist, its Tier 1 cards are scored:
   `skill = 1 − MAE(model) / MAE(naive)`, where naive predicts a zero return.
   No calendar window is used: every matured date is scored.

3. **Alert and retrain when skill < −5% on 3 evaluated dates in a row.**
   Calibrated on 14 live dates with walk-forward models (no lookahead):
   - normal weeks scored −0.9% … +0.8%;
   - models trained across the July feed switch scored −18% … −27%.

   The trigger joins ban/unban events in `should_retrain`. The existing
   promotion guard still refuses to promote a retrained model that is not
   better.

4. **Measure drift on 7-day returns, not price levels.** The current week of
   matured returns is compared with the four weeks before it, using the
   Wasserstein distance normalised by the reference spread. It is computed with
   numpy, so Evidently is no longer a dependency. The threshold is 0.15,
   calibrated on every snapshot date in the history:
   - normal weeks scored 0.03–0.08, with one 0.103;
   - the feed switch scored 1.59, 0.80 and 0.69 on its first days.

   Drift **alerts only**. A changed market does not by itself make the model
   wrong; the skill check measures whether it did.

5. **A failed monitoring step alerts and is recorded, never fatal.** The retrain
   decision and the heartbeat still run.

## Consequences

### Positive
- The monitor answers the question that matters: is the served model better
  than doing nothing? It would have fired on the model served from July to
  September, which was trained across the feed switch.
- Every alert traces back to a model run through `model_run_id` in the stored
  predictions.
- A price-feed incident like July's is visible within a week, as return drift.
- One dependency fewer, and no reliance on Evidently's changing API.

### Negative
- Nothing can be scored until 7 days after the first recorded run, and the
  alert needs 3 matured dates, so it becomes armed about 9–10 days after
  deployment.
- At today's model quality (a tie with naive) the skill sits near zero. A
  genuinely worse model will fire the alarm, and retraining may not fix it:
  the promotion guard then keeps the incumbent, and the operator decides
  whether to serve predictions at all (runbook §6).
- Thresholds are calibrated on four months of history with a data gap. They
  should be re-checked once more history exists.

### Neutral
- The ban/unban trigger is unchanged, including its exact-date match.
- `cv_mape_tier1`, used by the promotion guard, is a separate CV metric and is
  not affected.
