"""Records the served model's daily predictions and scores them against naive.

Why this replaces the old MAPE tracker (ADR-034):
    The old tracker waited for rows that nothing ever wrote, stored them in a
    Gold table the nightly rebuild deleted, windowed on ``CURRENT_DATE`` so the
    alert could never collect its three days, and alerted on a 30% price-level
    error the model never approaches even when it is useless. It asked "is the
    model's error large?" when the question that matters is "is the model worse
    than predicting no change at all?".

What happens now, once per ``scripts/check_and_retrain.py`` run:
    1. :func:`record_daily_predictions` scores every card on the latest Gold
       snapshot with the served model — the same feature path the API uses —
       and stores the predicted 7-day log return in the monitoring database.
    2. Seven days later the actual return exists in Gold. :func:`daily_skill`
       compares, per prediction date, the model's MAE with the naive
       forecast's MAE (naive predicts a zero return) on the same Tier 1 cards:
       ``skill = 1 - model_mae / naive_mae``. Positive means the model beats
       "no change"; negative means it is worse.
    3. :func:`is_degraded` fires when the latest ``consecutive`` evaluated
       dates are all below ``threshold``.

Calibration (2026-09-30, walk-forward models, 14 live dates): on normal weeks
skill stays within -0.9% .. +0.8%; models trained across the July 2026
price-feed switch scored -18% .. -27%. The default threshold of -5% sits
between the two with room on both sides.

Storage lives in its own DuckDB file (``MONITORING_DB_PATH``), not in Gold:
Gold is rebuilt every night and is opened read-only by this job and the API.
Prediction history cannot be recomputed later (it is what the model *said* on
the day), so ``scripts/backup_data.py`` copies this file too.
"""

import os
from dataclasses import dataclass

import duckdb
import numpy as np
import pandas as pd

from src.ml.features.lag import build_target
from src.ml.models.tiered import TIER1_MAX_EUR

MONITORING_DB_PATH = os.getenv(
    "MONITORING_DB_PATH", "data/monitoring/monitoring.duckdb"
)

SKILL_THRESHOLD = -0.05
"""Alert when the model is more than 5% worse than naive (see calibration above)."""

CONSECUTIVE_DAYS = 3
"""Evaluated prediction dates in a row that must breach the threshold."""

_CREATE_PREDICTIONS = """
CREATE TABLE IF NOT EXISTS predictions (
    uuid                 VARCHAR NOT NULL,
    snapshot_date        DATE NOT NULL,
    eur                  DOUBLE,
    predicted_log_return DOUBLE NOT NULL,
    model_run_id         VARCHAR NOT NULL,
    created_at           TIMESTAMP DEFAULT current_timestamp
)
"""


def ensure_predictions_table(mon_conn: duckdb.DuckDBPyConnection) -> None:
    """Create the ``predictions`` table if it does not exist. Idempotent."""
    mon_conn.execute(_CREATE_PREDICTIONS)


def save_predictions(
    mon_conn: duckdb.DuckDBPyConnection,
    predictions: pd.DataFrame,
    snapshot_date: str,
    model_run_id: str,
) -> int:
    """Store one day's predictions, replacing any earlier ones for that day and model.

    Args:
        mon_conn:      Writable connection to the monitoring database.
        predictions:   ``uuid``, ``eur`` and ``predicted_log_return`` columns.
        snapshot_date: The Gold snapshot the predictions were made on.
        model_run_id:  MLflow run of the model that made them.

    Returns:
        Number of rows stored.
    """
    ensure_predictions_table(mon_conn)
    frame = predictions[["uuid", "eur", "predicted_log_return"]].copy()
    frame["snapshot_date"] = pd.Timestamp(snapshot_date).date()
    frame["model_run_id"] = model_run_id
    mon_conn.register("_predictions_in", frame)
    try:
        mon_conn.execute("BEGIN TRANSACTION")
        mon_conn.execute(
            "DELETE FROM predictions WHERE snapshot_date = ? AND model_run_id = ?",
            [snapshot_date, model_run_id],
        )
        mon_conn.execute(
            "INSERT INTO predictions "
            "(uuid, snapshot_date, eur, predicted_log_return, model_run_id) "
            "SELECT uuid, snapshot_date, eur, predicted_log_return, model_run_id "
            "FROM _predictions_in"
        )
        mon_conn.execute("COMMIT")
    except Exception:
        mon_conn.execute("ROLLBACK")
        raise
    finally:
        mon_conn.unregister("_predictions_in")
    return len(frame)


def record_daily_predictions(
    gold_conn: duckdb.DuckDBPyConnection,
    mon_conn: duckdb.DuckDBPyConnection,
    model_run_id: str,
    snapshot_date: str,
) -> int:
    """Score every card on *snapshot_date* with the served model and store it.

    Uses the same steps as the API's startup (``build_inference_features`` →
    ``fit_transform_features`` → ``Booster.predict``), so what is recorded is
    what ``/predict`` would have answered that day.
    """
    # Deferred: MLflow and the feature pipeline are heavy, and the rest of this
    # module (daily_skill, is_degraded) must stay cheap to import and test.
    from src.ml.features.pipeline import (
        build_inference_features,
        fit_transform_features,
    )
    from src.ml.training.tracking import load_model_from_mlflow

    model = load_model_from_mlflow(model_run_id)
    X_raw = build_inference_features(gold_conn, snapshot_date)
    X, _, _ = fit_transform_features(X_raw)
    predictions = pd.DataFrame(
        {
            "uuid": X_raw["uuid"].to_numpy(),
            "eur": X_raw["eur"].to_numpy(dtype=float),
            "predicted_log_return": model.predict(X),
        }
    )
    return save_predictions(mon_conn, predictions, snapshot_date, model_run_id)


def daily_skill(
    mon_conn: duckdb.DuckDBPyConnection,
    gold_conn: duckdb.DuckDBPyConnection,
    max_price: float = TIER1_MAX_EUR,
) -> pd.DataFrame:
    """Model vs naive MAE for every prediction date whose actuals have arrived.

    The actual 7-day log return comes from ``build_target`` — the same
    definition the model is trained on — so a date appears here only once its
    t+7 snapshot exists in Gold. There is no calendar window: every recorded
    date is scored, and :func:`is_degraded` looks at the latest ones.

    Returns:
        ``snapshot_date``, ``model_run_id``, ``n``, ``model_mae``,
        ``naive_mae``, ``skill`` — one row per prediction date and model,
        oldest first. Empty when nothing has matured yet.
    """
    columns = ["snapshot_date", "model_run_id", "n", "model_mae", "naive_mae", "skill"]
    ensure_predictions_table(mon_conn)
    recorded = mon_conn.execute(
        "SELECT DISTINCT snapshot_date, model_run_id FROM predictions ORDER BY 1, 2"
    ).fetchall()
    rows = []
    for snapshot_date, model_run_id in recorded:
        actual = build_target(gold_conn, str(snapshot_date))
        if actual.empty:
            continue  # t+7 not collected yet
        preds = mon_conn.execute(
            "SELECT uuid, eur, predicted_log_return FROM predictions "
            "WHERE snapshot_date = ? AND model_run_id = ?",
            [snapshot_date, model_run_id],
        ).df()
        scored = preds.merge(actual[["uuid", "log_return_7d"]], on="uuid")
        scored = scored[
            scored["log_return_7d"].notna()
            & scored["eur"].notna()
            & (scored["eur"] < max_price)
        ]
        if scored.empty:
            continue
        model_mae = float(
            np.mean(np.abs(scored["predicted_log_return"] - scored["log_return_7d"]))
        )
        naive_mae = float(np.mean(np.abs(scored["log_return_7d"])))
        skill = 1.0 - model_mae / naive_mae if naive_mae > 0 else float("nan")
        rows.append(
            (snapshot_date, model_run_id, len(scored), model_mae, naive_mae, skill)
        )
    return pd.DataFrame(rows, columns=columns)


def is_degraded(
    skill_df: pd.DataFrame,
    threshold: float = SKILL_THRESHOLD,
    consecutive: int = CONSECUTIVE_DAYS,
) -> bool:
    """True when the latest *consecutive* evaluated dates all have skill below *threshold*.

    "Evaluated dates" are prediction dates with matured actuals, so a missed
    daily run shortens the streak rather than breaking the check. A NaN skill
    (no price moved at all that week) never counts as a breach.
    """
    if len(skill_df) < consecutive:
        return False
    recent = skill_df.sort_values("snapshot_date")["skill"].tail(consecutive)
    return bool((recent < threshold).all())


@dataclass(frozen=True)
class SkillStatus:
    evaluated_dates: int
    latest_date: str | None
    latest_skill: float | None
    degraded: bool


def skill_status(skill_df: pd.DataFrame) -> SkillStatus:
    """Summary for the status file and alert text."""
    if skill_df.empty:
        return SkillStatus(0, None, None, False)
    last = skill_df.sort_values("snapshot_date").iloc[-1]
    return SkillStatus(
        evaluated_dates=len(skill_df),
        latest_date=str(last["snapshot_date"]),
        latest_skill=float(last["skill"]),
        degraded=is_degraded(skill_df),
    )
