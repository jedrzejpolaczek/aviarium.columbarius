"""Tracks model MAPE over time and triggers retraining alerts.

Prediction lineage:
    Every price prediction is written to ``gold_predictions`` with its
    ``model_run_id``.  Seven days later, once actual prices arrive in
    ``gold_price_features``, the same rows are used to compute MAPE.
    This gives full lineage: alert date → prediction date → model run.

Alert logic:
    MAPE > ``threshold`` for ``consecutive_days`` straight → retrain signal.
    Three consecutive days of high error avoids false alarms from isolated
    volatile trading days (e.g. a single buyout spike).

Schema of ``gold_predictions``:
    uuid          VARCHAR        -- card identifier
    snapshot_date DATE           -- date the prediction was made
    predicted_eur DOUBLE         -- model's EUR price forecast
    model_run_id  VARCHAR        -- MLflow run_id for full lineage
    created_at    TIMESTAMP      -- insertion timestamp (default: now)
"""

import duckdb
import pandas as pd


CREATE_PREDICTIONS_TABLE = """
CREATE TABLE IF NOT EXISTS gold_predictions (
    uuid          VARCHAR,
    snapshot_date DATE,
    predicted_eur DOUBLE,
    model_run_id  VARCHAR,
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""


def ensure_predictions_table(conn: duckdb.DuckDBPyConnection) -> None:
    """Create ``gold_predictions`` if it does not already exist.

    Idempotent — safe to call on every run without side effects.

    Args:
        conn: Open DuckDB connection with write access.
    """
    conn.execute(CREATE_PREDICTIONS_TABLE)


def save_predictions(
    conn: duckdb.DuckDBPyConnection,
    predictions_df: pd.DataFrame,
    model_run_id: str,
    snapshot_date: str,
) -> None:
    """Persist model predictions to ``gold_predictions`` for future MAPE auditing.

    Idempotent — calling this function multiple times for the same
    (snapshot_date, model_run_id) pair replaces the existing predictions.

    Args:
        conn:            Open DuckDB connection with write access.
        predictions_df:  DataFrame with at least ``uuid`` and ``predicted_eur``
                         columns.  Extra columns are ignored.
        model_run_id:    MLflow run_id of the model that produced the predictions.
                         Stored for lineage — allows tracing any later alert back
                         to the exact model version.
        snapshot_date:   ISO date string of the price snapshot (``'YYYY-MM-DD'``).
    """
    ensure_predictions_table(conn)
    df = predictions_df[["uuid"]].copy()
    df["snapshot_date"] = snapshot_date
    df["predicted_eur"] = predictions_df["predicted_eur"].values
    df["model_run_id"] = model_run_id
    # Remove existing rows for this snapshot+model to make the call idempotent.
    conn.execute(
        "DELETE FROM gold_predictions WHERE snapshot_date = ? AND model_run_id = ?",
        [snapshot_date, model_run_id],
    )
    conn.execute(
        "INSERT INTO gold_predictions (uuid, snapshot_date, predicted_eur, model_run_id)"
        " SELECT * FROM df"
    )


def compute_rolling_mape(
    conn: duckdb.DuckDBPyConnection,
    window_days: int = 7,
) -> pd.DataFrame:
    """Compute mean absolute percentage error per day over the last N days.

    Joins ``gold_predictions`` with ``gold_price_features`` on uuid and
    ``snapshot_date + 7 days`` to compare predicted vs actual prices.
    Only rows where a matching actual price exists are included.

    Args:
        conn:        Open DuckDB connection with both tables in scope.
        window_days: Number of past days to include. Defaults to 7.

    Returns:
        DataFrame with columns ``snapshot_date`` (DATE) and ``mape`` (DOUBLE,
        percent).  Sorted by snapshot_date ascending.  Empty if no matched rows
        exist within the window.
    """
    sql = """
        SELECT
            p.snapshot_date,
            AVG(ABS(p.predicted_eur - f.eur) / GREATEST(f.eur, 0.01)) * 100 AS mape
        FROM gold_predictions p
        JOIN gold_price_features f
            ON  p.uuid = f.uuid
            AND f.snapshot_date = p.snapshot_date + INTERVAL (7) DAYS
        WHERE p.snapshot_date >= CURRENT_DATE - INTERVAL (?) DAYS
        GROUP BY p.snapshot_date
        ORDER BY p.snapshot_date
    """
    return conn.execute(sql, [window_days]).df()


def is_mape_alert(
    mape_df: pd.DataFrame,
    threshold: float = 30.0,
    consecutive_days: int = 3,
) -> bool:
    """Return ``True`` if MAPE has exceeded ``threshold`` for N consecutive days.

    Looks at the most recent ``consecutive_days`` rows in ``mape_df``.  All
    rows must exceed the threshold; a single day below it resets the streak.

    Args:
        mape_df:          DataFrame returned by :func:`compute_rolling_mape`,
                          with a ``mape`` column.
        threshold:        MAPE percentage above which the model is considered
                          degraded. Default 30.0 (30%).
        consecutive_days: Number of consecutive days above threshold required
                          to trigger an alert. Default 3.

    Returns:
        ``True`` when the alert condition is met, ``False`` otherwise (including
        when fewer than ``consecutive_days`` rows are available).
    """
    if len(mape_df) < consecutive_days:
        return False
    recent = mape_df["mape"].tail(consecutive_days)
    return bool((recent > threshold).all())
