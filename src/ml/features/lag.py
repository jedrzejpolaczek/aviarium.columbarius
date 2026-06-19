"""
Builds time-based lag features from DuckDB price history.

All features are computed BEFORE the train/val/test split to avoid data
leakage — the window functions look only backwards, never at future rows.

WHY LAG FEATURES:
The model predicts log_return_7d = log1p(price_t+7) - log1p(price_today).
To make that prediction it needs to know:
- what the price was doing over recent days (lag_1d, lag_7d, lag_14d, lag_30d)
- how stable it has been (rolling_std_14d)
- whether it is trending up or down (momentum_7d)

DATA SOURCE:
Table gold_price_features in DuckDB — the same table used in
notebooks/model_preparation/04_baseline_models.ipynb.

WINDOW FUNCTIONS:
LAG(eur, N) OVER (PARTITION BY uuid ORDER BY snapshot_date) looks back N
rows within each card's price history. All window calculations are bounded
to rows preceding the current one, so no future data is visible.
"""

import duckdb
import pandas as pd


def build_lag_features(
    conn: duckdb.DuckDBPyConnection,
    snapshot_date: str,
) -> pd.DataFrame:
    """Return a DataFrame of time-based features for every card on a given date.

    All window functions look only at rows on or before snapshot_date, so
    this function is safe to call before the train/val split.

    Args:
        conn:          Open DuckDB connection.
        snapshot_date: Date string in 'YYYY-MM-DD' format, e.g. '2026-06-09'.

    Returns:
        DataFrame with one row per card present on snapshot_date and columns:
            uuid, snapshot_date, eur,
            edhrec_rank                         — daily EDHREC rank (NaN if no MTGJson data),
            foil_premium                        — eur_foil / eur ratio (NaN if no foil price),
            lag_1d, lag_7d, lag_14d, lag_30d   — price N days ago (NaN if history too short),
            rolling_mean_7d                     — 7-day trailing average price,
            rolling_std_14d                     — 14-day trailing price std deviation,
            rolling_min_30d, rolling_max_30d    — 30-day trailing min/max,
            momentum_7d                         — (eur - lag_7d) / lag_7d, NaN when lag_7d is 0.
    """
    sql = """
        WITH lagged AS (
            SELECT
                uuid,
                snapshot_date,
                eur,
                edhrec_rank,
                foil_premium,
                LAG(eur,  1) OVER w AS lag_1d,
                LAG(eur,  7) OVER w AS lag_7d,
                LAG(eur, 14) OVER w AS lag_14d,
                LAG(eur, 30) OVER w AS lag_30d,
                AVG(eur) OVER (
                    PARTITION BY uuid ORDER BY snapshot_date
                    ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
                ) AS rolling_mean_7d,
                STDDEV(eur) OVER (
                    PARTITION BY uuid ORDER BY snapshot_date
                    ROWS BETWEEN 13 PRECEDING AND CURRENT ROW
                ) AS rolling_std_14d,
                MIN(eur) OVER (
                    PARTITION BY uuid ORDER BY snapshot_date
                    ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
                ) AS rolling_min_30d,
                MAX(eur) OVER (
                    PARTITION BY uuid ORDER BY snapshot_date
                    ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
                ) AS rolling_max_30d
            FROM gold_price_features
            WINDOW w AS (PARTITION BY uuid ORDER BY snapshot_date)
        )
        SELECT
            *,
            (eur - lag_7d) / NULLIF(lag_7d, 0) AS momentum_7d
        FROM lagged
        WHERE snapshot_date = ?
    """
    return conn.execute(sql, [snapshot_date]).df()


def build_target(
    conn: duckdb.DuckDBPyConnection,
    snapshot_date: str,
) -> pd.DataFrame:
    """Compute the model target: log_return_7d = log1p(eur_t+7) - log1p(eur_t).

    Only cards with a price snapshot on both snapshot_date and snapshot_date + 7
    days are returned. Cards missing either date are silently excluded.

    Args:
        conn:          Open DuckDB connection.
        snapshot_date: Base date in 'YYYY-MM-DD' format. The target looks 7 days ahead.

    Returns:
        DataFrame with columns: uuid, log_return_7d.
    """
    sql = """
        WITH t0 AS (
            SELECT uuid, eur AS eur_t0
            FROM gold_price_features
            WHERE snapshot_date = ?
        ),
        t7 AS (
            SELECT uuid, eur AS eur_t7
            FROM gold_price_features
            WHERE snapshot_date = CAST(? AS DATE) + INTERVAL 7 DAY
        )
        SELECT
            t0.uuid,
            LN(1 + t7.eur_t7) - LN(1 + t0.eur_t0) AS log_return_7d
        FROM t0
        JOIN t7 ON t0.uuid = t7.uuid
    """
    return conn.execute(sql, [snapshot_date, snapshot_date]).df()
