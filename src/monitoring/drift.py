"""Detects a shift in how card prices *move*, as an early warning (ADR-034).

What is compared, and why not prices:
    The previous version compared the distribution of price *levels* across
    the whole catalogue. That distribution is dominated by ~60 000 sub-€1
    cards and barely moves in months: on real history the drift score stayed
    at 0.001–0.010 against a 0.1 threshold even across the July 2026
    price-feed switch, the biggest data shock the project has had. A ban moves
    a handful of cards out of ~80 000 and is invisible to it.

    This version compares the distribution of 7-day log returns — the quantity
    the model predicts. Measured on the same history, normal weeks score
    0.02–0.03 and the feed-switch week scores 0.34–0.42.

Statistic:
    Wasserstein distance between the reference and current return
    distributions, divided by the reference standard deviation. It is the
    statistic Evidently used for large samples, computed directly with numpy so
    the check no longer depends on Evidently's API.

    Default windows: current = returns *starting* in the last 7 days whose
    t+7 end has been collected; reference = the 28 days before that.

Drift only raises an alert. It is not a retrain trigger: a changed market
does not make the model wrong by itself, and the skill check in
``prediction_tracker`` measures whether it did.
"""

from dataclasses import dataclass
from datetime import date, timedelta

import duckdb
import numpy as np
import pandas as pd

DRIFT_THRESHOLD = 0.15
"""Calibrated on 2026-06..09 history, scoring every snapshot date.

Normal September weeks scored 0.027-0.076, with a single 0.103 on 2026-09-28.
The July price-feed switch scored 1.59, 0.80 and 0.69 on the first days it
entered the current window, and stayed above 0.16 for the next week.
Evidently's default of 0.1 would have fired on that September outlier."""
CURRENT_DAYS = 7
REFERENCE_DAYS = 28
HORIZON_DAYS = 7


@dataclass(frozen=True)
class DriftResult:
    score: float
    drifted: bool
    reference_n: int
    current_n: int
    current_start: str
    current_end: str


def fetch_returns(conn: duckdb.DuckDBPyConnection, start: date, end: date) -> pd.Series:
    """7-day log returns ``ln(1+eur_t+7) - ln(1+eur_t)`` for start dates in [start, end]."""
    return (
        conn.execute(
            """
            SELECT LN(1 + f.eur) - LN(1 + p.eur) AS r
            FROM gold_price_features p
            JOIN gold_price_features f
              ON p.uuid = f.uuid
             AND f.snapshot_date = p.snapshot_date + CAST(? AS INTEGER)
            WHERE p.snapshot_date BETWEEN ? AND ?
              AND p.eur IS NOT NULL AND f.eur IS NOT NULL
            """,
            [HORIZON_DAYS, start, end],
        )
        .df()["r"]
        .astype(float)
    )


def wasserstein_1d(u: np.ndarray, v: np.ndarray) -> float:
    """First Wasserstein distance between two 1-D samples (area between their CDFs).

    Same algorithm as ``scipy.stats.wasserstein_distance`` for unweighted
    samples; written out so the check needs neither scipy nor Evidently.
    """
    u = np.sort(u)
    v = np.sort(v)
    grid = np.sort(np.concatenate([u, v]))
    deltas = np.diff(grid)
    u_cdf = np.searchsorted(u, grid[:-1], side="right") / len(u)
    v_cdf = np.searchsorted(v, grid[:-1], side="right") / len(v)
    return float(np.sum(np.abs(u_cdf - v_cdf) * deltas))


def drift_score(reference: pd.Series, current: pd.Series) -> float:
    """Wasserstein distance normalised by the reference spread; NaN if undefined."""
    if reference.empty or current.empty:
        return float("nan")
    spread = float(np.std(reference))
    if spread == 0.0:
        return float("nan")
    return wasserstein_1d(reference.to_numpy(), current.to_numpy()) / spread


def return_drift(
    conn: duckdb.DuckDBPyConnection,
    latest_snapshot: date,
    threshold: float = DRIFT_THRESHOLD,
) -> DriftResult:
    """Compare the latest week of matured 7-day returns with the four weeks before it.

    Args:
        conn:            Connection with ``gold_price_features`` in scope.
        latest_snapshot: Newest snapshot in Gold. Returns ending after it do
                         not exist yet, so the current window ends 7 days
                         earlier.
    """
    current_end = latest_snapshot - timedelta(days=HORIZON_DAYS)
    current_start = current_end - timedelta(days=CURRENT_DAYS - 1)
    reference_end = current_start - timedelta(days=1)
    reference_start = reference_end - timedelta(days=REFERENCE_DAYS - 1)

    reference = fetch_returns(conn, reference_start, reference_end)
    current = fetch_returns(conn, current_start, current_end)
    score = drift_score(reference, current)
    return DriftResult(
        score=score,
        drifted=bool(np.isfinite(score) and score > threshold),
        reference_n=len(reference),
        current_n=len(current),
        current_start=str(current_start),
        current_end=str(current_end),
    )
