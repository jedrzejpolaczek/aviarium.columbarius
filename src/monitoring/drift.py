"""Detects distribution drift in card prices using Evidently.

Why drift monitoring:
    Format bans (e.g. a card banned in Modern) can shift the EUR price
    distribution within 24 hours — cards drop 50–90%.  A model trained on
    pre-ban data has no knowledge of this regime change.  Drift detection
    is the safety net that flags "the world changed, retrain before the
    MAPE alarm fires three days later."

Comparison window:
    ``reference`` = previous 30 days (stable baseline distribution)
    ``current``   = last 7 days (what the model is currently seeing)

    Evidently computes per-column statistical tests (KS test for continuous
    features, Jensen-Shannon divergence) and reports an overall ``dataset_drift``
    boolean when enough columns drift simultaneously.

Dependency:
    Requires ``evidently>=0.4.0`` (``uv add evidently``).
    Only imported inside :func:`compute_drift_report` so the rest of the module
    loads without evidently installed (useful for non-drift monitoring paths).
"""

from typing import Any

import duckdb
import pandas as pd


def fetch_prices_for_period(
    conn: duckdb.DuckDBPyConnection,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Return EUR prices and their log transform for a date range.

    ``log_eur`` is computed in SQL as ``LN(1 + eur)`` to avoid a round-trip
    through Python and to keep NULLs intact (NULL eur → NULL log_eur).

    Args:
        conn:       Open DuckDB connection with ``gold_price_features`` in scope.
        start_date: Start of the period (inclusive), ISO format ``'YYYY-MM-DD'``.
        end_date:   End of the period (inclusive), ISO format ``'YYYY-MM-DD'``.

    Returns:
        DataFrame with columns ``uuid`` (VARCHAR), ``eur`` (DOUBLE),
        ``log_eur`` (DOUBLE), ``snapshot_date`` (DATE).
        May be empty if no data exists in the given range.
    """
    return conn.execute(
        """
        SELECT
            uuid,
            eur,
            CASE WHEN eur IS NOT NULL THEN LN(1 + eur) END AS log_eur,
            snapshot_date
        FROM gold_price_features
        WHERE snapshot_date BETWEEN ? AND ?
        """,
        [start_date, end_date],
    ).df()


def compute_drift_report(
    reference: pd.DataFrame, current: pd.DataFrame
) -> dict[str, Any]:
    """Build an Evidently data-drift report and return the result as a dict.

    Compares the ``eur`` and ``log_eur`` column distributions between the
    reference period (30 days) and the current period (7 days).  Evidently
    runs per-column statistical tests and produces an overall drift verdict.

    Args:
        reference: Historical price DataFrame (``eur``, ``log_eur`` columns
                   required).  Typically covers the previous 30 days.
        current:   Recent price DataFrame (same schema).  Typically covers
                   the last 7 days.

    Returns:
        Evidently ``report.as_dict()`` — a nested dict whose first element
        is the ``DatasetDriftMetric`` result.  Pass this directly to
        :func:`is_drift_detected`.

    Raises:
        ImportError: ``evidently`` is not installed.
    """
    from evidently.metric_preset import DataDriftPreset
    from evidently.report import Report

    report = Report(metrics=[DataDriftPreset()])
    report.run(
        reference_data=reference[["eur", "log_eur"]].dropna(),
        current_data=current[["eur", "log_eur"]].dropna(),
    )
    result: dict[str, Any] = report.as_dict()
    return result


def is_drift_detected(drift_report: dict[str, Any]) -> bool:
    """Extract the overall drift verdict from an Evidently report dict.

    Reads the ``dataset_drift`` boolean from the first metric entry, which
    is always ``DatasetDriftMetric`` when the report was built with
    :func:`compute_drift_report`.

    Args:
        drift_report: Dict returned by :func:`compute_drift_report` (i.e.
                      ``report.as_dict()`` from Evidently).

    Returns:
        ``True`` when Evidently detected statistically significant drift in
        the price distribution, ``False`` otherwise.
    """
    return bool(drift_report["metrics"][0]["result"]["dataset_drift"])
