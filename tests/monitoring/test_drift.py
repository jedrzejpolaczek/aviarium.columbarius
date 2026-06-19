"""Unit tests for src/monitoring/drift.py."""

from datetime import date, timedelta

import duckdb
import pandas as pd
import pytest

from src.monitoring.drift import fetch_prices_for_period, is_drift_detected


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def conn():
    """In-memory DuckDB with 40 days of price data for two cards."""
    con = duckdb.connect()
    con.execute("""
        CREATE TABLE gold_price_features (
            uuid          VARCHAR,
            snapshot_date DATE,
            eur           DOUBLE,
            edhrec_rank   DOUBLE,
            foil_premium  DOUBLE
        )
    """)
    for i in range(40):
        d = date(2026, 1, 1) + timedelta(days=i)
        eur_a = round(1.0 + i * 0.05, 2)
        con.execute(
            "INSERT INTO gold_price_features VALUES (?, ?, ?, ?, ?)",
            ["card_a", str(d), eur_a, None, None],
        )
        con.execute(
            "INSERT INTO gold_price_features VALUES (?, ?, ?, ?, ?)",
            ["card_b", str(d), 5.0, None, None],
        )
    yield con
    con.close()


# ---------------------------------------------------------------------------
# fetch_prices_for_period
# ---------------------------------------------------------------------------


def test_fetch_prices_returns_dataframe(conn):
    result = fetch_prices_for_period(conn, "2026-01-01", "2026-01-10")
    assert isinstance(result, pd.DataFrame)


def test_fetch_prices_has_eur_column(conn):
    result = fetch_prices_for_period(conn, "2026-01-01", "2026-01-10")
    assert "eur" in result.columns


def test_fetch_prices_has_log_eur_column(conn):
    result = fetch_prices_for_period(conn, "2026-01-01", "2026-01-10")
    assert "log_eur" in result.columns


def test_fetch_prices_has_uuid_column(conn):
    result = fetch_prices_for_period(conn, "2026-01-01", "2026-01-10")
    assert "uuid" in result.columns


def test_fetch_prices_has_snapshot_date_column(conn):
    result = fetch_prices_for_period(conn, "2026-01-01", "2026-01-10")
    assert "snapshot_date" in result.columns


def test_fetch_prices_date_filter_inclusive(conn):
    result = fetch_prices_for_period(conn, "2026-01-05", "2026-01-05")
    dates = result["snapshot_date"].astype(str).unique()
    assert list(dates) == ["2026-01-05"]


def test_fetch_prices_returns_empty_for_future_dates(conn):
    result = fetch_prices_for_period(conn, "2099-01-01", "2099-01-10")
    assert result.empty


def test_fetch_prices_log_eur_is_log1p_of_eur(conn):
    import numpy as np

    result = fetch_prices_for_period(conn, "2026-01-01", "2026-01-01")
    for _, row in result.iterrows():
        if pd.notna(row["eur"]) and pd.notna(row["log_eur"]):
            expected = np.log1p(row["eur"])
            assert abs(row["log_eur"] - expected) < 1e-9


def test_fetch_prices_respects_both_bounds(conn):
    result = fetch_prices_for_period(conn, "2026-01-01", "2026-01-07")
    dates = result["snapshot_date"].astype(str).unique()
    assert all("2026-01-01" <= d <= "2026-01-07" for d in dates)


# ---------------------------------------------------------------------------
# is_drift_detected
# ---------------------------------------------------------------------------


def _make_report(dataset_drift: bool) -> dict:
    return {"metrics": [{"result": {"dataset_drift": dataset_drift}}]}


def test_is_drift_detected_returns_true_when_flagged():
    assert is_drift_detected(_make_report(True)) is True


def test_is_drift_detected_returns_false_when_not_flagged():
    assert is_drift_detected(_make_report(False)) is False


def test_is_drift_detected_returns_bool():
    result = is_drift_detected(_make_report(True))
    assert isinstance(result, bool)


def test_is_drift_detected_with_integer_true():
    report = {"metrics": [{"result": {"dataset_drift": 1}}]}
    assert is_drift_detected(report) is True


def test_is_drift_detected_with_integer_zero():
    report = {"metrics": [{"result": {"dataset_drift": 0}}]}
    assert is_drift_detected(report) is False


# ---------------------------------------------------------------------------
# compute_drift_report (requires evidently — skipped if not installed)
# ---------------------------------------------------------------------------

evidently = pytest.importorskip("evidently", reason="evidently not installed")


def test_compute_drift_report_returns_dict(conn):
    from src.monitoring.drift import compute_drift_report

    reference = fetch_prices_for_period(conn, "2026-01-01", "2026-01-30")
    current = fetch_prices_for_period(conn, "2026-01-31", "2026-02-09")
    result = compute_drift_report(reference, current)
    assert isinstance(result, dict)


def test_compute_drift_report_has_metrics_key(conn):
    from src.monitoring.drift import compute_drift_report

    reference = fetch_prices_for_period(conn, "2026-01-01", "2026-01-30")
    current = fetch_prices_for_period(conn, "2026-01-31", "2026-02-09")
    result = compute_drift_report(reference, current)
    assert "metrics" in result


def test_compute_drift_report_first_metric_has_dataset_drift(conn):
    from src.monitoring.drift import compute_drift_report

    reference = fetch_prices_for_period(conn, "2026-01-01", "2026-01-30")
    current = fetch_prices_for_period(conn, "2026-01-31", "2026-02-09")
    result = compute_drift_report(reference, current)
    assert "dataset_drift" in result["metrics"][0]["result"]


def test_compute_drift_report_dataset_drift_is_bool(conn):
    from src.monitoring.drift import compute_drift_report

    reference = fetch_prices_for_period(conn, "2026-01-01", "2026-01-30")
    current = fetch_prices_for_period(conn, "2026-01-31", "2026-02-09")
    result = compute_drift_report(reference, current)
    assert isinstance(result["metrics"][0]["result"]["dataset_drift"], bool)
