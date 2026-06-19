"""Unit tests for src/monitoring/mape_tracker.py."""

from datetime import date, timedelta

import duckdb
import pandas as pd
import pytest

from src.monitoring.mape_tracker import (
    compute_rolling_mape,
    ensure_predictions_table,
    is_mape_alert,
    save_predictions,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def conn():
    """In-memory DuckDB with gold_price_features populated for MAPE tests.

    gold_predictions is intentionally absent — ensure_predictions_table and
    save_predictions are responsible for creating and populating it.

    Layout:
        card_a: 15 daily snapshots from 2026-01-01.
                eur grows from 1.00 to 2.40 in 0.10 increments.
        card_b: same dates, eur constant at 5.0.
    """
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
    for i in range(15):
        d = date(2026, 1, 1) + timedelta(days=i)
        con.execute(
            "INSERT INTO gold_price_features VALUES (?, ?, ?, ?, ?)",
            ["card_a", str(d), round(1.0 + i * 0.1, 2), None, None],
        )
        con.execute(
            "INSERT INTO gold_price_features VALUES (?, ?, ?, ?, ?)",
            ["card_b", str(d), 5.0, None, None],
        )
    yield con
    con.close()


@pytest.fixture
def conn_with_predictions(conn):
    """Fixture with gold_predictions pre-populated.

    Predictions were made on 2026-01-01 for card_a and card_b.
    Actuals land on 2026-01-08 (snapshot_date + 7 days).

    card_a: predicted 1.50, actual 1.70 → |error| / actual = 0.118 → MAPE ≈ 11.8%
    card_b: predicted 5.50, actual 5.00 → |error| / actual = 0.100 → MAPE = 10.0%
    Combined MAPE for 2026-01-01 ≈ (11.8 + 10.0) / 2 ≈ 10.9%
    """
    ensure_predictions_table(conn)
    df = pd.DataFrame(
        {
            "uuid": ["card_a", "card_b"],
            "predicted_eur": [1.50, 5.50],
        }
    )
    save_predictions(conn, df, "run_abc", "2026-01-01")
    return conn


# ---------------------------------------------------------------------------
# ensure_predictions_table
# ---------------------------------------------------------------------------


def test_ensure_creates_table(conn):
    ensure_predictions_table(conn)
    tables = conn.execute("SHOW TABLES").df()["name"].tolist()
    assert "gold_predictions" in tables


def test_ensure_is_idempotent(conn):
    ensure_predictions_table(conn)
    ensure_predictions_table(conn)  # should not raise
    tables = conn.execute("SHOW TABLES").df()["name"].tolist()
    assert tables.count("gold_predictions") == 1


def test_ensure_table_has_uuid_column(conn):
    ensure_predictions_table(conn)
    cols = conn.execute("DESCRIBE gold_predictions").df()["column_name"].tolist()
    assert "uuid" in cols


def test_ensure_table_has_model_run_id_column(conn):
    ensure_predictions_table(conn)
    cols = conn.execute("DESCRIBE gold_predictions").df()["column_name"].tolist()
    assert "model_run_id" in cols


# ---------------------------------------------------------------------------
# save_predictions
# ---------------------------------------------------------------------------


def test_save_predictions_inserts_rows(conn):
    df = pd.DataFrame({"uuid": ["card_a", "card_b"], "predicted_eur": [1.5, 5.5]})
    save_predictions(conn, df, "run_1", "2026-01-01")
    count = conn.execute("SELECT COUNT(*) FROM gold_predictions").fetchone()[0]
    assert count == 2


def test_save_predictions_sets_model_run_id(conn):
    df = pd.DataFrame({"uuid": ["card_a"], "predicted_eur": [1.5]})
    save_predictions(conn, df, "my_run_id", "2026-01-01")
    row = conn.execute("SELECT model_run_id FROM gold_predictions").fetchone()
    assert row[0] == "my_run_id"


def test_save_predictions_sets_snapshot_date(conn):
    df = pd.DataFrame({"uuid": ["card_a"], "predicted_eur": [1.5]})
    save_predictions(conn, df, "run_1", "2026-03-15")
    row = conn.execute("SELECT snapshot_date FROM gold_predictions").fetchone()
    assert str(row[0]) == "2026-03-15"


def test_save_predictions_ignores_extra_columns(conn):
    df = pd.DataFrame(
        {"uuid": ["card_a"], "predicted_eur": [1.5], "extra_col": ["ignore"]}
    )
    save_predictions(conn, df, "run_1", "2026-01-01")  # should not raise
    count = conn.execute("SELECT COUNT(*) FROM gold_predictions").fetchone()[0]
    assert count == 1


def test_save_predictions_is_idempotent(conn):
    """Calling save_predictions twice with the same args must not double rows."""
    df = pd.DataFrame({"uuid": ["card_a", "card_b"], "predicted_eur": [1.5, 5.5]})
    save_predictions(conn, df, "run_1", "2026-01-01")
    save_predictions(conn, df, "run_1", "2026-01-01")  # second call — same args
    count = conn.execute("SELECT COUNT(*) FROM gold_predictions").fetchone()[0]
    assert count == 2  # still only the original 2 rows, not 4


def test_save_predictions_idempotent_different_run_id_keeps_both(conn):
    """Different model_run_id for the same snapshot_date coexist as separate rows."""
    df = pd.DataFrame({"uuid": ["card_a"], "predicted_eur": [1.5]})
    save_predictions(conn, df, "run_1", "2026-01-01")
    save_predictions(conn, df, "run_2", "2026-01-01")
    count = conn.execute("SELECT COUNT(*) FROM gold_predictions").fetchone()[0]
    assert count == 2


# ---------------------------------------------------------------------------
# compute_rolling_mape
# ---------------------------------------------------------------------------


def test_compute_rolling_mape_returns_dataframe(conn_with_predictions):
    result = compute_rolling_mape(conn_with_predictions)
    assert isinstance(result, pd.DataFrame)


def test_compute_rolling_mape_has_snapshot_date_column(conn_with_predictions):
    result = compute_rolling_mape(conn_with_predictions)
    assert "snapshot_date" in result.columns


def test_compute_rolling_mape_has_mape_column(conn_with_predictions):
    result = compute_rolling_mape(conn_with_predictions)
    assert "mape" in result.columns


def test_compute_rolling_mape_returns_non_negative_values(conn_with_predictions):
    result = compute_rolling_mape(conn_with_predictions)
    if not result.empty:
        assert (result["mape"] >= 0).all()


def test_compute_rolling_mape_empty_when_no_predictions(conn):
    # No predictions table → ensure_predictions_table returns empty result
    ensure_predictions_table(conn)
    result = compute_rolling_mape(conn)
    assert result.empty


def test_compute_rolling_mape_mape_positive_when_prediction_wrong(
    conn_with_predictions,
):
    # window_days=365 ensures fixture dates (2026-01-01) fall within the query window
    result = compute_rolling_mape(conn_with_predictions, window_days=365)
    assert not result.empty
    assert result["mape"].iloc[0] > 0


# ---------------------------------------------------------------------------
# is_mape_alert
# ---------------------------------------------------------------------------


def _mape_df(*values: float) -> pd.DataFrame:
    return pd.DataFrame({"mape": list(values)})


def test_is_mape_alert_returns_true_when_all_above_threshold():
    df = _mape_df(35.0, 38.0, 40.0)
    assert is_mape_alert(df, threshold=30.0, consecutive_days=3) is True


def test_is_mape_alert_returns_false_when_one_below_threshold():
    df = _mape_df(35.0, 25.0, 40.0)
    assert is_mape_alert(df, threshold=30.0, consecutive_days=3) is False


def test_is_mape_alert_returns_false_when_insufficient_rows():
    df = _mape_df(40.0, 50.0)
    assert is_mape_alert(df, threshold=30.0, consecutive_days=3) is False


def test_is_mape_alert_returns_false_on_empty_df():
    assert is_mape_alert(pd.DataFrame({"mape": []})) is False


def test_is_mape_alert_checks_last_n_rows_only():
    # First rows below threshold, last 3 above → alert
    df = _mape_df(10.0, 10.0, 40.0, 40.0, 40.0)
    assert is_mape_alert(df, threshold=30.0, consecutive_days=3) is True


def test_is_mape_alert_no_alert_when_only_last_day_high():
    df = _mape_df(10.0, 10.0, 40.0)
    assert is_mape_alert(df, threshold=30.0, consecutive_days=3) is False


def test_is_mape_alert_respects_custom_threshold():
    df = _mape_df(20.0, 20.0, 20.0)
    assert is_mape_alert(df, threshold=15.0, consecutive_days=3) is True
    assert is_mape_alert(df, threshold=25.0, consecutive_days=3) is False


def test_is_mape_alert_respects_custom_consecutive_days():
    df = _mape_df(40.0, 40.0)
    assert is_mape_alert(df, threshold=30.0, consecutive_days=2) is True
    assert is_mape_alert(df, threshold=30.0, consecutive_days=3) is False


def test_is_mape_alert_returns_bool():
    df = _mape_df(40.0, 40.0, 40.0)
    result = is_mape_alert(df)
    assert isinstance(result, bool)
