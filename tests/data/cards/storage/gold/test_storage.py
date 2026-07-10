import duckdb

from src.data.cards.storage.gold.storage import get_latest_trainable_snapshot_date


def _conn_with_snapshots(dates: list[str]) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")
    rows = ", ".join(f"('uuid-1', DATE '{d}', 1.0)" for d in dates)
    con.execute(
        f"CREATE TABLE gold_price_features AS "
        f"SELECT * FROM (VALUES {rows}) AS t(uuid, snapshot_date, eur)"
    )
    return con


def test_returns_none_when_table_missing():
    con = duckdb.connect(":memory:")
    assert get_latest_trainable_snapshot_date(con) is None


def test_returns_none_when_no_snapshot_has_a_t_plus_7_counterpart():
    # Only 5 days of history — nothing is 7 days apart yet.
    con = _conn_with_snapshots(
        ["2026-07-05", "2026-07-06", "2026-07-07", "2026-07-08", "2026-07-09"]
    )
    assert get_latest_trainable_snapshot_date(con) is None


def test_skips_latest_snapshot_when_it_has_no_t_plus_7_counterpart():
    # 2026-07-09 is the latest snapshot, but nothing exists at 2026-07-16 yet.
    # 2026-07-02 -> 2026-07-09 is the newest pair that is actually trainable.
    con = _conn_with_snapshots(["2026-06-25", "2026-07-02", "2026-07-09"])
    assert get_latest_trainable_snapshot_date(con) == "2026-07-02"


def test_ignores_pairs_that_are_not_exactly_7_days_apart():
    # 2026-06-23 -> 2026-06-30 is missing, so 2026-06-23 is not trainable,
    # but 2026-06-16 -> 2026-06-23 is, and should still be picked up.
    con = _conn_with_snapshots(["2026-06-16", "2026-06-23", "2026-06-29"])
    assert get_latest_trainable_snapshot_date(con) == "2026-06-16"
