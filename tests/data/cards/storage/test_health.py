import datetime

import duckdb
import pytest

from src.data.cards.storage.health import CheckResult


def test_check_result_pass():
    r = CheckResult(name="silver_cards rows", layer="silver", status="PASS", detail="515728 rows")
    assert r.name == "silver_cards rows"
    assert r.layer == "silver"
    assert r.status == "PASS"
    assert r.detail == "515728 rows"


def test_check_result_fail():
    r = CheckResult(name="silver_cards rows", layer="gold", status="FAIL", detail="0 rows")
    assert r.status == "FAIL"


from src.data.cards.storage.health import _check_table_has_rows


class TestCheckTableHasRows:
    def test_pass_when_table_has_rows(self):
        con = duckdb.connect(":memory:")
        con.execute("CREATE TABLE silver_cards (uuid VARCHAR)")
        con.execute("INSERT INTO silver_cards VALUES ('abc')")
        result = _check_table_has_rows(con, "silver", "silver_cards")
        assert result.status == "PASS"
        assert "1" in result.detail
        con.close()

    def test_fail_when_table_missing(self):
        con = duckdb.connect(":memory:")
        result = _check_table_has_rows(con, "silver", "silver_cards")
        assert result.status == "FAIL"
        assert "not found" in result.detail
        con.close()

    def test_fail_when_table_empty(self):
        con = duckdb.connect(":memory:")
        con.execute("CREATE TABLE silver_cards (uuid VARCHAR)")
        result = _check_table_has_rows(con, "silver", "silver_cards")
        assert result.status == "FAIL"
        assert "0 rows" in result.detail
        con.close()


from src.data.cards.storage.health import _check_snapshot_date_today


class TestCheckSnapshotDateToday:
    def _make_prices(self, con: duckdb.DuckDBPyConnection, dates: list[datetime.date]) -> None:
        con.execute(
            "CREATE TABLE silver_prices_history (uuid VARCHAR, snapshot_date DATE)"
        )
        for d in dates:
            con.execute("INSERT INTO silver_prices_history VALUES ('x', ?)", [d])

    def test_pass_when_today_present(self):
        con = duckdb.connect(":memory:")
        today = datetime.date(2026, 6, 22)
        self._make_prices(con, [today])
        result = _check_snapshot_date_today(con, "silver_prices_history", today)
        assert result.status == "PASS"
        assert "2026-06-22" in result.detail
        con.close()

    def test_fail_when_only_yesterday(self):
        con = duckdb.connect(":memory:")
        today = datetime.date(2026, 6, 22)
        yesterday = datetime.date(2026, 6, 21)
        self._make_prices(con, [yesterday])
        result = _check_snapshot_date_today(con, "silver_prices_history", today)
        assert result.status == "FAIL"
        assert "no rows" in result.detail
        con.close()

    def test_fail_when_table_empty(self):
        con = duckdb.connect(":memory:")
        today = datetime.date(2026, 6, 22)
        self._make_prices(con, [])
        result = _check_snapshot_date_today(con, "silver_prices_history", today)
        assert result.status == "FAIL"
        con.close()
