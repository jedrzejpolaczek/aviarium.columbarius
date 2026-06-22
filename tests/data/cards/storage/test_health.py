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
