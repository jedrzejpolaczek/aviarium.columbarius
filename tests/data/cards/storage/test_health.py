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
