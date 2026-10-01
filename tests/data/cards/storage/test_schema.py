"""Tests for src/data/cards/storage/schema.py."""

import duckdb
import pandas as pd
import pytest

from src.data.cards.storage.schema import find_mistyped_date_columns


@pytest.fixture
def con():
    c = duckdb.connect(":memory:")
    yield c
    c.close()


@pytest.mark.legacy_date_schema  # builds wrong types on purpose
def test_reports_varchar_and_timestamp_but_not_date(con):
    con.execute("CREATE TABLE a (id VARCHAR, snapshot_date VARCHAR)")
    con.execute("CREATE TABLE b (id VARCHAR, tournament_date TIMESTAMP)")
    con.execute("CREATE TABLE ok (id VARCHAR, snapshot_date DATE, other_date VARCHAR)")

    assert find_mistyped_date_columns(con) == [
        ("a", "snapshot_date", "VARCHAR"),
        ("b", "tournament_date", "TIMESTAMP"),
    ]


def test_ignores_views_and_registered_frames(con):
    con.execute("CREATE VIEW v AS SELECT '2026-01-01' AS snapshot_date")
    con.register("frame", pd.DataFrame({"snapshot_date": ["2026-01-01"]}))

    assert find_mistyped_date_columns(con) == []


@pytest.mark.legacy_date_schema  # builds wrong types on purpose
def test_attached_databases_only_when_asked(con):
    con.execute("ATTACH ':memory:' AS other")
    con.execute("CREATE TABLE other.t (snapshot_date VARCHAR)")

    assert find_mistyped_date_columns(con) == []
    assert find_mistyped_date_columns(con, include_attached=True) == [
        ("other.main.t", "snapshot_date", "VARCHAR")
    ]
