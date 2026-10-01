"""Tests for the DATE schema guard in tests/conftest.py and its helper.

The guard runs inside every test, so it is exercised here through pytester:
a throwaway test session that loads the real conftest and reports outcomes.
"""

from pathlib import Path

import duckdb
import pandas as pd
import pytest

from tests.duckdb_helpers import create_table_from_df

pytest_plugins = ["pytester"]

CONFTEST = (Path(__file__).parent / "conftest.py").read_text(encoding="utf-8")


def _run(pytester: pytest.Pytester, body: str) -> pytest.RunResult:
    pytester.makeini(
        "[pytest]\nmarkers =\n    legacy_date_schema: legacy VARCHAR dates\n"
    )
    pytester.makeconftest(CONFTEST)
    pytester.makepyfile(test_inner="import duckdb\nimport pytest\n\n" + body)
    return pytester.runpytest_inprocess("-p", "no:cacheprovider")


@pytest.mark.legacy_date_schema  # builds wrong types on purpose
def test_guard_fails_a_test_that_leaves_a_varchar_date(pytester):
    result = _run(
        pytester,
        "def test_x():\n"
        "    con = duckdb.connect()\n"
        "    con.execute(\"CREATE TABLE t AS SELECT '2026-01-01' AS snapshot_date\")\n",
    )
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*t.snapshot_date is VARCHAR*"])


def test_guard_passes_date_columns(pytester):
    result = _run(
        pytester,
        "def test_x():\n"
        "    con = duckdb.connect()\n"
        "    con.execute(\"CREATE TABLE t AS SELECT DATE '2026-01-01' AS snapshot_date\")\n",
    )
    result.assert_outcomes(passed=1)


@pytest.mark.legacy_date_schema  # builds wrong types on purpose
def test_guard_inspects_file_databases_the_test_already_closed(pytester):
    result = _run(
        pytester,
        "def test_x(tmp_path):\n"
        "    con = duckdb.connect(str(tmp_path / 'silver.duckdb'))\n"
        "    con.execute('CREATE TABLE s (tournament_date VARCHAR)')\n"
        "    con.close()\n",
    )
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*silver.duckdb:s.tournament_date is VARCHAR*"])


@pytest.mark.legacy_date_schema  # builds wrong types on purpose
def test_legacy_marker_opts_out(pytester):
    result = _run(
        pytester,
        "@pytest.mark.legacy_date_schema\n"
        "def test_x():\n"
        "    con = duckdb.connect()\n"
        "    con.execute(\"CREATE TABLE t AS SELECT '2026-01-01' AS snapshot_date\")\n",
    )
    result.assert_outcomes(passed=1)


def test_create_table_from_df_types_dates_for_strings_and_empty_frames():
    con = duckdb.connect()
    create_table_from_df(
        con, "prices", pd.DataFrame({"id": ["a"], "snapshot_date": ["2026-05-01"]})
    )
    create_table_from_df(con, "empty", pd.DataFrame(columns=["id", "tournament_date"]))

    types = dict(
        con.execute(
            "SELECT table_name || '.' || column_name, data_type FROM duckdb_columns() "
            "WHERE column_name IN ('snapshot_date', 'tournament_date')"
        ).fetchall()
    )
    assert types == {"prices.snapshot_date": "DATE", "empty.tournament_date": "DATE"}
    con.close()
