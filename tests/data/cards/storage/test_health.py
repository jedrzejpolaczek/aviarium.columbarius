import datetime
from pathlib import Path

import duckdb
import pytest

from src.data.cards.storage.health import (
    CheckResult,
    _check_table_has_rows,
    _check_snapshot_date_today,
    _check_no_nulls,
    _check_no_duplicate_canonical_uuid,
    _check_oracle_id_conflicts,
    _check_silver_prices_no_negative_eur,
    _check_gold_ml_dataset_has_target,
    _check_bronze_prices_schema_drift,
    run_health_checks,
)


def test_check_result_pass():
    r = CheckResult(
        name="silver_cards rows", layer="silver", status="PASS", detail="515728 rows"
    )
    assert r.name == "silver_cards rows"
    assert r.layer == "silver"
    assert r.status == "PASS"
    assert r.detail == "515728 rows"


def test_check_result_fail():
    r = CheckResult(
        name="silver_cards rows", layer="gold", status="FAIL", detail="0 rows"
    )
    assert r.status == "FAIL"


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


class TestCheckSnapshotDateToday:
    def _make_prices(
        self, con: duckdb.DuckDBPyConnection, dates: list[datetime.date]
    ) -> None:
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


def _make_silver_cards(
    con: duckdb.DuckDBPyConnection,
    rows: list[tuple],
) -> None:
    """rows: (uuid, canonical_uuid, name, set_code, collector_number, oracle_id)"""
    con.execute("""
        CREATE TABLE silver_cards (
            uuid VARCHAR,
            canonical_uuid VARCHAR,
            name VARCHAR,
            set_code VARCHAR,
            collector_number VARCHAR,
            oracle_id VARCHAR
        )
    """)
    for row in rows:
        con.execute("INSERT INTO silver_cards VALUES (?, ?, ?, ?, ?, ?)", list(row))


class TestCheckNoNulls:
    def test_pass_when_no_nulls(self):
        con = duckdb.connect(":memory:")
        _make_silver_cards(con, [("u1", "u1", "Serra Angel", "10E", "1", "o1")])
        result = _check_no_nulls(con, "silver", "silver_cards", "name")
        assert result.status == "PASS"
        assert "no NULLs" in result.detail
        con.close()

    def test_fail_when_null_present(self):
        con = duckdb.connect(":memory:")
        _make_silver_cards(con, [(None, None, None, None, None, None)])
        result = _check_no_nulls(con, "silver", "silver_cards", "name")
        assert result.status == "FAIL"
        assert "1 NULL" in result.detail
        con.close()


class TestCheckNoDuplicateCanonicalUuid:
    def test_pass_when_no_duplicates(self):
        con = duckdb.connect(":memory:")
        _make_silver_cards(
            con,
            [
                ("u1", "u1", "Serra Angel", "10E", "1", "o1"),
                ("u2", "u2", "Shivan Dragon", "10E", "2", "o2"),
            ],
        )
        result = _check_no_duplicate_canonical_uuid(con)
        assert result.status == "PASS"
        con.close()

    def test_fail_when_duplicate_canonical_uuid(self):
        con = duckdb.connect(":memory:")
        _make_silver_cards(
            con,
            [
                ("u1", "u1", "Serra Angel", "10E", "1", "o1"),
                ("u1", "u1", "Serra Angel", "10E", "1a", "o1"),
            ],
        )
        result = _check_no_duplicate_canonical_uuid(con)
        assert result.status == "FAIL"
        assert "1 duplicated" in result.detail
        con.close()


class TestCheckOracleIdConflicts:
    def test_pass_when_conflicts_within_threshold(self):
        # 5 conflicts (split cards like the production run) — well under threshold of 20
        con = duckdb.connect(":memory:")
        rows: list[tuple] = []
        for i in range(5):
            name = f"Split {i}"
            rows.append((f"u{i}a", f"u{i}a", name, "10E", f"{i}a", f"oa{i}"))
            rows.append((f"u{i}b", f"u{i}b", name, "10E", f"{i}b", f"ob{i}"))
        _make_silver_cards(con, rows)
        result = _check_oracle_id_conflicts(con)
        assert result.status == "PASS"
        assert "5 conflicts" in result.detail
        con.close()

    def test_fail_when_conflicts_exceed_threshold(self):
        # 21 names each mapping to 2 oracle_ids — exceeds threshold of 20
        con = duckdb.connect(":memory:")
        rows = []
        for i in range(21):
            name = f"Conflict {i}"
            rows.append((f"u{i}a", f"u{i}a", name, "10E", f"{i}a", f"oa{i}"))
            rows.append((f"u{i}b", f"u{i}b", name, "10E", f"{i}b", f"ob{i}"))
        _make_silver_cards(con, rows)
        result = _check_oracle_id_conflicts(con)
        assert result.status == "FAIL"
        assert "21" in result.detail
        con.close()


class TestCheckSilverPricesNegativeEur:
    def _make_prices(self, con: duckdb.DuckDBPyConnection, rows: list[tuple]) -> None:
        con.execute(
            "CREATE TABLE silver_prices_history"
            " (uuid VARCHAR, snapshot_date DATE, eur FLOAT)"
        )
        for r in rows:
            con.execute("INSERT INTO silver_prices_history VALUES (?, ?, ?)", list(r))

    def test_pass_when_all_prices_positive_today(self):
        con = duckdb.connect(":memory:")
        today = datetime.date(2026, 6, 22)
        self._make_prices(con, [("u1", today, 1.5)])
        result = _check_silver_prices_no_negative_eur(con, today)
        assert result.status == "PASS"
        con.close()

    def test_fail_when_zero_price_today(self):
        con = duckdb.connect(":memory:")
        today = datetime.date(2026, 6, 22)
        self._make_prices(con, [("u1", today, 0.0)])
        result = _check_silver_prices_no_negative_eur(con, today)
        assert result.status == "FAIL"
        assert "1 rows" in result.detail
        con.close()

    def test_ignores_other_dates(self):
        con = duckdb.connect(":memory:")
        today = datetime.date(2026, 6, 22)
        yesterday = datetime.date(2026, 6, 21)
        self._make_prices(con, [("u1", yesterday, 0.0)])
        result = _check_silver_prices_no_negative_eur(con, today)
        assert result.status == "PASS"
        con.close()


class TestCheckGoldMlDatasetHasTarget:
    def test_pass_when_some_targets_non_null(self):
        con = duckdb.connect(":memory:")
        con.execute(
            "CREATE TABLE gold_ml_dataset (uuid VARCHAR, target_price_7d FLOAT)"
        )
        con.execute("INSERT INTO gold_ml_dataset VALUES ('u1', 5.0)")
        con.execute("INSERT INTO gold_ml_dataset VALUES ('u2', NULL)")
        result = _check_gold_ml_dataset_has_target(con)
        assert result.status == "PASS"
        assert "1 rows" in result.detail
        con.close()

    def test_fail_when_all_targets_null(self):
        con = duckdb.connect(":memory:")
        con.execute(
            "CREATE TABLE gold_ml_dataset (uuid VARCHAR, target_price_7d FLOAT)"
        )
        con.execute("INSERT INTO gold_ml_dataset VALUES ('u1', NULL)")
        result = _check_gold_ml_dataset_has_target(con)
        assert result.status == "FAIL"
        assert "100% NULL" in result.detail
        con.close()


def _make_all_dbs(tmp_path: Path, today: datetime.date) -> tuple[str, str, str]:
    """Create minimal valid Bronze/Silver/Gold DuckDB files under tmp_path."""
    bronze_path = str(tmp_path / "bronze.duckdb")
    b = duckdb.connect(bronze_path)
    b.execute("CREATE TABLE bronze_scryfall_cards (id VARCHAR)")
    b.execute("INSERT INTO bronze_scryfall_cards VALUES ('x')")
    b.execute("CREATE TABLE bronze_mtgjson_cards (uuid VARCHAR)")
    b.execute("INSERT INTO bronze_mtgjson_cards VALUES ('x')")
    b.execute("CREATE TABLE bronze_mtgjson_prices_history (uuid VARCHAR)")
    b.execute("INSERT INTO bronze_mtgjson_prices_history VALUES ('x')")
    b.close()

    silver_path = str(tmp_path / "silver.duckdb")
    s = duckdb.connect(silver_path)
    s.execute("""
        CREATE TABLE silver_cards (
            uuid VARCHAR, canonical_uuid VARCHAR, name VARCHAR,
            set_code VARCHAR, collector_number VARCHAR, oracle_id VARCHAR
        )
    """)
    s.execute(
        "INSERT INTO silver_cards VALUES ('u1', 'u1', 'Serra Angel', '10E', '1', 'o1')"
    )
    s.execute(
        "CREATE TABLE silver_prices_history (uuid VARCHAR, snapshot_date DATE, eur FLOAT)"
    )
    s.execute("INSERT INTO silver_prices_history VALUES ('u1', ?, 1.5)", [today])
    s.execute(
        "CREATE TABLE silver_language_prices_history (scryfall_id VARCHAR, snapshot_date DATE)"
    )
    s.execute("INSERT INTO silver_language_prices_history VALUES ('x', ?)", [today])
    s.execute("CREATE TABLE silver_meta_history (id VARCHAR, snapshot_date DATE)")
    s.execute("INSERT INTO silver_meta_history VALUES ('x', ?)", [today])
    s.execute(
        "CREATE TABLE silver_format_staples_history (id VARCHAR, snapshot_date DATE)"
    )
    s.execute("INSERT INTO silver_format_staples_history VALUES ('x', ?)", [today])
    s.execute(
        "CREATE TABLE silver_tournament_results_history"
        " (id VARCHAR, tournament_date VARCHAR)"
    )
    s.execute(
        "INSERT INTO silver_tournament_results_history VALUES ('x', '2026-06-20')"
    )
    s.close()

    gold_path = str(tmp_path / "gold.duckdb")
    g = duckdb.connect(gold_path)
    g.execute("CREATE TABLE gold_card_features (uuid VARCHAR)")
    g.execute("INSERT INTO gold_card_features VALUES ('u1')")
    g.execute("CREATE TABLE gold_price_features (uuid VARCHAR)")
    g.execute("INSERT INTO gold_price_features VALUES ('u1')")
    g.execute("CREATE TABLE gold_language_premiums (scryfall_id VARCHAR)")
    g.execute("INSERT INTO gold_language_premiums VALUES ('x')")
    g.execute("CREATE TABLE gold_demand_signals (id VARCHAR)")
    g.execute("INSERT INTO gold_demand_signals VALUES ('x')")
    g.execute("CREATE TABLE gold_format_staples (id VARCHAR)")
    g.execute("INSERT INTO gold_format_staples VALUES ('x')")
    g.execute("CREATE TABLE gold_tournament_signals (oracle_id VARCHAR)")
    g.execute("INSERT INTO gold_tournament_signals VALUES ('o1')")
    g.execute("CREATE TABLE gold_ml_dataset (uuid VARCHAR, target_price_7d FLOAT)")
    g.execute("INSERT INTO gold_ml_dataset VALUES ('u1', 5.0)")
    g.close()

    return bronze_path, silver_path, gold_path


class TestRunHealthChecks:
    def test_all_pass_returns_list_of_results(self, tmp_path):
        today = datetime.date(2026, 6, 22)
        bronze, silver, gold = _make_all_dbs(tmp_path, today)
        results = run_health_checks(bronze, silver, gold, today)
        assert len(results) > 0
        assert all(r.status in ("PASS", "WARN") for r in results)

    def test_exits_one_on_any_fail(self, tmp_path):
        today = datetime.date(2026, 6, 22)
        bronze, silver, gold = _make_all_dbs(tmp_path, today)
        # Corrupt: gold_ml_dataset has all-NULL targets
        g = duckdb.connect(gold)
        g.execute("DELETE FROM gold_ml_dataset")
        g.execute("INSERT INTO gold_ml_dataset VALUES ('u1', NULL)")
        g.close()
        with pytest.raises(SystemExit) as exc:
            run_health_checks(bronze, silver, gold, today)
        assert exc.value.code == 1

    def test_skips_silver_quality_when_structure_fails(self, tmp_path):
        today = datetime.date(2026, 6, 22)
        bronze, silver, gold = _make_all_dbs(tmp_path, today)
        # Remove silver_cards — structure FAIL should prevent quality checks
        s = duckdb.connect(silver)
        s.execute("DROP TABLE silver_cards")
        s.close()
        with pytest.raises(SystemExit):
            run_health_checks(bronze, silver, gold, today)


def test_check_result_warn():
    r = CheckResult(
        name="schema drift", layer="bronze", status="WARN",
        detail="new combo: ('newretailer', 'retail', 'normal')"
    )
    assert r.status == "WARN"


_EXPECTED_COMBOS = {
    ("cardmarket", "retail",  "normal"),
    ("cardmarket", "retail",  "foil"),
    ("cardmarket", "buylist", "normal"),
    ("tcgplayer",  "retail",  "normal"),
    ("tcgplayer",  "retail",  "foil"),
    ("tcgplayer",  "buylist", "normal"),
}


class TestCheckBronzePricesSchemaWarn:
    def _make_eav_table(self, con: duckdb.DuckDBPyConnection, rows: list) -> None:
        con.execute("""
            CREATE TABLE bronze_mtgjson_prices_history (
                uuid VARCHAR, snapshot_date VARCHAR,
                retailer VARCHAR, tx_type VARCHAR, finish VARCHAR, price FLOAT
            )
        """)
        for r in rows:
            con.execute(
                "INSERT INTO bronze_mtgjson_prices_history VALUES (?, ?, ?, ?, ?, ?)",
                list(r),
            )

    def test_pass_when_combos_match_expected(self):
        con = duckdb.connect(":memory:")
        today = datetime.date(2026, 6, 24)
        rows = [(f"u{i}", today.isoformat(), r, t, f, 1.0)
                for i, (r, t, f) in enumerate(_EXPECTED_COMBOS)]
        self._make_eav_table(con, rows)
        results = _check_bronze_prices_schema_drift(con, today, _EXPECTED_COMBOS)
        assert all(r.status == "PASS" for r in results)
        con.close()

    def test_warn_on_new_retailer_not_in_map(self):
        con = duckdb.connect(":memory:")
        today = datetime.date(2026, 6, 24)
        rows = [("u1", today.isoformat(), "newretailer", "retail", "normal", 1.0)]
        self._make_eav_table(con, rows)
        results = _check_bronze_prices_schema_drift(con, today, _EXPECTED_COMBOS)
        statuses = {r.status for r in results}
        assert "WARN" in statuses
        con.close()

    def test_warn_on_missing_expected_combo(self):
        con = duckdb.connect(":memory:")
        today = datetime.date(2026, 6, 24)
        rows = [("u1", today.isoformat(), "cardmarket", "retail", "normal", 1.0)]
        self._make_eav_table(con, rows)
        results = _check_bronze_prices_schema_drift(con, today, _EXPECTED_COMBOS)
        statuses = {r.status for r in results}
        assert "WARN" in statuses
        con.close()

    def test_returns_empty_when_table_missing(self):
        con = duckdb.connect(":memory:")
        today = datetime.date(2026, 6, 24)
        results = _check_bronze_prices_schema_drift(con, today, _EXPECTED_COMBOS)
        assert results == []
        con.close()

    def test_returns_empty_when_table_has_no_eav_columns(self):
        con = duckdb.connect(":memory:")
        today = datetime.date(2026, 6, 24)
        con.execute("CREATE TABLE bronze_mtgjson_prices_history (uuid VARCHAR)")
        results = _check_bronze_prices_schema_drift(con, today, _EXPECTED_COMBOS)
        assert results == []
        con.close()

    def test_warn_does_not_cause_exit_in_run_health_checks(self, tmp_path):
        today = datetime.date(2026, 6, 24)
        bronze_path, silver_path, gold_path = _make_all_dbs(tmp_path, today)
        b = duckdb.connect(bronze_path)
        b.execute("DROP TABLE IF EXISTS bronze_mtgjson_prices_history")
        b.execute("""
            CREATE TABLE bronze_mtgjson_prices_history (
                uuid VARCHAR, snapshot_date VARCHAR,
                retailer VARCHAR, tx_type VARCHAR, finish VARCHAR, price FLOAT
            )
        """)
        b.execute(
            "INSERT INTO bronze_mtgjson_prices_history VALUES (?, ?, ?, ?, ?, ?)",
            ["u1", today.isoformat(), "newretailer", "retail", "normal", 1.0],
        )
        b.close()
        results = run_health_checks(bronze_path, silver_path, gold_path, today)
        assert any(r.status == "WARN" for r in results)
