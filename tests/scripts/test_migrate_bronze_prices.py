"""Tests for scripts/migrate_bronze_prices.py."""

import json
import shutil
from pathlib import Path

import duckdb
import pytest

from scripts.migrate_bronze_prices import migrate_mtgjson_prices, migrate_scryfall_prices


def _make_mtgjson_source(path: str) -> None:
    """Create a Bronze DB with old JSON-column schema for bronze_mtgjson_prices_history."""
    con = duckdb.connect(path)
    con.execute("""
        CREATE TABLE bronze_mtgjson_prices_history (
            uuid          VARCHAR,
            snapshot_date VARCHAR,
            paper         VARCHAR,
            mtgo          VARCHAR
        )
    """)
    paper = json.dumps({
        "cardmarket": {
            "retail": {"normal": {"2026-05-11": 3.20}, "foil": {"2026-05-11": 8.50}},
            "buylist": {"normal": {"2026-05-11": 1.80}},
        },
        "tcgplayer": {
            "retail": {"normal": {"2026-05-11": 3.50}, "foil": {"2026-05-11": 9.00}},
            "buylist": {"normal": {"2026-05-11": 2.10}},
        },
    })
    con.execute(
        "INSERT INTO bronze_mtgjson_prices_history VALUES (?, ?, ?, NULL)",
        ["uuid-1", "2026-05-11", paper],
    )
    con.close()


def _make_scryfall_source(path: str) -> None:
    """Create a Bronze DB with old JSON-column schema for bronze_scryfall_prices_history."""
    con = duckdb.connect(path)
    con.execute("""
        CREATE TABLE bronze_scryfall_prices_history (
            id            VARCHAR,
            snapshot_date VARCHAR,
            prices        VARCHAR
        )
    """)
    prices = json.dumps({"eur": "3.20", "eur_foil": "8.50", "usd": "3.50", "usd_foil": None, "tix": "0.05"})
    con.execute(
        "INSERT INTO bronze_scryfall_prices_history VALUES (?, ?, ?)",
        ["s1", "2026-05-11", prices],
    )
    con.close()


def _make_target_with_old_mtgjson(source_path: str, target_path: str) -> None:
    """Copy the old mtgjson prices table into the target DB (simulates live DB)."""
    shutil.copy(source_path, target_path)


def _make_target_with_old_scryfall(source_path: str, target_path: str) -> None:
    shutil.copy(source_path, target_path)


class TestMigrateMtgjsonPrices:
    def test_creates_eav_schema(self, tmp_path):
        source = str(tmp_path / "source.duckdb")
        target = str(tmp_path / "target.duckdb")
        _make_mtgjson_source(source)
        _make_target_with_old_mtgjson(source, target)

        migrate_mtgjson_prices(source, target)

        con = duckdb.connect(target, read_only=True)
        cols = {r[0] for r in con.execute("DESCRIBE bronze_mtgjson_prices_history").fetchall()}
        con.close()
        assert cols == {"uuid", "snapshot_date", "retailer", "tx_type", "finish", "price"}

    def test_no_paper_or_mtgo_columns(self, tmp_path):
        source = str(tmp_path / "source.duckdb")
        target = str(tmp_path / "target.duckdb")
        _make_mtgjson_source(source)
        _make_target_with_old_mtgjson(source, target)

        migrate_mtgjson_prices(source, target)

        con = duckdb.connect(target, read_only=True)
        cols = {r[0] for r in con.execute("DESCRIBE bronze_mtgjson_prices_history").fetchall()}
        con.close()
        assert "paper" not in cols
        assert "mtgo" not in cols

    def test_extracts_all_retailers_from_source(self, tmp_path):
        source = str(tmp_path / "source.duckdb")
        target = str(tmp_path / "target.duckdb")
        _make_mtgjson_source(source)
        _make_target_with_old_mtgjson(source, target)

        migrate_mtgjson_prices(source, target)

        con = duckdb.connect(target, read_only=True)
        retailers = {r[0] for r in con.execute(
            "SELECT DISTINCT retailer FROM bronze_mtgjson_prices_history"
        ).fetchall()}
        con.close()
        assert "cardmarket" in retailers
        assert "tcgplayer" in retailers

    def test_extracts_correct_price_value(self, tmp_path):
        source = str(tmp_path / "source.duckdb")
        target = str(tmp_path / "target.duckdb")
        _make_mtgjson_source(source)
        _make_target_with_old_mtgjson(source, target)

        migrate_mtgjson_prices(source, target)

        con = duckdb.connect(target, read_only=True)
        row = con.execute(
            "SELECT price FROM bronze_mtgjson_prices_history"
            " WHERE retailer='cardmarket' AND tx_type='retail' AND finish='normal'"
        ).fetchone()
        con.close()
        assert row is not None
        assert row[0] == pytest.approx(3.20)

    def test_preserves_uuid_and_date(self, tmp_path):
        source = str(tmp_path / "source.duckdb")
        target = str(tmp_path / "target.duckdb")
        _make_mtgjson_source(source)
        _make_target_with_old_mtgjson(source, target)

        migrate_mtgjson_prices(source, target)

        con = duckdb.connect(target, read_only=True)
        row = con.execute(
            "SELECT uuid, snapshot_date FROM bronze_mtgjson_prices_history LIMIT 1"
        ).fetchone()
        con.close()
        assert row[0] == "uuid-1"
        assert str(row[1]) == "2026-05-11"

    def test_returns_eav_row_count(self, tmp_path):
        source = str(tmp_path / "source.duckdb")
        target = str(tmp_path / "target.duckdb")
        _make_mtgjson_source(source)
        _make_target_with_old_mtgjson(source, target)

        count = migrate_mtgjson_prices(source, target)
        # fixture has: cardmarket retail normal+foil, buylist normal; tcgplayer retail normal+foil, buylist normal = 6 EAV rows
        assert count == 6

    def test_empty_source_produces_no_rows(self, tmp_path):
        source = str(tmp_path / "source.duckdb")
        target = str(tmp_path / "target.duckdb")
        con = duckdb.connect(source)
        con.execute(
            "CREATE TABLE bronze_mtgjson_prices_history"
            " (uuid VARCHAR, snapshot_date VARCHAR, paper VARCHAR, mtgo VARCHAR)"
        )
        con.close()
        _make_target_with_old_mtgjson(source, target)

        count = migrate_mtgjson_prices(source, target)
        assert count == 0


class TestMigrateScryfallPrices:
    def test_creates_scalar_columns(self, tmp_path):
        source = str(tmp_path / "source.duckdb")
        target = str(tmp_path / "target.duckdb")
        _make_scryfall_source(source)
        _make_target_with_old_scryfall(source, target)

        migrate_scryfall_prices(source, target)

        con = duckdb.connect(target, read_only=True)
        cols = {r[0] for r in con.execute("DESCRIBE bronze_scryfall_prices_history").fetchall()}
        con.close()
        assert "eur" in cols
        assert "eur_foil" in cols
        assert "usd" in cols
        assert "usd_foil" in cols
        assert "prices" not in cols
        assert "tix" in cols

    def test_extracts_correct_scalar_values(self, tmp_path):
        source = str(tmp_path / "source.duckdb")
        target = str(tmp_path / "target.duckdb")
        _make_scryfall_source(source)
        _make_target_with_old_scryfall(source, target)

        migrate_scryfall_prices(source, target)

        con = duckdb.connect(target, read_only=True)
        row = con.execute(
            "SELECT eur, eur_foil, usd, usd_foil FROM bronze_scryfall_prices_history"
        ).fetchone()
        con.close()
        assert row is not None
        assert row[0] == pytest.approx(3.20)
        assert row[1] == pytest.approx(8.50)
        assert row[2] == pytest.approx(3.50)
        assert row[3] is None

    def test_null_prices_dict_row_produces_all_null(self, tmp_path):
        source = str(tmp_path / "source.duckdb")
        target = str(tmp_path / "target.duckdb")
        con = duckdb.connect(source)
        con.execute("""
            CREATE TABLE bronze_scryfall_prices_history (
                id VARCHAR, snapshot_date VARCHAR, prices VARCHAR
            )
        """)
        con.execute(
            "INSERT INTO bronze_scryfall_prices_history VALUES (?, ?, NULL)",
            ["s1", "2026-05-11"],
        )
        con.close()
        _make_target_with_old_scryfall(source, target)

        migrate_scryfall_prices(source, target)

        con = duckdb.connect(target, read_only=True)
        row = con.execute(
            "SELECT eur FROM bronze_scryfall_prices_history"
        ).fetchone()
        con.close()
        assert row is not None and row[0] is None

    def test_returns_row_count(self, tmp_path):
        source = str(tmp_path / "source.duckdb")
        target = str(tmp_path / "target.duckdb")
        _make_scryfall_source(source)
        _make_target_with_old_scryfall(source, target)

        count = migrate_scryfall_prices(source, target)
        assert count == 1

    def test_tix_column_present(self, tmp_path):
        source = str(tmp_path / "source.duckdb")
        target = str(tmp_path / "target.duckdb")
        _make_scryfall_source(source)
        _make_target_with_old_scryfall(source, target)

        migrate_scryfall_prices(source, target)

        con = duckdb.connect(target, read_only=True)
        cols = {r[0] for r in con.execute("DESCRIBE bronze_scryfall_prices_history").fetchall()}
        con.close()
        assert "tix" in cols

    def test_tix_value_extracted(self, tmp_path):
        source = str(tmp_path / "source.duckdb")
        target = str(tmp_path / "target.duckdb")
        _make_scryfall_source(source)
        _make_target_with_old_scryfall(source, target)

        migrate_scryfall_prices(source, target)

        con = duckdb.connect(target, read_only=True)
        row = con.execute("SELECT tix FROM bronze_scryfall_prices_history").fetchone()
        con.close()
        assert row is not None
        assert row[0] == pytest.approx(0.05)
