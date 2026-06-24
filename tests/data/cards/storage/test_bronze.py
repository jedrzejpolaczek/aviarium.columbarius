"""Unit tests for src/data/cards/storage/bronze/ (config, writers, storage)."""

from unittest.mock import MagicMock, patch

import duckdb
import pandas as pd
import pytest
from pydantic import BaseModel

from src.data.cards.storage.bronze import STORAGE_CONFIG, BronzeStorage
from src.data.cards.storage.bronze.storage import (
    _extract_paper_eav_rows,
    _records_to_df,
)
from src.data.cards.storage.errors import StorageWriteError


# ---------------------------------------------------------------------------
# Minimal test models
# ---------------------------------------------------------------------------


class _Card(BaseModel):
    id: str
    name: str
    tags: list[str] | None = None


class _Item(BaseModel):
    uuid: str
    value: float | None = None


def _bronze() -> BronzeStorage:
    """Return a BronzeStorage backed by an in-memory DuckDB database."""
    return BronzeStorage(":memory:")


# ---------------------------------------------------------------------------
# _records_to_df (module-level helper, replaces BronzeStorage._to_df)
# ---------------------------------------------------------------------------


class TestToDF:
    def test_returns_dataframe(self):
        df = _records_to_df([_Card(id="1", name="Alpha")])
        assert isinstance(df, pd.DataFrame)

    def test_row_count_matches_records(self):
        records = [_Card(id=str(i), name=f"Card{i}") for i in range(5)]
        df = _records_to_df(records)
        assert len(df) == 5

    def test_list_cell_is_plain_list(self):
        """_records_to_df does not serialize lists — DuckDBWriter handles that downstream."""
        df = _records_to_df([_Card(id="1", name="Alpha", tags=["a", "b"])])
        assert isinstance(df["tags"].iloc[0], list)
        assert df["tags"].iloc[0] == ["a", "b"]

    def test_none_list_cell_remains_null(self):
        df = _records_to_df([_Card(id="1", name="Alpha", tags=None)])
        assert pd.isna(df["tags"].iloc[0])


# ---------------------------------------------------------------------------
# _extract_paper_eav_rows
# ---------------------------------------------------------------------------


class TestExtractPaperEavRows:
    def test_returns_empty_for_none(self):
        assert _extract_paper_eav_rows(None, "u1", "2026-05-11") == []

    def test_returns_empty_for_empty_dict(self):
        assert _extract_paper_eav_rows({}, "u1", "2026-05-11") == []

    def test_emits_one_row_per_price_point(self):
        paper = {"cardmarket": {"retail": {"normal": {"2026-05-11": 3.20}}}}
        rows = _extract_paper_eav_rows(paper, "u1", "2026-05-11")
        assert len(rows) == 1
        assert rows[0] == {
            "uuid": "u1",
            "snapshot_date": "2026-05-11",
            "retailer": "cardmarket",
            "tx_type": "retail",
            "finish": "normal",
            "price": pytest.approx(3.20),
        }

    def test_captures_all_retailers_including_cardkingdom(self):
        paper = {
            "cardmarket":  {"retail": {"normal": {"2026-05-11": 3.20}}},
            "tcgplayer":   {"retail": {"normal": {"2026-05-11": 3.50}}},
            "cardkingdom": {"retail": {"normal": {"2026-05-11": 4.00}}},
        }
        rows = _extract_paper_eav_rows(paper, "u1", "2026-05-11")
        retailers = {r["retailer"] for r in rows}
        assert retailers == {"cardmarket", "tcgplayer", "cardkingdom"}

    def test_lookback_selects_max_date_leq_target(self):
        paper = {"cardmarket": {"retail": {"normal": {"2026-05-10": 1.0, "2026-05-11": 3.20}}}}
        rows = _extract_paper_eav_rows(paper, "u1", "2026-05-11")
        assert rows[0]["price"] == pytest.approx(3.20)

    def test_excludes_dates_after_target(self):
        paper = {"cardmarket": {"retail": {"normal": {"2026-05-12": 5.00, "2026-05-10": 1.0}}}}
        rows = _extract_paper_eav_rows(paper, "u1", "2026-05-11")
        assert len(rows) == 1
        assert rows[0]["price"] == pytest.approx(1.0)

    def test_no_date_leq_target_emits_no_row(self):
        paper = {"cardmarket": {"retail": {"normal": {"2026-05-12": 5.00}}}}
        assert _extract_paper_eav_rows(paper, "u1", "2026-05-11") == []

    def test_captures_all_tx_types(self):
        paper = {
            "cardmarket": {
                "retail":  {"normal": {"2026-05-11": 3.20}},
                "buylist": {"normal": {"2026-05-11": 1.80}},
            }
        }
        rows = _extract_paper_eav_rows(paper, "u1", "2026-05-11")
        assert {r["tx_type"] for r in rows} == {"retail", "buylist"}

    def test_captures_all_finishes_including_etched(self):
        paper = {
            "cardmarket": {
                "retail": {
                    "normal": {"2026-05-11": 3.20},
                    "foil":   {"2026-05-11": 8.50},
                    "etched": {"2026-05-11": 12.00},
                }
            }
        }
        rows = _extract_paper_eav_rows(paper, "u1", "2026-05-11")
        assert {r["finish"] for r in rows} == {"normal", "foil", "etched"}

    def test_price_is_float_not_str(self):
        paper = {"cardmarket": {"retail": {"normal": {"2026-05-11": "3.20"}}}}
        rows = _extract_paper_eav_rows(paper, "u1", "2026-05-11")
        assert isinstance(rows[0]["price"], float)


# ---------------------------------------------------------------------------
# BronzeStorage._full_load_table
# ---------------------------------------------------------------------------


class TestFullLoadTable:
    def test_creates_table_from_records(self):
        with _bronze() as b:
            b._full_load_table([_Card(id="1", name="Alpha")], "test_table")
            row = b._con.execute("SELECT count(*) FROM test_table").fetchone()
        assert row is not None and row[0] == 1

    def test_replaces_table_on_second_call(self):
        with _bronze() as b:
            b._full_load_table([_Card(id="1", name="Alpha")], "test_table")
            b._full_load_table(
                [_Card(id="2", name="Beta"), _Card(id="3", name="Gamma")],
                "test_table",
            )
            row = b._con.execute("SELECT count(*) FROM test_table").fetchone()
        assert row is not None and row[0] == 2

    def test_persists_field_values(self):
        with _bronze() as b:
            b._full_load_table([_Card(id="abc", name="Lightning Bolt")], "test_table")
            row = b._con.execute(
                "SELECT name FROM test_table WHERE id='abc'"
            ).fetchone()
        assert row is not None and row[0] == "Lightning Bolt"

    def test_skips_when_records_empty(self):
        with _bronze() as b:
            b._full_load_table([], "test_table")
            row = b._con.execute(
                "SELECT count(*) FROM information_schema.tables "
                "WHERE table_name='test_table'"
            ).fetchone()
        assert row is not None and row[0] == 0

    def test_raises_storage_write_error_on_duckdb_failure(self):
        # DuckDBPyConnection.execute is a C extension attribute — replace the
        # whole connection object with a MagicMock to inject the error.
        with _bronze() as b:
            mock_con = MagicMock()
            mock_con.execute.side_effect = duckdb.Error("boom")
            b._con = mock_con
            b._writer._con = mock_con
            with pytest.raises(StorageWriteError, match="Failed to full-load"):
                b._full_load_table([_Card(id="1", name="X")], "test_table")


# ---------------------------------------------------------------------------
# BronzeStorage._incremental_load
# ---------------------------------------------------------------------------


class TestIncrementalLoad:
    def test_creates_table_when_not_exists(self):
        with _bronze() as b:
            b._incremental_load([_Card(id="1", name="Alpha")], "test_table", "id")
            row = b._con.execute("SELECT count(*) FROM test_table").fetchone()
        assert row is not None and row[0] == 1

    def test_upserts_existing_record(self):
        with _bronze() as b:
            b._incremental_load([_Card(id="1", name="Old")], "test_table", "id")
            b._incremental_load([_Card(id="1", name="New")], "test_table", "id")
            row = b._con.execute("SELECT name FROM test_table WHERE id='1'").fetchone()
        assert row is not None and row[0] == "New"

    def test_leaves_unrelated_records_intact(self):
        with _bronze() as b:
            b._incremental_load(
                [_Card(id="1", name="Alpha"), _Card(id="2", name="Beta")],
                "test_table",
                "id",
            )
            b._incremental_load(
                [_Card(id="1", name="Alpha Updated")], "test_table", "id"
            )
            row = b._con.execute("SELECT count(*) FROM test_table").fetchone()
        assert row is not None and row[0] == 2

    def test_skips_when_records_empty(self):
        with _bronze() as b:
            b._incremental_load([], "test_table", "id")
            row = b._con.execute(
                "SELECT count(*) FROM information_schema.tables "
                "WHERE table_name='test_table'"
            ).fetchone()
        assert row is not None and row[0] == 0

    def test_raises_storage_write_error_on_duckdb_failure(self):
        with _bronze() as b:
            mock_con = MagicMock()
            mock_con.execute.side_effect = duckdb.Error("boom")
            b._con = mock_con
            b._writer._con = mock_con
            with pytest.raises(StorageWriteError, match="Failed to upsert"):
                b._incremental_load([_Card(id="1", name="X")], "test_table", "id")


# ---------------------------------------------------------------------------
# BronzeStorage._snapshot
# ---------------------------------------------------------------------------


class TestSnapshot:
    def test_creates_history_table_on_first_call(self):
        with _bronze() as b:
            b._snapshot([_Card(id="1", name="Alpha")], "id", "test_history")
            row = b._con.execute("SELECT count(*) FROM test_history").fetchone()
        assert row is not None and row[0] == 1

    def test_skips_duplicate_key_date_pair(self):
        with _bronze() as b:
            b._snapshot([_Card(id="1", name="Alpha")], "id", "test_history")
            b._snapshot([_Card(id="1", name="Alpha")], "id", "test_history")
            row = b._con.execute("SELECT count(*) FROM test_history").fetchone()
        assert row is not None and row[0] == 1

    def test_appends_new_row_for_different_date(self):
        from datetime import date as date_cls

        record = _Card(id="1", name="Alpha")
        with _bronze() as b:
            with patch("src.data.cards.storage.bronze.storage.date") as mock_date:
                mock_date.today.side_effect = [
                    date_cls.fromisoformat("2026-05-01"),
                    date_cls.fromisoformat("2026-05-02"),
                ]
                b._snapshot([record], "id", "test_history")
                b._snapshot([record], "id", "test_history")
            row = b._con.execute("SELECT count(*) FROM test_history").fetchone()
        assert row is not None and row[0] == 2

    def test_snapshots_only_specified_fields(self):
        with _bronze() as b:
            b._snapshot(
                [_Card(id="1", name="Alpha", tags=["x"])],
                "id",
                "test_history",
                fields=["name"],
            )
            cols = {r[0] for r in b._con.execute("DESCRIBE test_history").fetchall()}
        assert "name" in cols
        assert "tags" not in cols

    def test_full_record_snapshotted_when_fields_is_none(self):
        with _bronze() as b:
            b._snapshot(
                [_Card(id="1", name="Alpha", tags=["x"])],
                "id",
                "test_history",
                fields=None,
            )
            cols = {r[0] for r in b._con.execute("DESCRIBE test_history").fetchall()}
        assert "name" in cols
        assert "tags" in cols

    def test_skips_when_records_empty(self):
        with _bronze() as b:
            b._snapshot([], "id", "test_history")
            row = b._con.execute(
                "SELECT count(*) FROM information_schema.tables "
                "WHERE table_name='test_history'"
            ).fetchone()
        assert row is not None and row[0] == 0

    def test_raises_storage_write_error_on_duckdb_failure(self):
        with _bronze() as b:
            mock_con = MagicMock()
            mock_con.execute.side_effect = duckdb.Error("boom")
            b._con = mock_con
            b._writer._con = mock_con
            with pytest.raises(StorageWriteError, match="Failed to append"):
                b._snapshot([_Card(id="1", name="X")], "id", "test_history")


# ---------------------------------------------------------------------------
# STORAGE_CONFIG
# ---------------------------------------------------------------------------


class TestStorageConfig:
    def test_format_staples_is_registered(self):
        assert "format_staples" in STORAGE_CONFIG

    def test_format_staples_has_no_main_table(self):
        assert STORAGE_CONFIG["format_staples"].table is None

    def test_format_staples_snapshots_to_history_table(self):
        snaps = STORAGE_CONFIG["format_staples"].snapshots
        assert len(snaps) == 1
        assert snaps[0].history_table == "bronze_format_staples_history"

    def test_format_staples_key_is_id(self):
        assert STORAGE_CONFIG["format_staples"].key == "id"


# ---------------------------------------------------------------------------
# BronzeStorage._process_sources
# ---------------------------------------------------------------------------


class TestProcessSources:
    def test_full_load_used_when_update_false(self):
        with _bronze() as b:
            with (
                patch.object(b, "_full_load_table") as mock_full,
                patch.object(b, "_incremental_load"),
                patch.object(b, "_snapshot"),
            ):
                b._process_sources({"scryfall": ([], [])}, update=False)
            called_tables = [c.args[1] for c in mock_full.call_args_list]
            assert "bronze_scryfall_cards" in called_tables

    def test_incremental_load_used_when_update_true(self):
        with _bronze() as b:
            with (
                patch.object(b, "_incremental_load") as mock_inc,
                patch.object(b, "_full_load_table"),
                patch.object(b, "_snapshot"),
            ):
                b._process_sources({"tournament_results": ([], [])}, update=True)
            mock_inc.assert_called()

    def test_full_load_used_when_update_false_even_for_incremental_source(self):
        with _bronze() as b:
            with (
                patch.object(b, "_full_load_table") as mock_full,
                patch.object(b, "_incremental_load") as mock_inc,
                patch.object(b, "_snapshot"),
            ):
                b._process_sources({"tournament_results": ([], [])}, update=False)
            called_tables = [c.args[1] for c in mock_full.call_args_list]
            assert "bronze_tournament_results" in called_tables
            mock_inc.assert_not_called()

    def test_snapshot_called_for_sources_with_snapshot_config(self):
        with _bronze() as b:
            with (
                patch.object(b, "_full_load_table"),
                patch.object(b, "_snapshot") as mock_snap,
            ):
                b._process_sources({"scryfall": ([], [])}, update=False)
            # scryfall: 1 (meta_history), format_staples: 1 = 2 total
            assert mock_snap.call_count >= 1

    def test_error_in_one_source_does_not_stop_others(self):
        with _bronze() as b:
            with (
                patch.object(
                    b, "_full_load_table", side_effect=StorageWriteError("fail")
                ),
                patch.object(b, "_snapshot"),
            ):
                b._process_sources(
                    {"scryfall": ([], []), "mtgjson_cards": ([], [])}, update=False
                )

    def test_empty_results_runs_without_error(self):
        with _bronze() as b:
            with (
                patch.object(b, "_full_load_table"),
                patch.object(b, "_snapshot"),
            ):
                b._process_sources({}, update=False)


# ---------------------------------------------------------------------------
# BronzeStorage.populate
# ---------------------------------------------------------------------------


class TestPopulate:
    def test_full_load_called_for_table_sources(self):
        with _bronze() as b:
            with (
                patch.object(b, "_full_load_table") as mock_full,
                patch.object(b, "_snapshot"),
                patch.object(b, "seed_historical_prices"),
            ):
                b.populate({"scryfall": ([], [])})
            # STORAGE_CONFIG has multiple table sources; verify at least scryfall triggered it
            called_tables = [c.args[1] for c in mock_full.call_args_list]
            assert "bronze_scryfall_cards" in called_tables

    def test_snapshot_called_for_scryfall_meta_history(self):
        with _bronze() as b:
            with (
                patch.object(b, "_full_load_table"),
                patch.object(b, "_snapshot") as mock_snap,
                patch.object(b, "seed_historical_prices"),
                patch.object(b, "_snapshot_scryfall_prices"),
            ):
                b.populate({"scryfall": ([], [])})
            history_tables = [
                c.kwargs["history_table"] for c in mock_snap.call_args_list
            ]
            assert "bronze_scryfall_meta_history" in history_tables
            assert "bronze_scryfall_prices_history" not in history_tables

    def test_format_staples_snapshotted_via_storage_config(self):
        with _bronze() as b:
            with (
                patch.object(b, "_full_load_table"),
                patch.object(b, "_snapshot") as mock_snap,
                patch.object(b, "seed_historical_prices"),
            ):
                b.populate({"format_staples": ([_Card(id="X__modern", name="X")], [])})
            history_tables = [
                c.kwargs["history_table"] for c in mock_snap.call_args_list
            ]
            assert "bronze_format_staples_history" in history_tables

    def test_storage_write_error_does_not_stop_other_sources(self):
        with _bronze() as b:
            with (
                patch.object(
                    b, "_full_load_table", side_effect=StorageWriteError("fail")
                ),
                patch.object(b, "_snapshot"),
                patch.object(b, "seed_historical_prices"),
            ):
                b.populate({"scryfall": ([], []), "mtgjson_cards": ([], [])})

    def test_empty_results_runs_without_error(self):
        with _bronze() as b:
            with (
                patch.object(b, "_full_load_table"),
                patch.object(b, "_snapshot"),
                patch.object(b, "seed_historical_prices"),
            ):
                b.populate({})


# ---------------------------------------------------------------------------
# BronzeStorage.daily_update
# ---------------------------------------------------------------------------


class TestDailyUpdate:
    def test_incremental_load_called_for_incremental_sources(self):
        with _bronze() as b:
            with (
                patch.object(b, "_incremental_load") as mock_inc,
                patch.object(b, "_full_load_table"),
                patch.object(b, "_snapshot"),
            ):
                b.daily_update({"scryfall": ([], [])})
            mock_inc.assert_called()

    def test_format_staples_snapshotted_via_storage_config(self):
        with _bronze() as b:
            with (
                patch.object(b, "_incremental_load"),
                patch.object(b, "_full_load_table"),
                patch.object(b, "_snapshot") as mock_snap,
            ):
                b.daily_update(
                    {"format_staples": ([_Card(id="X__modern", name="X")], [])}
                )
            history_tables = [
                c.kwargs["history_table"] for c in mock_snap.call_args_list
            ]
            assert "bronze_format_staples_history" in history_tables

    def test_storage_write_error_does_not_stop_other_sources(self):
        with _bronze() as b:
            with (
                patch.object(
                    b, "_incremental_load", side_effect=StorageWriteError("fail")
                ),
                patch.object(
                    b, "_full_load_table", side_effect=StorageWriteError("fail")
                ),
                patch.object(b, "_snapshot"),
            ):
                b.daily_update({"scryfall": ([], []), "mtgjson_cards": ([], [])})

    def test_empty_results_runs_without_error(self):
        with _bronze() as b:
            with (
                patch.object(b, "_incremental_load"),
                patch.object(b, "_full_load_table"),
                patch.object(b, "_snapshot"),
            ):
                b.daily_update({})


# ---------------------------------------------------------------------------
# BronzeStorage._snapshot_scryfall_prices
# ---------------------------------------------------------------------------


class _ScryfallCard(BaseModel):
    id: str
    prices: dict | None = None


class TestSnapshotScryfallPrices:
    HISTORY_TABLE = "bronze_scryfall_prices_history"

    def test_creates_history_table_on_first_call(self):
        with _bronze() as b:
            record = _ScryfallCard(
                id="s1", prices={"eur": "3.20", "eur_foil": None, "usd": None, "usd_foil": None}
            )
            b._snapshot_scryfall_prices([record])
            row = b._con.execute(
                f"SELECT count(*) FROM {self.HISTORY_TABLE}"
            ).fetchone()
        assert row is not None and row[0] == 1

    def test_extracts_eur_as_float(self):
        with _bronze() as b:
            record = _ScryfallCard(
                id="s1",
                prices={"eur": "3.20", "eur_foil": "8.50", "usd": "3.50", "usd_foil": "9.00"},
            )
            b._snapshot_scryfall_prices([record])
            row = b._con.execute(
                f"SELECT eur, eur_foil, usd, usd_foil FROM {self.HISTORY_TABLE}"
            ).fetchone()
        assert row is not None
        assert row[0] == pytest.approx(3.20)
        assert row[1] == pytest.approx(8.50)
        assert row[2] == pytest.approx(3.50)
        assert row[3] == pytest.approx(9.00)

    def test_null_price_fields_produce_null_columns(self):
        with _bronze() as b:
            record = _ScryfallCard(
                id="s1",
                prices={"eur": "3.20", "eur_foil": None, "usd": None, "usd_foil": None},
            )
            b._snapshot_scryfall_prices([record])
            row = b._con.execute(
                f"SELECT eur, eur_foil FROM {self.HISTORY_TABLE}"
            ).fetchone()
        assert row is not None
        assert row[0] == pytest.approx(3.20)
        assert row[1] is None

    def test_tix_key_is_ignored(self):
        with _bronze() as b:
            record = _ScryfallCard(
                id="s1",
                prices={"eur": "3.20", "tix": "0.05"},
            )
            b._snapshot_scryfall_prices([record])
            cols = {r[0] for r in b._con.execute(
                f"DESCRIBE {self.HISTORY_TABLE}"
            ).fetchall()}
        assert "tix" not in cols
        assert "eur" in cols

    def test_none_prices_dict_produces_all_null_columns(self):
        with _bronze() as b:
            record = _ScryfallCard(id="s1", prices=None)
            b._snapshot_scryfall_prices([record])
            row = b._con.execute(
                f"SELECT eur FROM {self.HISTORY_TABLE}"
            ).fetchone()
        assert row is not None and row[0] is None

    def test_idempotent_on_duplicate_id_date(self):
        with _bronze() as b:
            record = _ScryfallCard(
                id="s1", prices={"eur": "3.20"}
            )
            b._snapshot_scryfall_prices([record])
            b._snapshot_scryfall_prices([record])
            row = b._con.execute(
                f"SELECT count(*) FROM {self.HISTORY_TABLE}"
            ).fetchone()
        assert row is not None and row[0] == 1

    def test_skips_when_records_empty(self):
        with _bronze() as b:
            b._snapshot_scryfall_prices([])
            row = b._con.execute(
                "SELECT count(*) FROM information_schema.tables"
                f" WHERE table_name = '{self.HISTORY_TABLE}'"
            ).fetchone()
        assert row is not None and row[0] == 0


# ---------------------------------------------------------------------------
# BronzeStorage._snapshot_mtgjson_prices
# ---------------------------------------------------------------------------


class _MtgjsonPrices(BaseModel):
    uuid: str
    paper: dict | None = None


class TestSnapshotMtgjsonPrices:
    HISTORY_TABLE = "bronze_mtgjson_prices_history"

    def test_table_has_eav_schema(self):
        with _bronze() as b:
            record = _MtgjsonPrices(
                uuid="u1",
                paper={"cardmarket": {"retail": {"normal": {"2026-06-24": 3.20}}}},
            )
            b._snapshot_mtgjson_prices([record])
            cols = {r[0] for r in b._con.execute(
                f"DESCRIBE {self.HISTORY_TABLE}"
            ).fetchall()}
        assert cols == {"uuid", "snapshot_date", "retailer", "tx_type", "finish", "price"}

    def test_emits_one_row_per_price_point(self):
        with _bronze() as b:
            record = _MtgjsonPrices(
                uuid="u1",
                paper={
                    "cardmarket": {
                        "retail":  {"normal": {"2026-06-24": 3.20}, "foil": {"2026-06-24": 8.50}},
                        "buylist": {"normal": {"2026-06-24": 1.80}},
                    },
                    "tcgplayer": {
                        "retail": {"normal": {"2026-06-24": 3.50}},
                    },
                },
            )
            b._snapshot_mtgjson_prices([record])
            count = b._con.execute(
                f"SELECT count(*) FROM {self.HISTORY_TABLE}"
            ).fetchone()[0]
        assert count == 4

    def test_captures_unlisted_retailer(self):
        with _bronze() as b:
            record = _MtgjsonPrices(
                uuid="u1",
                paper={
                    "cardmarket":  {"retail": {"normal": {"2026-06-24": 3.20}}},
                    "cardkingdom": {"retail": {"normal": {"2026-06-24": 4.00}}},
                },
            )
            b._snapshot_mtgjson_prices([record])
            retailers = {r[0] for r in b._con.execute(
                f"SELECT DISTINCT retailer FROM {self.HISTORY_TABLE}"
            ).fetchall()}
        assert "cardkingdom" in retailers

    def test_null_paper_produces_no_rows(self):
        with _bronze() as b:
            b._snapshot_mtgjson_prices([_MtgjsonPrices(uuid="u1", paper=None)])
            count = b._con.execute(
                "SELECT count(*) FROM information_schema.tables"
                f" WHERE table_name = '{self.HISTORY_TABLE}'"
            ).fetchone()[0]
        assert count == 0

    def test_idempotent_on_duplicate(self):
        with _bronze() as b:
            record = _MtgjsonPrices(
                uuid="u1",
                paper={"cardmarket": {"retail": {"normal": {"2026-06-24": 3.20}}}},
            )
            b._snapshot_mtgjson_prices([record])
            b._snapshot_mtgjson_prices([record])
            count = b._con.execute(
                f"SELECT count(*) FROM {self.HISTORY_TABLE}"
            ).fetchone()[0]
        assert count == 1

    def test_uses_today_as_snapshot_date(self):
        from datetime import date as date_cls
        with _bronze() as b:
            record = _MtgjsonPrices(
                uuid="u1",
                paper={"cardmarket": {"retail": {"normal": {"2026-06-24": 3.20}}}},
            )
            with patch("src.data.cards.storage.bronze.storage.date") as mock_date:
                mock_date.today.return_value = date_cls.fromisoformat("2026-06-24")
                b._snapshot_mtgjson_prices([record])
            snap = b._con.execute(
                f"SELECT snapshot_date FROM {self.HISTORY_TABLE}"
            ).fetchone()[0]
        assert str(snap) == "2026-06-24"

    def test_skips_when_records_empty(self):
        with _bronze() as b:
            b._snapshot_mtgjson_prices([])
            count = b._con.execute(
                "SELECT count(*) FROM information_schema.tables"
                f" WHERE table_name = '{self.HISTORY_TABLE}'"
            ).fetchone()[0]
        assert count == 0
