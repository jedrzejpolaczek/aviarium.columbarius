import json
from datetime import date
from unittest.mock import MagicMock, patch

import duckdb
import pytest
from pydantic import BaseModel

from src.data.cards.storage.bronze import BronzeStorage
from src.data.cards.storage.errors import StorageConnectionError, StorageWriteError


# ---------------------------------------------------------------------------
# Minimal Pydantic models used to exercise storage operations
# ---------------------------------------------------------------------------


class _Card(BaseModel):
    id: str
    name: str


class _CardWithNested(BaseModel):
    id: str
    name: str
    tags: list[str] | None = None
    meta: dict[str, str] | None = None


class _PricedCard(BaseModel):
    id: str
    name: str
    price: float
    score: int | None = None


class _PriceRecord(BaseModel):
    uuid: str
    paper: dict | None = None
    mtgo: dict | None = None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def storage():
    s = BronzeStorage(":memory:")
    yield s
    s.close()


def _count(storage: BronzeStorage, table: str) -> int:
    row = storage._con.execute(f"SELECT count(*) FROM {table}").fetchone()
    assert row is not None
    return int(row[0])


def _fetch(storage: BronzeStorage, table: str) -> list[tuple]:
    return storage._con.execute(f"SELECT * FROM {table}").fetchall()


def _columns(storage: BronzeStorage, table: str) -> set[str]:
    return {
        row[0]
        for row in storage._con.execute(
            f"SELECT column_name FROM information_schema.columns WHERE table_name = '{table}'"
        ).fetchall()
    }


# ---------------------------------------------------------------------------
# BronzeStorage.__init__ / context manager
# ---------------------------------------------------------------------------


class TestBronzeStorageInit:
    def test_memory_db_opens_successfully(self):
        s = BronzeStorage(":memory:")
        row = s._con.execute("SELECT 42").fetchone()
        assert row is not None
        result = row[0]
        s.close()
        assert result == 42

    def test_file_db_creates_parent_directory(self, tmp_path):
        db_path = tmp_path / "nested" / "subdir" / "cards.duckdb"
        s = BronzeStorage(str(db_path))
        s.close()
        assert db_path.exists()

    def test_context_manager_returns_self(self):
        with BronzeStorage(":memory:") as s:
            assert isinstance(s, BronzeStorage)

    def test_context_manager_closes_connection_on_exit(self):
        with BronzeStorage(":memory:") as s:
            con = s._con
        with pytest.raises(duckdb.Error):
            con.execute("SELECT 1")

    def test_connection_error_raises_storage_connection_error(self):
        with patch(
            "src.data.cards.storage.base.storage.duckdb.connect",
            side_effect=duckdb.Error("locked"),
        ):
            with pytest.raises(StorageConnectionError, match="Cannot open DuckDB"):
                BronzeStorage(":memory:")


# ---------------------------------------------------------------------------
# _full_load_table
# ---------------------------------------------------------------------------


class TestFullLoadTable:
    def test_empty_records_is_noop(self, storage):
        storage._full_load_table([], "test_table")
        tables = storage._con.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_name = 'test_table'"
        ).fetchall()
        assert tables == []

    def test_creates_table_with_correct_row_count(self, storage):
        records = [_Card(id="1", name="Alpha"), _Card(id="2", name="Beta")]
        storage._full_load_table(records, "test_table")
        assert _count(storage, "test_table") == 2

    def test_data_is_queryable(self, storage):
        storage._full_load_table([_Card(id="1", name="Alpha")], "test_table")
        row = storage._con.execute(
            "SELECT id, name FROM test_table WHERE id = '1'"
        ).fetchone()
        assert row == ("1", "Alpha")

    def test_second_call_replaces_table(self, storage):
        storage._full_load_table([_Card(id="1", name="Old")], "test_table")
        storage._full_load_table(
            [_Card(id="2", name="New1"), _Card(id="3", name="New2")], "test_table"
        )
        assert _count(storage, "test_table") == 2
        ids = {
            row[0]
            for row in storage._con.execute("SELECT id FROM test_table").fetchall()
        }
        assert ids == {"2", "3"}

    def test_list_values_serialized_as_json_string(self, storage):
        records = [
            _CardWithNested(id="1", name="A", tags=None),
            _CardWithNested(id="2", name="B", tags=["x", "y"]),
        ]
        storage._full_load_table(records, "test_table")
        row = storage._con.execute(
            "SELECT tags FROM test_table WHERE id = '2'"
        ).fetchone()
        assert json.loads(row[0]) == ["x", "y"]

    def test_dict_values_serialized_as_json_string(self, storage):
        records = [
            _CardWithNested(id="1", name="A", meta=None),
            _CardWithNested(id="2", name="B", meta={"key": "val"}),
        ]
        storage._full_load_table(records, "test_table")
        row = storage._con.execute(
            "SELECT meta FROM test_table WHERE id = '2'"
        ).fetchone()
        assert json.loads(row[0]) == {"key": "val"}

    def test_none_optional_fields_stored_as_null(self, storage):
        storage._full_load_table(
            [_CardWithNested(id="1", name="A", tags=None, meta=None)], "test_table"
        )
        row = storage._con.execute("SELECT tags, meta FROM test_table").fetchone()
        assert row[0] is None
        assert row[1] is None

    def test_staging_view_cleaned_up_after_write(self, storage):
        storage._full_load_table([_Card(id="1", name="A")], "test_table")
        views = storage._con.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_name = '_save_staging'"
        ).fetchall()
        assert views == []


# ---------------------------------------------------------------------------
# _incremental_load
# ---------------------------------------------------------------------------


class TestIncrementalLoad:
    def test_empty_records_is_noop(self, storage):
        storage._incremental_load([], "test_table", key_column="id")
        tables = storage._con.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_name = 'test_table'"
        ).fetchall()
        assert tables == []

    def test_creates_table_on_first_call(self, storage):
        storage._incremental_load(
            [_Card(id="1", name="A")], "test_table", key_column="id"
        )
        assert _count(storage, "test_table") == 1

    def test_inserts_rows_on_first_call(self, storage):
        records = [_Card(id="1", name="A"), _Card(id="2", name="B")]
        storage._incremental_load(records, "test_table", key_column="id")
        ids = {
            row[0]
            for row in storage._con.execute("SELECT id FROM test_table").fetchall()
        }
        assert ids == {"1", "2"}

    def test_upsert_replaces_matching_key(self, storage):
        storage._incremental_load(
            [_Card(id="1", name="Old")], "test_table", key_column="id"
        )
        storage._incremental_load(
            [_Card(id="1", name="Updated")], "test_table", key_column="id"
        )

        assert _count(storage, "test_table") == 1
        row = storage._con.execute(
            "SELECT name FROM test_table WHERE id = '1'"
        ).fetchone()
        assert row[0] == "Updated"

    def test_upsert_preserves_non_matching_rows(self, storage):
        storage._incremental_load(
            [_Card(id="1", name="Keep"), _Card(id="2", name="Also keep")],
            "test_table",
            key_column="id",
        )
        storage._incremental_load(
            [_Card(id="2", name="Updated")], "test_table", key_column="id"
        )

        assert _count(storage, "test_table") == 2
        row = storage._con.execute(
            "SELECT name FROM test_table WHERE id = '1'"
        ).fetchone()
        assert row[0] == "Keep"

    def test_new_key_appended_when_table_exists(self, storage):
        storage._incremental_load(
            [_Card(id="1", name="A")], "test_table", key_column="id"
        )
        storage._incremental_load(
            [_Card(id="2", name="B")], "test_table", key_column="id"
        )
        assert _count(storage, "test_table") == 2

    def test_staging_view_cleaned_up_after_write(self, storage):
        storage._incremental_load(
            [_Card(id="1", name="A")], "test_table", key_column="id"
        )
        views = storage._con.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_name = '_incremental_staging'"
        ).fetchall()
        assert views == []


# ---------------------------------------------------------------------------
# _snapshot
# ---------------------------------------------------------------------------


class TestSnapshot:
    def test_empty_records_is_noop(self, storage):
        storage._snapshot([], key_column="id", history_table="hist")
        tables = storage._con.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_name = 'hist'"
        ).fetchall()
        assert tables == []

    def test_creates_history_table_on_first_call(self, storage):
        storage._snapshot(
            [_Card(id="1", name="A")], key_column="id", history_table="hist"
        )
        assert _count(storage, "hist") == 1

    def test_snapshot_row_contains_key_and_date(self, storage):
        with patch("src.data.cards.storage.bronze.storage.date") as mock_date:
            mock_date.today.return_value = date(2026, 1, 1)
            storage._snapshot(
                [_Card(id="42", name="A")], key_column="id", history_table="hist"
            )

        row = storage._con.execute("SELECT id, snapshot_date FROM hist").fetchone()
        assert row[0] == "42"
        assert row[1] == "2026-01-01"

    def test_idempotent_on_same_day(self, storage):
        records = [_Card(id="1", name="A")]
        with patch("src.data.cards.storage.bronze.storage.date") as mock_date:
            mock_date.today.return_value = date(2026, 1, 1)
            storage._snapshot(records, key_column="id", history_table="hist")
            storage._snapshot(records, key_column="id", history_table="hist")

        assert _count(storage, "hist") == 1

    def test_different_date_adds_new_row(self, storage):
        records = [_Card(id="1", name="A")]
        with patch("src.data.cards.storage.bronze.storage.date") as mock_date:
            mock_date.today.return_value = date(2026, 1, 1)
            storage._snapshot(records, key_column="id", history_table="hist")

            mock_date.today.return_value = date(2026, 1, 2)
            storage._snapshot(records, key_column="id", history_table="hist")

        assert _count(storage, "hist") == 2

    def test_fields_filter_limits_snapshot_columns(self, storage):
        records = [_PricedCard(id="1", name="A", price=1.5, score=10)]
        storage._snapshot(
            records, key_column="id", history_table="hist", fields=["price"]
        )
        cols = _columns(storage, "hist")
        assert "price" in cols
        assert "score" not in cols
        assert "name" not in cols

    def test_fields_none_snapshots_all_model_fields(self, storage):
        records = [_PricedCard(id="1", name="A", price=1.5, score=10)]
        storage._snapshot(records, key_column="id", history_table="hist", fields=None)
        cols = _columns(storage, "hist")
        assert {"id", "name", "price", "score", "snapshot_date"}.issubset(cols)

    def test_key_and_snapshot_date_always_present(self, storage):
        records = [_PricedCard(id="7", name="A", price=0.5)]
        storage._snapshot(
            records, key_column="id", history_table="hist", fields=["price"]
        )
        cols = _columns(storage, "hist")
        assert "id" in cols
        assert "snapshot_date" in cols

    def test_staging_view_cleaned_up_after_write(self, storage):
        storage._snapshot(
            [_Card(id="1", name="A")], key_column="id", history_table="hist"
        )
        views = storage._con.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_name = '_snapshot_staging'"
        ).fetchall()
        assert views == []

    def test_multiple_records_all_snapshotted(self, storage):
        records = [
            _Card(id="1", name="A"),
            _Card(id="2", name="B"),
            _Card(id="3", name="C"),
        ]
        storage._snapshot(records, key_column="id", history_table="hist")
        assert _count(storage, "hist") == 3


# ---------------------------------------------------------------------------
# populate
# ---------------------------------------------------------------------------


class TestPopulate:
    def test_calls_full_load_for_all_configured_sources(self, storage):
        with (
            patch.object(storage, "_full_load_table") as mock_full,
            patch.object(storage, "_snapshot"),
        ):
            storage.populate(
                {
                    "scryfall": ([MagicMock()], []),
                    "mtgjson_cards": ([MagicMock()], []),
                    "mtgjson_prices": ([MagicMock()], []),
                }
            )
        # scryfall + mtgjson_cards + tournament_results (mtgjson_prices has no table)
        assert mock_full.call_count == 3

    def test_calls_snapshot_for_configured_sources(self, storage):
        with (
            patch.object(storage, "_full_load_table"),
            patch.object(storage, "_snapshot") as mock_snap,
            patch.object(storage, "_snapshot_scryfall_prices"),
            patch.object(storage, "seed_historical_prices"),
        ):
            storage.populate(
                {
                    "scryfall": ([MagicMock()], []),
                    "mtgjson_cards": ([], []),
                    "mtgjson_prices": ([MagicMock()], []),
                }
            )
        # scryfall meta (1) + format_staples (1) = 2
        # mtgjson_prices snapshots are now handled by _snapshot_mtgjson_prices (not _snapshot)
        assert mock_snap.call_count == 2

    def test_missing_source_in_results_treated_as_empty(self, storage):
        with (
            patch.object(storage, "_full_load_table") as mock_full,
            patch.object(storage, "_snapshot"),
        ):
            storage.populate({})  # no results at all

        # _full_load_table called with empty list for each source → all skipped
        for c in mock_full.call_args_list:
            assert c.args[0] == []

    def test_write_error_skips_source_without_raising(self, storage):
        with patch.object(
            storage, "_full_load_table", side_effect=StorageWriteError("boom")
        ):
            storage.populate({"scryfall": ([MagicMock()], [])})  # must not raise

    def test_error_in_one_source_does_not_block_others(self, storage):
        call_log = []

        def fake_full(records, table_name):
            if table_name == "bronze_scryfall_cards":
                raise StorageWriteError("scryfall failed")
            call_log.append(table_name)

        with (
            patch.object(storage, "_full_load_table", side_effect=fake_full),
            patch.object(storage, "_snapshot"),
            patch.object(storage, "seed_historical_prices"),
        ):
            storage.populate(
                {
                    "scryfall": ([MagicMock()], []),
                    "mtgjson_cards": ([MagicMock()], []),
                }
            )

        assert "bronze_mtgjson_cards" in call_log

    def test_calls_seed_historical_prices_with_mtgjson_prices(self, storage):
        price_records = [MagicMock()]
        with (
            patch.object(storage, "_full_load_table"),
            patch.object(storage, "_snapshot"),
            patch.object(storage, "seed_historical_prices") as mock_seed,
        ):
            storage.populate({"mtgjson_prices": (price_records, [])})

        mock_seed.assert_called_once_with(price_records)

    def test_seed_error_skips_without_raising(self, storage):
        with (
            patch.object(storage, "_full_load_table"),
            patch.object(storage, "_snapshot"),
            patch.object(
                storage,
                "seed_historical_prices",
                side_effect=StorageWriteError("seed boom"),
            ),
        ):
            storage.populate({"mtgjson_prices": ([MagicMock()], [])})  # must not raise


# ---------------------------------------------------------------------------
# daily_update
# ---------------------------------------------------------------------------


class TestDailyUpdate:
    def test_incremental_source_calls_incremental_load(self, storage):
        with (
            patch.object(storage, "_incremental_load") as mock_inc,
            patch.object(storage, "_full_load_table"),
            patch.object(storage, "_snapshot"),
        ):
            storage.daily_update(
                {
                    "scryfall": ([MagicMock()], []),
                    "mtgjson_cards": ([MagicMock()], []),
                }
            )

        # scryfall, mtgjson_cards, and tournament_results are incremental=True
        assert mock_inc.call_count == 3
        inc_tables = {c.args[1] for c in mock_inc.call_args_list}
        assert "bronze_scryfall_cards" in inc_tables
        assert "bronze_mtgjson_cards" in inc_tables
        assert "bronze_tournament_results" in inc_tables

    def test_daily_update_calls_snapshot(self, storage):
        with (
            patch.object(storage, "_incremental_load"),
            patch.object(storage, "_snapshot") as mock_snap,
            patch.object(storage, "_snapshot_scryfall_prices") as mock_scryfall_prices,
            patch.object(storage, "_snapshot_mtgjson_prices") as mock_mtgjson_prices,
        ):
            storage.daily_update(
                {
                    "scryfall": ([MagicMock()], []),
                    "mtgjson_prices": ([MagicMock()], []),
                    "mtgjson_cards": ([], []),
                }
            )
        # _process_sources: scryfall meta (1) + format_staples (1) = 2
        assert mock_snap.call_count == 2
        mock_scryfall_prices.assert_called_once()
        mock_mtgjson_prices.assert_called_once()

    def test_write_error_skips_source_without_raising(self, storage):
        with patch.object(
            storage, "_incremental_load", side_effect=StorageWriteError("boom")
        ):
            storage.daily_update({"scryfall": ([MagicMock()], [])})  # must not raise

    def test_error_in_one_source_does_not_block_others(self, storage):
        call_log = []

        def fake_inc(records, table_name, key_column):
            if table_name == "bronze_scryfall_cards":
                raise StorageWriteError("scryfall failed")
            call_log.append(table_name)

        with (
            patch.object(storage, "_incremental_load", side_effect=fake_inc),
            patch.object(storage, "_snapshot"),
        ):
            storage.daily_update(
                {
                    "scryfall": ([MagicMock()], []),
                    "mtgjson_cards": ([MagicMock()], []),
                    "mtgjson_prices": ([], []),
                }
            )

        assert "bronze_mtgjson_cards" in call_log


# ---------------------------------------------------------------------------
# seed_historical_prices
# ---------------------------------------------------------------------------

_PAPER_PRICES = {
    "cardmarket": {
        "retail": {
            "normal": {"2026-04-01": 1.0, "2026-04-02": 1.1},
            "foil":   {"2026-04-01": 2.0},
        },
    },
    "cardkingdom": {
        "retail": {"normal": {"2026-04-01": 3.5}},
    },
}


class TestSeedHistoricalPrices:
    HISTORY_TABLE = "bronze_mtgjson_prices_history"

    def test_empty_records_is_noop(self, storage):
        storage.seed_historical_prices([])
        tables = storage._con.execute(
            f"SELECT table_name FROM information_schema.tables "
            f"WHERE table_name = '{self.HISTORY_TABLE}'"
        ).fetchall()
        assert tables == []

    def test_creates_history_table_on_first_call(self, storage):
        record = _PriceRecord(uuid="uuid-1", paper=_PAPER_PRICES)
        storage.seed_historical_prices([record])
        assert _count(storage, self.HISTORY_TABLE) > 0

    def test_row_count_matches_eav_leaf_entries(self, storage):
        record = _PriceRecord(uuid="uuid-1", paper=_PAPER_PRICES)
        storage.seed_historical_prices([record])
        # 4 EAV leaf entries: cm/retail/normal/04-01, cm/retail/normal/04-02,
        # cm/retail/foil/04-01, ck/retail/normal/04-01
        assert _count(storage, self.HISTORY_TABLE) == 4

    def test_row_contains_uuid_and_snapshot_date(self, storage):
        record = _PriceRecord(uuid="uuid-1", paper=_PAPER_PRICES)
        storage.seed_historical_prices([record])
        dates = {
            row[1]
            for row in storage._con.execute(
                f"SELECT uuid, snapshot_date FROM {self.HISTORY_TABLE}"
            ).fetchall()
        }
        assert dates == {"2026-04-01", "2026-04-02"}

    def test_idempotent_second_call_does_not_duplicate(self, storage):
        record = _PriceRecord(uuid="uuid-1", paper=_PAPER_PRICES)
        storage.seed_historical_prices([record])
        storage.seed_historical_prices([record])
        assert _count(storage, self.HISTORY_TABLE) == 4

    def test_multiple_cards_all_seeded(self, storage):
        r1 = _PriceRecord(uuid="uuid-1", paper=_PAPER_PRICES)
        r2 = _PriceRecord(uuid="uuid-2", paper=_PAPER_PRICES)
        storage.seed_historical_prices([r1, r2])
        assert _count(storage, self.HISTORY_TABLE) == 8

    def test_record_with_no_dates_produces_no_rows(self, storage):
        record = _PriceRecord(uuid="uuid-1", paper=None, mtgo=None)
        storage.seed_historical_prices([record])
        tables = storage._con.execute(
            f"SELECT table_name FROM information_schema.tables "
            f"WHERE table_name = '{self.HISTORY_TABLE}'"
        ).fetchall()
        assert tables == []

    def test_mtgo_prices_not_collected(self, storage):
        record = _PriceRecord(
            uuid="uuid-1",
            paper=None,
            mtgo={"cardhoarder": {"retail": {"normal": {"2026-03-15": 0.5}}}},
        )
        storage.seed_historical_prices([record])
        tables = storage._con.execute(
            f"SELECT table_name FROM information_schema.tables"
            f" WHERE table_name = '{self.HISTORY_TABLE}'"
        ).fetchall()
        assert tables == []

    def test_eav_schema_has_correct_columns(self, storage):
        record = _PriceRecord(uuid="uuid-1", paper=_PAPER_PRICES)
        storage.seed_historical_prices([record])
        cols = {r[0] for r in storage._con.execute(
            f"DESCRIBE {self.HISTORY_TABLE}"
        ).fetchall()}
        assert cols == {"uuid", "snapshot_date", "retailer", "tx_type", "finish", "price"}

    def test_eav_row_has_correct_values(self, storage):
        record = _PriceRecord(uuid="uuid-1", paper=_PAPER_PRICES)
        storage.seed_historical_prices([record])
        row = storage._con.execute(
            f"SELECT uuid, snapshot_date, retailer, tx_type, finish, price"
            f" FROM {self.HISTORY_TABLE}"
            f" WHERE retailer='cardmarket' AND tx_type='retail'"
            f"   AND finish='normal' AND snapshot_date='2026-04-01'"
        ).fetchone()
        assert row is not None
        assert row[0] == "uuid-1"
        assert str(row[1]) == "2026-04-01"
        assert row[2] == "cardmarket"
        assert row[3] == "retail"
        assert row[4] == "normal"
        assert row[5] == pytest.approx(1.0)

    def test_captures_cardkingdom(self, storage):
        record = _PriceRecord(uuid="uuid-1", paper=_PAPER_PRICES)
        storage.seed_historical_prices([record])
        retailers = {r[0] for r in storage._con.execute(
            f"SELECT DISTINCT retailer FROM {self.HISTORY_TABLE}"
        ).fetchall()}
        assert "cardkingdom" in retailers

    def test_duckdb_error_raises_storage_write_error(self, storage):
        record = _PriceRecord(uuid="uuid-1", paper=_PAPER_PRICES)
        mock_con = MagicMock()
        mock_con.execute.side_effect = duckdb.Error("disk full")
        storage._con = mock_con
        storage._writer._con = mock_con
        with pytest.raises(StorageWriteError, match="Failed to append into"):
            storage.seed_historical_prices([record])
