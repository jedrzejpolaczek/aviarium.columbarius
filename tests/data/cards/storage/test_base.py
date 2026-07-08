"""Tests for BaseStorage, TransformStorage, and DuckDBWriter base classes."""

from unittest.mock import MagicMock, patch

import duckdb
import pandas as pd
import pytest

from src.data.cards.storage.base.storage import get_tables
from src.data.cards.storage.base.writers import DuckDBWriter
from src.data.cards.storage.base.transformer import TransformStorage
from src.data.cards.storage.base.storage import BaseStorage
from src.data.cards.storage.errors import StorageConnectionError, StorageWriteError


# ---------------------------------------------------------------------------
# Concrete minimal implementations for testing abstract classes
# ---------------------------------------------------------------------------


class ConcreteStorage(BaseStorage):
    """Minimal concrete subclass of BaseStorage for testing."""

    def __init__(self, db_path: str = ":memory:") -> None:
        self._con = self._open_connection(db_path, read_only=False)

    def close(self) -> None:
        self._con.close()


class ConcreteTransformStorage(TransformStorage):
    """Minimal concrete subclass of TransformStorage for testing."""

    def __init__(self) -> None:
        self._pipeline_calls: list[dict] = []

    def close(self) -> None:
        pass

    def _pipeline(self, update: bool, report_path: str = "") -> None:
        self._pipeline_calls.append({"update": update, "report_path": report_path})


# ---------------------------------------------------------------------------
# BaseStorage._open_connection
# ---------------------------------------------------------------------------


class TestOpenConnection:
    def test_returns_duckdb_connection_for_memory(self):
        con = BaseStorage._open_connection(":memory:", read_only=False)
        assert con is not None
        con.close()

    def test_returns_duckdb_connection_for_file(self, tmp_path):
        db_path = str(tmp_path / "test.duckdb")
        con = BaseStorage._open_connection(db_path, read_only=False)
        assert con is not None
        con.close()

    def test_creates_parent_directory_if_missing(self, tmp_path):
        db_path = str(tmp_path / "nested" / "dir" / "test.duckdb")
        con = BaseStorage._open_connection(db_path, read_only=False)
        con.close()
        assert (tmp_path / "nested" / "dir").exists()

    def test_read_only_flag_propagated(self, tmp_path):
        db_path = str(tmp_path / "test.duckdb")
        # Create the file first
        duckdb.connect(db_path).close()
        con = BaseStorage._open_connection(db_path, read_only=True)
        con.close()

    def test_raises_storage_connection_error_on_bad_path(self, tmp_path):
        bad_path = str(tmp_path / "does_not_exist" / "test.duckdb")
        # Monkeypatch mkdir to prevent creation so duckdb.connect fails
        with patch("src.data.db.Path") as MockPath:
            mock_instance = MagicMock()
            mock_instance.parent.mkdir.side_effect = PermissionError("denied")
            MockPath.return_value = mock_instance
            with pytest.raises((StorageConnectionError, PermissionError)):
                BaseStorage._open_connection(bad_path, read_only=False)

    def test_raises_storage_connection_error_when_duckdb_fails(self):
        with patch("src.data.db.duckdb.connect") as mock_connect:
            mock_connect.side_effect = duckdb.Error("connection failed")
            with pytest.raises(StorageConnectionError, match="Cannot open DuckDB"):
                BaseStorage._open_connection(":memory:", read_only=False)


# ---------------------------------------------------------------------------
# BaseStorage context manager
# ---------------------------------------------------------------------------


class TestContextManager:
    def test_enter_returns_self(self):
        storage = ConcreteStorage()
        with storage as s:
            assert s is storage

    def test_exit_calls_close(self):
        with patch.object(ConcreteStorage, "close") as mock_close:
            with ConcreteStorage():
                pass
            mock_close.assert_called_once()

    def test_exit_calls_close_on_exception(self):
        with patch.object(ConcreteStorage, "close") as mock_close:
            with pytest.raises(ValueError):
                with ConcreteStorage():
                    raise ValueError("test error")
            mock_close.assert_called_once()


# ---------------------------------------------------------------------------
# TransformStorage.populate / update
# ---------------------------------------------------------------------------


class TestTransformStoragePipeline:
    def test_populate_calls_pipeline_with_update_false(self):
        storage = ConcreteTransformStorage()
        storage.populate()
        assert len(storage._pipeline_calls) == 1
        assert storage._pipeline_calls[0]["update"] is False

    def test_update_calls_pipeline_with_update_true(self):
        storage = ConcreteTransformStorage()
        storage.update()
        assert len(storage._pipeline_calls) == 1
        assert storage._pipeline_calls[0]["update"] is True

    def test_populate_then_update_calls_pipeline_twice(self):
        storage = ConcreteTransformStorage()
        storage.populate()
        storage.update()
        assert len(storage._pipeline_calls) == 2
        assert storage._pipeline_calls[0]["update"] is False
        assert storage._pipeline_calls[1]["update"] is True

    def test_populate_emits_info_log(self, caplog):
        import logging

        storage = ConcreteTransformStorage()
        with caplog.at_level(logging.INFO):
            storage.populate()
        assert any(r.levelno == logging.INFO for r in caplog.records)

    def test_update_emits_info_log(self, caplog):
        import logging

        storage = ConcreteTransformStorage()
        with caplog.at_level(logging.INFO):
            storage.update()
        assert any(r.levelno == logging.INFO for r in caplog.records)


# ---------------------------------------------------------------------------
# get_tables
# ---------------------------------------------------------------------------


class TestGetTables:
    def test_returns_table_names(self, memory_con):
        memory_con.execute("CREATE TABLE test_table AS SELECT 1 AS col")
        tables = get_tables(memory_con)
        assert "test_table" in tables

    def test_returns_empty_set_for_empty_database(self, memory_con):
        tables = get_tables(memory_con)
        assert tables == set()

    def test_returns_multiple_tables(self, memory_con):
        memory_con.execute("CREATE TABLE table1 AS SELECT 1 AS col")
        memory_con.execute("CREATE TABLE table2 AS SELECT 2 AS col")
        tables = get_tables(memory_con)
        assert "table1" in tables
        assert "table2" in tables


# ---------------------------------------------------------------------------
# DuckDBWriter
# ---------------------------------------------------------------------------


@pytest.fixture
def mem_con():
    """In-memory DuckDB connection, closed after each test."""
    # Pre-existing local fixture, functionally identical to conftest.py's memory_con;
    # not consolidated in this pilot — see Task 13 in the maintainability remediation plan.
    con = duckdb.connect(":memory:")
    yield con
    con.close()


class TestDuckDBWriter:
    # ------------------------------------------------------------------ full_load

    def test_full_load_creates_table(self, mem_con):
        DuckDBWriter(mem_con).full_load(pd.DataFrame({"a": [1, 2, 3]}), "tbl")
        row = mem_con.execute("SELECT count(*) FROM tbl").fetchone()
        assert row is not None and row[0] == 3

    def test_full_load_replaces_table(self, mem_con):
        w = DuckDBWriter(mem_con)
        w.full_load(pd.DataFrame({"x": [1, 2, 3]}), "tbl")
        w.full_load(pd.DataFrame({"x": [99]}), "tbl")
        count = mem_con.execute("SELECT count(*) FROM tbl").fetchone()
        val = mem_con.execute("SELECT x FROM tbl").fetchone()
        assert count is not None and count[0] == 1
        assert val is not None and val[0] == 99

    def test_full_load_empty_skips(self, mem_con):
        DuckDBWriter(mem_con).full_load(pd.DataFrame(), "tbl")
        tables = {r[0] for r in mem_con.execute("SHOW TABLES").fetchall()}
        assert "tbl" not in tables

    def test_full_load_raises_storage_write_error(self):
        mock_con = MagicMock()
        mock_con.execute.side_effect = duckdb.Error("boom")
        with pytest.raises(StorageWriteError):
            DuckDBWriter(mock_con).full_load(pd.DataFrame({"a": [1]}), "tbl")

    # ------------------------------------------------------------------ upsert

    def test_upsert_creates_on_first_call(self, mem_con):
        DuckDBWriter(mem_con).upsert(pd.DataFrame({"id": ["a"], "v": [1]}), "tbl", "id")
        count = mem_con.execute("SELECT count(*) FROM tbl").fetchone()
        assert count is not None and count[0] == 1

    def test_upsert_replaces_keyed_rows(self, mem_con):
        w = DuckDBWriter(mem_con)
        w.upsert(pd.DataFrame({"id": ["a"], "v": [1]}), "tbl", "id")
        w.upsert(pd.DataFrame({"id": ["a"], "v": [99]}), "tbl", "id")
        val = mem_con.execute("SELECT v FROM tbl WHERE id='a'").fetchone()
        assert val is not None and val[0] == 99

    def test_upsert_leaves_other_rows(self, mem_con):
        w = DuckDBWriter(mem_con)
        w.upsert(pd.DataFrame({"id": ["a", "b"], "v": [1, 2]}), "tbl", "id")
        w.upsert(pd.DataFrame({"id": ["a"], "v": [10]}), "tbl", "id")
        val = mem_con.execute("SELECT v FROM tbl WHERE id='b'").fetchone()
        assert val is not None and val[0] == 2

    def test_upsert_skips_empty_dataframe(self, mem_con):
        DuckDBWriter(mem_con).upsert(
            pd.DataFrame({"id": pd.Series([], dtype=str)}), "tbl", "id"
        )
        tables = {r[0] for r in mem_con.execute("SHOW TABLES").fetchall()}
        assert "tbl" not in tables

    # ------------------------------------------------------------------ append

    def test_append_inserts_new_rows(self, mem_con):
        w = DuckDBWriter(mem_con)
        df1 = pd.DataFrame({"id": ["a"], "snapshot_date": ["2026-01-01"], "v": [1]})
        df2 = pd.DataFrame({"id": ["b"], "snapshot_date": ["2026-01-02"], "v": [2]})
        w.append(df1, "hist", "id")
        w.append(df2, "hist", "id")
        count = mem_con.execute("SELECT count(*) FROM hist").fetchone()
        assert count is not None and count[0] == 2

    def test_append_skips_duplicate_rows(self, mem_con):
        w = DuckDBWriter(mem_con)
        df = pd.DataFrame({"id": ["a"], "snapshot_date": ["2026-01-01"], "v": [1]})
        w.append(df, "hist", "id")
        w.append(df, "hist", "id")  # same (id, snapshot_date) — must be skipped
        count = mem_con.execute("SELECT count(*) FROM hist").fetchone()
        assert count is not None and count[0] == 1

    def test_append_skips_empty_dataframe(self, mem_con):
        DuckDBWriter(mem_con).append(pd.DataFrame(), "hist", "id")
        tables = {r[0] for r in mem_con.execute("SHOW TABLES").fetchall()}
        assert "hist" not in tables

    def test_append_inserts_same_key_on_new_date(self, mem_con):
        w = DuckDBWriter(mem_con)
        df1 = pd.DataFrame({"id": ["a"], "snapshot_date": ["2026-01-01"], "v": [1]})
        df2 = pd.DataFrame({"id": ["a"], "snapshot_date": ["2026-01-02"], "v": [2]})
        w.append(df1, "hist", "id")
        w.append(df2, "hist", "id")  # same key, different date — must be inserted
        count = mem_con.execute("SELECT count(*) FROM hist").fetchone()
        assert count is not None and count[0] == 2

    def test_append_raises_storage_write_error(self):
        mock_con = MagicMock()
        # _table_exists → execute().fetchone() → (1,) means table exists
        exists_result = MagicMock()
        exists_result.fetchone.return_value = (1,)
        # second execute (INSERT) raises duckdb.Error
        mock_con.execute.side_effect = [exists_result, duckdb.Error("boom")]
        df = pd.DataFrame({"id": ["a"], "snapshot_date": ["2026-01-01"]})
        with pytest.raises(StorageWriteError):
            DuckDBWriter(mock_con).append(df, "hist", "id")

    def test_append_composite_key_deduplicates_on_all_columns(self, mem_con):
        w = DuckDBWriter(mem_con)
        df1 = pd.DataFrame(
            [
                {
                    "uuid": "u1",
                    "snapshot_date": "2026-06-24",
                    "retailer": "cardmarket",
                    "tx_type": "retail",
                    "finish": "normal",
                    "price": 3.20,
                }
            ]
        )
        w.append(df1, "t", ["uuid", "retailer", "tx_type", "finish"])
        # Same composite key — must be skipped
        w.append(df1, "t", ["uuid", "retailer", "tx_type", "finish"])
        count = mem_con.execute("SELECT count(*) FROM t").fetchone()[0]
        assert count == 1

    def test_append_composite_key_allows_different_finish(self, mem_con):
        w = DuckDBWriter(mem_con)
        row_normal = pd.DataFrame(
            [
                {
                    "uuid": "u1",
                    "snapshot_date": "2026-06-24",
                    "retailer": "cardmarket",
                    "tx_type": "retail",
                    "finish": "normal",
                    "price": 3.20,
                }
            ]
        )
        row_foil = pd.DataFrame(
            [
                {
                    "uuid": "u1",
                    "snapshot_date": "2026-06-24",
                    "retailer": "cardmarket",
                    "tx_type": "retail",
                    "finish": "foil",
                    "price": 8.50,
                }
            ]
        )
        w.append(row_normal, "t", ["uuid", "retailer", "tx_type", "finish"])
        w.append(row_foil, "t", ["uuid", "retailer", "tx_type", "finish"])
        count = mem_con.execute("SELECT count(*) FROM t").fetchone()[0]
        assert count == 2

    def test_append_composite_key_allows_different_retailer(self, mem_con):
        w = DuckDBWriter(mem_con)
        row_cm = pd.DataFrame(
            [
                {
                    "uuid": "u1",
                    "snapshot_date": "2026-06-24",
                    "retailer": "cardmarket",
                    "tx_type": "retail",
                    "finish": "normal",
                    "price": 3.20,
                }
            ]
        )
        row_tcp = pd.DataFrame(
            [
                {
                    "uuid": "u1",
                    "snapshot_date": "2026-06-24",
                    "retailer": "tcgplayer",
                    "tx_type": "retail",
                    "finish": "normal",
                    "price": 3.50,
                }
            ]
        )
        w.append(row_cm, "t", ["uuid", "retailer", "tx_type", "finish"])
        w.append(row_tcp, "t", ["uuid", "retailer", "tx_type", "finish"])
        count = mem_con.execute("SELECT count(*) FROM t").fetchone()[0]
        assert count == 2

    def test_append_str_key_still_works_after_change(self, mem_con):
        # Regression: existing str callers must not break
        w = DuckDBWriter(mem_con)
        df = pd.DataFrame({"id": ["a"], "snapshot_date": ["2026-01-01"], "v": [1]})
        w.append(df, "hist", "id")
        w.append(df, "hist", "id")
        count = mem_con.execute("SELECT count(*) FROM hist").fetchone()[0]
        assert count == 1
