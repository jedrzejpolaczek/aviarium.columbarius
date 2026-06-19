"""Tests for BaseStorage, TransformStorage, and DuckDBWriter base classes."""

import json
from unittest.mock import MagicMock, patch

import duckdb
import pandas as pd
import pytest

from src.data.cards.storage.base import (
    BaseStorage,
    DuckDBWriter,
    TransformStorage,
    _serialize_objects,
    get_tables,
)
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
        with patch("src.data.cards.storage.base.Path") as MockPath:
            mock_instance = MagicMock()
            mock_instance.parent.mkdir.side_effect = PermissionError("denied")
            MockPath.return_value = mock_instance
            with pytest.raises((StorageConnectionError, PermissionError)):
                BaseStorage._open_connection(bad_path, read_only=False)

    def test_raises_storage_connection_error_when_duckdb_fails(self):
        with patch("src.data.cards.storage.base.duckdb.connect") as mock_connect:
            mock_connect.side_effect = duckdb.Error("connection failed")
            with pytest.raises(StorageConnectionError, match="Cannot open DuckDB"):
                BaseStorage._open_connection(":memory:", read_only=False)


# ---------------------------------------------------------------------------
# _serialize_objects (module-level function)
# ---------------------------------------------------------------------------


class TestSerializeObjects:
    def test_serializes_dict_cells_to_json_string(self):
        df = pd.DataFrame({"col": [{"a": 1}]})
        result = _serialize_objects(df)
        assert json.loads(result["col"].iloc[0]) == {"a": 1}

    def test_leaves_non_object_columns_unchanged(self):
        df = pd.DataFrame({"num": [1, 2], "flt": [1.1, 2.2]})
        result = _serialize_objects(df)
        assert list(result["num"]) == [1, 2]
        assert list(result["flt"]) == [1.1, 2.2]

    def test_leaves_string_cells_unchanged(self):
        df = pd.DataFrame({"col": ["hello", "world"]})
        result = _serialize_objects(df)
        assert list(result["col"]) == ["hello", "world"]

    def test_leaves_none_cells_unchanged(self):
        df = pd.DataFrame({"col": [None, {"a": 1}]})
        result = _serialize_objects(df)
        assert pd.isna(result["col"].iloc[0])
        assert json.loads(result["col"].iloc[1]) == {"a": 1}

    def test_returns_dataframe(self):
        df = pd.DataFrame({"a": [1]})
        result = _serialize_objects(df)
        assert isinstance(result, pd.DataFrame)

    def test_serialize_objects_does_not_mutate_input(self):
        df = pd.DataFrame({"a": [{"key": "val"}, None], "b": [1, 2]})
        original_a = df["a"].tolist()
        _serialize_objects(df)
        # Input must be unchanged after the call
        assert df["a"].tolist() == original_a

    def test_serialize_objects_returns_serialized_copy(self):
        df = pd.DataFrame({"a": [{"key": "val"}, None], "b": [1, 2]})
        result = _serialize_objects(df)
        assert result["a"].iloc[0] == '{"key": "val"}'
        assert result["a"].iloc[1] is None
        # Original is not changed
        assert isinstance(df["a"].iloc[0], dict)

    def test_list_cells_serialized_to_json_string(self):
        df = pd.DataFrame({"col": [[1, 2, 3]]})
        result = _serialize_objects(df)
        assert json.loads(result["col"].iloc[0]) == [1, 2, 3]

    def test_list_with_none_serializes_list_leaves_none(self):
        df = pd.DataFrame({"col": [[1, 2], None]})
        result = _serialize_objects(df)
        assert json.loads(result["col"].iloc[0]) == [1, 2]
        assert pd.isna(result["col"].iloc[1])


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
    def test_returns_table_names(self):
        conn = duckdb.connect(":memory:")
        conn.execute("CREATE TABLE test_table AS SELECT 1 AS col")
        tables = get_tables(conn)
        assert "test_table" in tables
        conn.close()

    def test_returns_empty_set_for_empty_database(self):
        conn = duckdb.connect(":memory:")
        tables = get_tables(conn)
        assert tables == set()
        conn.close()

    def test_returns_multiple_tables(self):
        conn = duckdb.connect(":memory:")
        conn.execute("CREATE TABLE table1 AS SELECT 1 AS col")
        conn.execute("CREATE TABLE table2 AS SELECT 2 AS col")
        tables = get_tables(conn)
        assert "table1" in tables
        assert "table2" in tables
        conn.close()


# ---------------------------------------------------------------------------
# DuckDBWriter
# ---------------------------------------------------------------------------


@pytest.fixture
def mem_con():
    """In-memory DuckDB connection, closed after each test."""
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


class TestDuckDBWriterColumnTypes:
    """Verify that column_types applies SQL CASTs when creating/inserting."""

    def test_full_load_with_column_types_stores_as_map(self, mem_con):
        df = pd.DataFrame(
            {"id": ["a"], "meta": [{"commander": "legal", "standard": "banned"}]}
        )
        DuckDBWriter(mem_con).full_load(
            df, "tbl", column_types={"meta": "MAP(VARCHAR, VARCHAR)"}
        )
        row = mem_con.execute("SELECT meta FROM tbl").fetchone()
        assert row is not None and row[0] == {
            "commander": "legal",
            "standard": "banned",
        }

    def test_upsert_creates_with_column_types(self, mem_con):
        df = pd.DataFrame({"id": ["a"], "meta": [{"commander": "legal"}]})
        DuckDBWriter(mem_con).upsert(
            df, "tbl", "id", column_types={"meta": "MAP(VARCHAR, VARCHAR)"}
        )
        row = mem_con.execute("SELECT meta FROM tbl").fetchone()
        assert row is not None and row[0] == {"commander": "legal"}

    def test_upsert_replaces_with_column_types(self, mem_con):
        w = DuckDBWriter(mem_con)
        df1 = pd.DataFrame({"id": ["a"], "meta": [{"commander": "legal"}]})
        df2 = pd.DataFrame({"id": ["a"], "meta": [{"standard": "banned"}]})
        ct = {"meta": "MAP(VARCHAR, VARCHAR)"}
        w.upsert(df1, "tbl", "id", column_types=ct)
        w.upsert(df2, "tbl", "id", column_types=ct)
        row = mem_con.execute("SELECT meta FROM tbl").fetchone()
        assert row is not None and row[0] == {"standard": "banned"}

    def test_append_creates_with_column_types(self, mem_con):
        df = pd.DataFrame(
            {
                "id": ["a"],
                "snapshot_date": ["2026-01-01"],
                "meta": [{"commander": "legal"}],
            }
        )
        DuckDBWriter(mem_con).append(
            df, "hist", "id", column_types={"meta": "MAP(VARCHAR, VARCHAR)"}
        )
        row = mem_con.execute("SELECT meta FROM hist").fetchone()
        assert row is not None and row[0] == {"commander": "legal"}

    def test_append_inserts_into_existing_with_column_types(self, mem_con):
        w = DuckDBWriter(mem_con)
        df1 = pd.DataFrame(
            {
                "id": ["a"],
                "snapshot_date": ["2026-01-01"],
                "meta": [{"commander": "legal"}],
            }
        )
        df2 = pd.DataFrame(
            {
                "id": ["b"],
                "snapshot_date": ["2026-01-02"],
                "meta": [{"standard": "banned"}],
            }
        )
        ct = {"meta": "MAP(VARCHAR, VARCHAR)"}
        w.append(df1, "hist", "id", column_types=ct)
        w.append(df2, "hist", "id", column_types=ct)
        rows = mem_con.execute("SELECT id, meta FROM hist ORDER BY id").fetchall()
        assert rows[0] == ("a", {"commander": "legal"})
        assert rows[1] == ("b", {"standard": "banned"})

    def test_append_skips_duplicate_with_column_types(self, mem_con):
        w = DuckDBWriter(mem_con)
        df = pd.DataFrame(
            {
                "id": ["a"],
                "snapshot_date": ["2026-01-01"],
                "meta": [{"commander": "legal"}],
            }
        )
        ct = {"meta": "MAP(VARCHAR, VARCHAR)"}
        w.append(df, "hist", "id", column_types=ct)
        w.append(df, "hist", "id", column_types=ct)
        count = mem_con.execute("SELECT count(*) FROM hist").fetchone()
        assert count is not None and count[0] == 1

    def test_column_types_none_behaves_identically_to_no_arg(self, mem_con):
        df = pd.DataFrame({"id": ["a"], "v": [1]})
        DuckDBWriter(mem_con).full_load(df, "tbl", column_types=None)
        row = mem_con.execute("SELECT v FROM tbl").fetchone()
        assert row is not None and row[0] == 1
