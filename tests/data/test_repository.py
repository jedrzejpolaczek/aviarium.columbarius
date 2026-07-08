import duckdb
import pytest

from src.data.repository import DuckDBRepository, open_repository


def test_get_tables_reflects_created_tables():
    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE foo (x INT)")
    repo = DuckDBRepository(con)

    assert repo.get_tables() == {"foo"}
    con.close()


def test_query_df_returns_dataframe():
    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE foo AS SELECT * FROM (VALUES (1), (2)) AS t(x)")
    repo = DuckDBRepository(con)

    df = repo.query_df("SELECT * FROM foo ORDER BY x")

    assert df["x"].tolist() == [1, 2]
    con.close()


def test_query_df_with_params():
    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE foo AS SELECT * FROM (VALUES (1), (2)) AS t(x)")
    repo = DuckDBRepository(con)

    df = repo.query_df("SELECT * FROM foo WHERE x = ?", [2])

    assert df["x"].tolist() == [2]
    con.close()


def test_open_repository_creates_working_connection(tmp_path):
    db_path = str(tmp_path / "test.duckdb")
    repo = open_repository(db_path, read_only=False)
    repo.connection.execute("CREATE TABLE foo (x INT)")

    assert repo.get_tables() == {"foo"}
    repo.close()


def test_open_repository_wraps_connect_errors(tmp_path):
    from src.data.cards.storage.errors import StorageConnectionError

    # A path that points at an existing directory (rather than a file) is
    # guaranteed to fail duckdb.connect on every platform, including
    # Windows where "\0" in a path does not reliably trigger an OS-level
    # error the way it does on Linux.
    with pytest.raises(StorageConnectionError):
        open_repository(str(tmp_path), read_only=False)
