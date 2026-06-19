"""Shared base classes and write primitives for the Bronze → Silver → Gold DuckDB storage tiers.

Exports:
    DuckDBWriter  — unified write primitives (full_load, upsert, append) used by
                    all three storage tiers.
    BaseStorage   — abstract base providing connection management and context-manager
                    protocol.
    TransformStorage — abstract base for Silver and Gold transformation layers.
    get_tables    — helper that returns the set of table names in a connection.
    _serialize_objects — serialises dict cells to JSON strings before writing.
    _prepare_staging   — prepares a DataFrame for DuckDB registration, applying
                         typed PyArrow MAP arrays for columns in column_types.
"""

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Self

import duckdb
import pandas as pd
import pyarrow as pa

from src.data.cards.storage.errors import StorageConnectionError, StorageWriteError
from src.logger import get_logger


logger = get_logger(__name__)


def get_tables(conn: duckdb.DuckDBPyConnection) -> set[str]:
    """Return the set of table names currently registered in a DuckDB connection.

    Args:
        conn: Open DuckDB connection.

    Returns:
        Set of table name strings from SHOW TABLES.
    """
    return {row[0] for row in conn.execute("SHOW TABLES").fetchall()}


def _serialize_objects(df: pd.DataFrame) -> pd.DataFrame:
    """Serialize dict cells to JSON strings before writing to DuckDB.

    DuckDB cannot infer a consistent STRUCT schema for dict columns when some rows
    have a dict value and others have NULL, or when different rows have different
    key sets. Serializing dicts to JSON VARCHAR gives every row a consistent type.

    All lists are serialized regardless of element type. Simple lists (list[str])
    were previously passed through as native arrays, but this caused two bugs:
    (1) DuckDB's pandas integration returns VARCHAR[] as numpy ndarray, breaking
    Silver's _clean_lists isinstance(x, list) check; (2) inserting a native list
    into an existing VARCHAR snapshot column (e.g. bronze_scryfall_meta_history)
    produces DuckDB's non-JSON format '[nonfoil, foil]' instead of valid JSON
    '["nonfoil", "foil"]', breaking Silver's _parse_json_columns.

    Args:
        df: DataFrame whose object columns may contain dicts or lists.

    Returns:
        DataFrame with dict and list cells replaced by their JSON representations.
    """
    df = df.copy()
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].map(
            lambda x: json.dumps(x) if isinstance(x, (dict, list)) else x
        )
    return df


def _prepare_staging(
    df: pd.DataFrame,
    column_types: dict[str, str] | None,
) -> pd.DataFrame | pa.Table:
    """Prepare a DataFrame for DuckDB registration.

    For columns NOT in column_types: serializes dict and list cells to JSON strings
    via the same logic as _serialize_objects, so DuckDB infers VARCHAR.

    For columns IN column_types: skips JSON serialization and converts Python dicts
    to typed PyArrow MAP arrays. DuckDB sees the correct MAP type directly — no SQL
    CAST is needed. CAST(VARCHAR → MAP) does not work in DuckDB; this Arrow-level
    conversion is the supported path.

    Args:
        df: Input DataFrame. Not modified.
        column_types: Mapping of column name → DuckDB type string, or None.
            Currently only MAP(VARCHAR, VARCHAR) is supported for typed columns.

    Returns:
        A pandas DataFrame when column_types is None; a PyArrow Table otherwise.
        DuckDB's register() accepts both.
    """
    typed_cols = frozenset(column_types) if column_types else frozenset()

    df_copy = df.copy()
    for col in df_copy.select_dtypes(include=["object"]).columns:
        if col not in typed_cols:
            df_copy[col] = df_copy[col].map(
                lambda x: json.dumps(x) if isinstance(x, (dict, list)) else x
            )

    if not column_types:
        return df_copy

    table = pa.Table.from_pandas(df_copy, preserve_index=False)
    col_names = list(df_copy.columns)
    for col in column_types:
        if col not in df_copy.columns:
            continue
        col_idx = col_names.index(col)
        arr = pa.array(
            [list(v.items()) if isinstance(v, dict) else None for v in df[col]],
            type=pa.map_(pa.string(), pa.string()),
        )
        table = table.set_column(col_idx, col, arr)
    return table


class DuckDBWriter:
    """Unified DuckDB write primitives shared by all storage tiers.

    Provides three write patterns:
        full_load — DROP + CREATE OR REPLACE (initial load or full rebuild)
        upsert    — DELETE + INSERT by key column (current-state tables)
        append    — LEFT JOIN anti-join insert, dedup by (key_column, snapshot_date)
                    (history tables that must never lose rows)

    All methods call _prepare_staging before writing and raise StorageWriteError
    on DuckDB failure.
    """

    def __init__(self, con: duckdb.DuckDBPyConnection) -> None:
        self._con = con

    def full_load(
        self,
        df: pd.DataFrame,
        table_name: str,
        column_types: dict[str, str] | None = None,
    ) -> None:
        """Drop and recreate a table from the given DataFrame.

        Intended for initial population or a full rebuild.
        For daily updates use upsert().

        Args:
            df: DataFrame to persist. If empty, the write is skipped and a
                warning is logged.
            table_name: Name of the table to create or replace.
            column_types: Optional mapping of column name → DuckDB type string.
                When provided, dict cells are first serialised to JSON by
                _serialize_objects, then cast to the declared type via SQL
                CAST (e.g. {"legalities": "MAP(VARCHAR, VARCHAR)"}).

        Raises:
            StorageWriteError: If the DuckDB write fails.
        """
        if df.empty:
            logger.warning("No data to load into %r — skipping", table_name)
            return
        staging = _prepare_staging(df, column_types)
        self._con.register("_staging", staging)
        try:
            self._con.execute(
                f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM _staging"
            )
            logger.info("Full load: %d rows into %r", len(df), table_name)
        except duckdb.Error as e:
            raise StorageWriteError(f"Failed to full-load {table_name!r}: {e}") from e
        finally:
            self._con.unregister("_staging")

    def upsert(
        self,
        df: pd.DataFrame,
        table_name: str,
        key_column: str,
        column_types: dict[str, str] | None = None,
    ) -> None:
        """Upsert a DataFrame into a table by key column.

        Deletes existing rows whose key matches any incoming record, then inserts
        all incoming records. Rows not present in the incoming batch are left
        untouched. Creates the table from the incoming data if it does not exist yet.

        Args:
            df: DataFrame to upsert. If empty, the write is skipped and a
                warning is logged.
            table_name: Target table name.
            key_column: Column used to identify existing rows (e.g. "uuid" or "id").
            column_types: Optional mapping of column name → DuckDB type string.
                When provided, dict cells are first serialised to JSON by
                _serialize_objects, then cast to the declared type via SQL CAST.

        Raises:
            StorageWriteError: If the DuckDB write fails.
        """
        if df.empty:
            logger.warning("No data to upsert into %r — skipping", table_name)
            return
        staging = _prepare_staging(df, column_types)
        self._con.register("_staging", staging)
        try:
            if not self._table_exists(table_name):
                self._con.execute(
                    f"CREATE TABLE {table_name} AS SELECT * FROM _staging"
                )
                logger.info("Created table %r with %d rows", table_name, len(df))
            else:
                if self._schema_differs("_staging", table_name):
                    logger.warning(
                        "Schema change detected for %r — recreating table with new schema",
                        table_name,
                    )
                    self._con.execute(
                        f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM _staging"
                    )
                    logger.info("Recreated %r with %d rows", table_name, len(df))
                else:
                    self._con.execute(
                        f"DELETE FROM {table_name} WHERE {key_column} IN "
                        f"(SELECT {key_column} FROM _staging)"
                    )
                    self._con.execute(
                        f"INSERT INTO {table_name} SELECT * FROM _staging"
                    )
                    logger.info("Upserted %d rows into %r", len(df), table_name)
        except duckdb.Error as e:
            raise StorageWriteError(f"Failed to upsert into {table_name!r}: {e}") from e
        finally:
            self._con.unregister("_staging")

    def append(
        self,
        df: pd.DataFrame,
        table_name: str,
        key_column: str,
        column_types: dict[str, str] | None = None,
    ) -> None:
        """Append rows to a history table, skipping already-snapshotted pairs.

        Rows already present for the same (key_column, snapshot_date) pair are
        skipped, so calling this multiple times on the same day is safe.
        History tables accumulate one snapshot per day and must never lose rows.
        Use upsert() for current-state tables that should replace rows.

        Deduplication uses a LEFT JOIN anti-join rather than a correlated NOT EXISTS
        subquery so DuckDB can use a hash join on (key, snapshot_date).

        Note: deduplication is inter-call only — duplicate (key, snapshot_date) pairs
        within the same ``df`` are not removed.

        Args:
            df: DataFrame to append. If empty, the write is skipped and a
                warning is logged.
            table_name: Target history table name.
            key_column: Primary key column (e.g. "uuid" or "id") used together
                with snapshot_date to detect already-snapshotted rows.
            column_types: Optional mapping of column name → DuckDB type string.
                When provided, dict cells are first serialised to JSON by
                _serialize_objects, then cast to the declared type via SQL CAST.
                The subquery wraps _staging so CASTs are applied before the
                anti-join deduplication.

        Raises:
            StorageWriteError: If the DuckDB write fails.
        """
        if df.empty:
            logger.warning("No data to append into %r — skipping", table_name)
            return
        staging = _prepare_staging(df, column_types)
        self._con.register("_staging", staging)
        try:
            if not self._table_exists(table_name):
                self._con.execute(
                    f"CREATE TABLE {table_name} AS SELECT * FROM _staging"
                )
                logger.info(
                    "Created history table %r with %d rows", table_name, len(df)
                )
            else:
                self._con.execute(
                    f"INSERT INTO {table_name} "
                    f"SELECT s.* FROM _staging s "
                    f"LEFT JOIN {table_name} t "
                    f"  ON t.{key_column} = s.{key_column} "
                    f"  AND t.snapshot_date = s.snapshot_date "
                    f"WHERE t.{key_column} IS NULL"
                )
                logger.info("Appended %d rows into %r", len(df), table_name)
        except duckdb.Error as e:
            raise StorageWriteError(f"Failed to append into {table_name!r}: {e}") from e
        finally:
            self._con.unregister("_staging")

    def _table_exists(self, table_name: str) -> bool:
        """Return True if table_name exists in the connected database."""
        row = self._con.execute(
            "SELECT count(*) FROM information_schema.tables WHERE table_name = ?",
            [table_name],
        ).fetchone()
        return row is not None and row[0] > 0

    def _schema_differs(self, staging_view: str, table_name: str) -> bool:
        """Return True if column names or types differ between a registered view and a table."""
        staging = {
            r[0]: r[1]
            for r in self._con.execute(
                f"DESCRIBE SELECT * FROM {staging_view}"
            ).fetchall()
        }
        table = {
            r[0]: r[1]
            for r in self._con.execute(
                "SELECT column_name, data_type FROM information_schema.columns "
                "WHERE table_name = ? ORDER BY ordinal_position",
                [table_name],
            ).fetchall()
        }
        return staging != table


class BaseStorage(ABC):
    """Common DuckDB utilities shared by all storage tiers.

    Provides connection opening and the context-manager protocol.
    Subclasses must implement close().
    """

    @staticmethod
    def _open_connection(db_path: str, read_only: bool) -> duckdb.DuckDBPyConnection:
        """Open (or create) a DuckDB connection.

        Args:
            db_path: File path for the database, or ":memory:" for an
                in-memory database that is lost when the connection closes.
            read_only: Whether to open the database in read-only mode.

        Returns:
            An open DuckDB connection.

        Raises:
            StorageConnectionError: If the connection cannot be established.
        """
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        try:
            con = duckdb.connect(db_path, read_only=read_only)
            logger.info("Connected to DuckDB (read_only=%s) at %s", read_only, db_path)
            return con
        except duckdb.Error as e:
            raise StorageConnectionError(
                f"Cannot open DuckDB at {db_path!r}: {e}"
            ) from e

    @abstractmethod
    def close(self) -> None:
        """Close all open DuckDB connections."""

    def __enter__(self) -> Self:
        """Return self to allow use as a context manager."""
        return self

    def __exit__(self, *_: object) -> None:
        """Close connections on exiting the context manager."""
        self.close()


class TransformStorage(BaseStorage):
    """Base for transformation-layer storage (Silver and Gold tiers).

    Subclasses implement _pipeline(update) which is called by the
    public populate() and update() entry points.
    """

    @abstractmethod
    def _pipeline(self, update: bool) -> None:
        """Run the transformation pipeline.

        Args:
            update: If True, upsert into existing tables. If False, full rebuild.
        """

    def populate(self) -> None:
        """Full rebuild of all tables."""
        logger.info("Starting %s populate (full rebuild)", self.__class__.__name__)
        self._pipeline(update=False)

    def update(self) -> None:
        """Incremental update of all tables."""
        logger.info("Starting %s update (incremental)", self.__class__.__name__)
        self._pipeline(update=True)
