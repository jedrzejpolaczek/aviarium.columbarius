"""Unified DuckDB write primitives (full_load, upsert, append) used across all storage tiers."""

import json

import duckdb
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype

from src.data.cards.storage.errors import StorageWriteError

from src.logger import get_logger


logger = get_logger(__name__)


class DuckDBWriter:
    """Unified DuckDB write primitives shared by all storage tiers.

    Provides three write patterns:
        full_load — DROP + CREATE OR REPLACE (initial load or full rebuild)
        upsert    — DELETE + INSERT by key column (current-state tables)
        append    — LEFT JOIN anti-join insert, dedup by (key_column(s), snapshot_date)
                    (history tables that must never lose rows)

    All methods call _serialize to prepare the DataFrame before DuckDB registration
    and raise StorageWriteError on DuckDB failure.
    """

    def __init__(self, con: duckdb.DuckDBPyConnection) -> None:
        self._con = con

    DATE_COLUMNS = ("snapshot_date", "tournament_date")
    """Columns that must land as DATE, never VARCHAR.

    Every one of these was VARCHAR in all three tiers until the 2026-09-26
    migration, because the ingestion side hands DuckDB ISO strings and
    ``CREATE TABLE … AS SELECT * FROM _staging`` adopts whatever type the
    staging frame has. Inserts into the migrated tables would keep working
    without this (DuckDB casts the string on the way in), but a table created
    from scratch — a fresh install, or initial_pipeline on a new machine —
    would silently go back to VARCHAR and need migrating all over again.
    """

    @classmethod
    def _serialize(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of df ready for conn.register().

        Two normalisations:

        1. dict and list cells become JSON strings. DuckDB cannot infer a
           consistent type for such columns when rows have different key sets
           or contain None; JSON VARCHAR gives every row the same type.
        2. DATE_COLUMNS are parsed to real dates, so a newly created table gets
           a DATE column rather than VARCHAR.

        Date parsing deliberately raises rather than coercing: an unparseable
        snapshot_date means the ingestion side produced something unexpected,
        and writing NULL into a history table's dedup key would be far worse
        than failing the run.
        """
        df = df.copy()

        for col in df.select_dtypes(include=["object"]).columns:
            # Cell-level isinstance check needed: a single column can mix dicts, None, and scalars.
            df[col] = df[col].map(
                lambda x: json.dumps(x) if isinstance(x, (dict, list)) else x
            )

        for col in cls.DATE_COLUMNS:
            if col in df.columns and not is_datetime64_any_dtype(df[col]):
                df[col] = pd.to_datetime(df[col], errors="raise").dt.date

        return df

    def full_load(
        self,
        df: pd.DataFrame,
        table_name: str,
    ) -> None:
        """Drop and recreate a table from the given DataFrame.

        Intended for initial population or a full rebuild.
        For daily updates use upsert().

        Args:
            df: DataFrame to persist. If empty, the write is skipped and a
                warning is logged.
            table_name: Name of the table to create or replace.

        Raises:
            StorageWriteError: If the DuckDB write fails.
        """
        if df.empty:
            logger.warning("No data to load into %r — skipping", table_name)
            return

        # Register df as a virtual view so it is readable inside SQL.
        staging = self._serialize(df)
        self._con.register("_staging", staging)
        try:
            self._con.execute(
                f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM _staging"
            )
            logger.info("Full load: %d rows into %r", len(df), table_name)
        except duckdb.Error as e:
            raise StorageWriteError(f"Failed to full-load {table_name!r}: {e}") from e
        finally:
            # Always unregister — a lingering _staging view collides on the next call.
            self._con.unregister("_staging")

    def upsert(
        self,
        df: pd.DataFrame,
        table_name: str,
        key_column: str,
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

        Raises:
            StorageWriteError: If the DuckDB write fails.
        """
        if df.empty:
            logger.warning("No data to upsert into %r — skipping", table_name)
            return

        staging = self._serialize(df)
        self._con.register("_staging", staging)
        try:
            if not self._table_exists(table_name):
                # First call — create the table directly from staging.
                self._con.execute(
                    f"CREATE TABLE {table_name} AS SELECT * FROM _staging"
                )
                logger.info("Created table %r with %d rows", table_name, len(df))

            else:
                if self._schema_differs("_staging", table_name):
                    # Schema change (added/removed columns): recreate to adopt the new shape.
                    # The next upsert cycle will restore all current-state data.
                    logger.warning(
                        "Schema change detected for %r — recreating table with new schema",
                        table_name,
                    )
                    self._con.execute(
                        f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM _staging"
                    )
                    logger.info("Recreated %r with %d rows", table_name, len(df))

                else:
                    # Normal upsert — remove stale versions of incoming keys, then reinsert.
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
        key_column: str | list[str],
    ) -> None:
        """Append rows to a history table, skipping already-snapshotted pairs.

        Rows already present for the same composite key are skipped, so calling
        this multiple times is safe. History tables accumulate one snapshot per
        day and must never lose rows.

        Deduplication key: (key_column, snapshot_date) when key_column is a str;
        (col1, col2, …, snapshot_date) when key_column is a list.

        Note: deduplication is inter-call only — duplicate key+date pairs
        within the same ``df`` are not removed.

        Args:
            df: DataFrame to append. If empty, the write is skipped and a
                warning is logged.
            table_name: Target history table name.
            key_column: Column name(s) forming the composite dedup key together
                with snapshot_date.

        Raises:
            StorageWriteError: If the DuckDB write fails.
        """
        if df.empty:
            logger.warning("No data to append into %r — skipping", table_name)
            return

        key_cols = [key_column] if isinstance(key_column, str) else list(key_column)
        if not key_cols:
            raise ValueError("key_column must not be empty")
        join_conditions = " AND ".join(
            ["t.snapshot_date = s.snapshot_date"]
            + [f"t.{col} = s.{col}" for col in key_cols]
        )
        null_check = f"t.{key_cols[0]} IS NULL"

        staging = self._serialize(df)
        self._con.register("_staging", staging)
        try:
            if not self._table_exists(table_name):
                # First call — create the history table directly from staging.
                self._con.execute(
                    f"CREATE TABLE {table_name} AS SELECT * FROM _staging"
                )
                logger.info(
                    "Created history table %r with %d rows", table_name, len(df)
                )

            else:
                # LEFT JOIN anti-join rather than NOT EXISTS: lets DuckDB use a hash join
                # on the composite key instead of a correlated subquery per row.
                self._con.execute(
                    f"INSERT INTO {table_name} "
                    f"SELECT s.* FROM _staging s "
                    f"LEFT JOIN {table_name} t ON {join_conditions} "
                    f"WHERE {null_check}"
                )
                logger.info("Appended rows into %r", table_name)

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
        # DESCRIBE gives the inferred schema of any view or query result.
        staging = {
            r[0]: r[1]
            for r in self._con.execute(
                f"DESCRIBE SELECT * FROM {staging_view}"
            ).fetchall()
        }

        # information_schema reflects the persisted table's column definitions.
        table = {
            r[0]: r[1]
            for r in self._con.execute(
                "SELECT column_name, data_type FROM information_schema.columns "
                "WHERE table_name = ? ORDER BY ordinal_position",
                [table_name],
            ).fetchall()
        }

        return staging != table
