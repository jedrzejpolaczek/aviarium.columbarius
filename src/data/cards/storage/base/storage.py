"""Base DuckDB storage class providing connection management and the context-manager protocol."""

import duckdb
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Self

from src.data.cards.storage.errors import StorageConnectionError
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