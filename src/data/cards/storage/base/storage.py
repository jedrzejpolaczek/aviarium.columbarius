"""Base DuckDB storage class providing connection management and the context-manager protocol."""

import duckdb
from abc import ABC, abstractmethod
from typing import Self

from src.data.db import open_connection


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
        """Open (or create) a DuckDB connection. Delegates to src.data.db.open_connection.

        Kept as a method so BronzeStorage/SilverStorage/GoldStorage/
        TransformStorage don't need to change their self._open_connection(...)
        call sites.
        """
        return open_connection(db_path, read_only=read_only)

    @abstractmethod
    def close(self) -> None:
        """Close all open DuckDB connections."""

    def __enter__(self) -> Self:
        """Return self to allow use as a context manager."""
        return self

    def __exit__(self, *_: object) -> None:
        """Close connections on exiting the context manager."""
        self.close()
