"""Base DuckDB storage class providing connection management and the context-manager protocol."""

import duckdb
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Self

from src.data.db import open_connection
from src.logger import ProgressLogger


def get_tables(conn: duckdb.DuckDBPyConnection) -> set[str]:
    """Return the set of table names currently registered in a DuckDB connection.

    Args:
        conn: Open DuckDB connection.

    Returns:
        Set of table name strings from SHOW TABLES.
    """
    return {row[0] for row in conn.execute("SHOW TABLES").fetchall()}


def warn_if_missing(
    logger: ProgressLogger,
    required: Sequence[str],
    available: set[str],
    skip_target: str,
    *,
    tier_label: str,
) -> bool:
    """Return True (and log a warning) if any of `required` is absent from `available`.

    Shared by bronze/silver/gold storage classes so the "skip this build if
    an upstream table is missing" guard clause is written once, not
    reinvented per tier — see ADR-030.
    """
    missing = [t for t in required if t not in available]
    if missing:
        logger.warning(
            "Missing %s tables %s — skipping %s", tier_label, missing, skip_target
        )
        return True
    return False


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
