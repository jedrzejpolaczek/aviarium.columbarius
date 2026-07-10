"""Shared DuckDB connection factory.

Used by both BaseStorage (the storage-tier ABC in
src/data/cards/storage/base/storage.py) and DuckDBRepository
(src/data/repository.py, for callers outside the storage tier: app/,
scripts/, health checks) — one implementation of connection-opening
instead of each call site reimplementing duckdb.connect + error wrapping.
"""

from pathlib import Path

import duckdb

from src.data.cards.storage.errors import StorageConnectionError
from src.logger import get_logger

logger = get_logger(__name__)


def open_connection(db_path: str, read_only: bool) -> duckdb.DuckDBPyConnection:
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
        raise StorageConnectionError(f"Cannot open DuckDB at {db_path!r}: {e}") from e
