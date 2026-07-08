"""Thin wrapper around a DuckDB connection for callers outside the storage
tier (app/, scripts/, health checks) that need a named, injectable type
instead of importing duckdb.DuckDBPyConnection directly.

get_tables()/query_df() are included as low-cost convenience methods
alongside connection management, not because current callers duplicate
this logic ad hoc today — none of the four migrated call sites (app/main.py,
health.py, scripts/check_and_retrain.py, scripts/train_model.py) use them
as of this writing; they only need `.connection` and `.close()`. Kept for
symmetry with the already-existing free `get_tables()` function and as a
natural complement for future callers, not as a fix for an observed
duplication problem.

Does NOT replace direct SQL execution: ADR-024 established DuckDB as the
compute layer for the storage tier and the ml/features, ml/training, and
monitoring modules that run window-function queries — those keep taking
duckdb.DuckDBPyConnection exactly as before via `.connection`. See
ADR-029 for the full rationale on why this repository is scoped to the
connection-creation boundary rather than threaded through every SQL-running
function signature in the codebase.
"""

from typing import Self

import duckdb
import pandas as pd

from src.data.cards.storage.base.storage import get_tables
from src.data.db import open_connection


class DuckDBRepository:
    """Wraps one DuckDB connection with common read operations.

    `.connection` is public — callers that need to run arbitrary SQL
    (e.g. src.ml.features.pipeline.build_inference_features, which takes a
    raw duckdb.DuckDBPyConnection) pass `repo.connection` through unchanged.

    Supports the context-manager protocol (matching BaseStorage in the
    storage tier) for callers that prefer `with open_repository(...) as
    repo:` over an explicit try/finally `repo.close()`.
    """

    def __init__(self, connection: duckdb.DuckDBPyConnection) -> None:
        self.connection = connection

    def get_tables(self) -> set[str]:
        """Return the set of table names currently registered."""
        return get_tables(self.connection)

    def query_df(self, sql: str, params: list[object] | None = None) -> pd.DataFrame:
        """Run a query and return the result as a DataFrame."""
        return self.connection.execute(sql, params or []).df()

    def close(self) -> None:
        """Close the wrapped connection."""
        self.connection.close()

    def __enter__(self) -> Self:
        """Return self to allow use as a context manager."""
        return self

    def __exit__(self, *_: object) -> None:
        """Close the connection on exiting the context manager."""
        self.close()


def open_repository(db_path: str, *, read_only: bool) -> DuckDBRepository:
    """Open a DuckDB connection and wrap it in a DuckDBRepository.

    Centralizes what were previously 5 separate duckdb.connect(...) call
    sites (app/main.py, health.py x3, scripts/check_and_retrain.py,
    scripts/train_model.py), each independently handling read_only/path
    resolution with no shared error-wrapping or logging.
    """
    return DuckDBRepository(open_connection(db_path, read_only=read_only))
