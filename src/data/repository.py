"""Thin wrapper around a DuckDB connection for callers outside the storage
tier (app/, scripts/, health checks) that need a named, injectable type
instead of importing duckdb.DuckDBPyConnection directly, plus the
get_tables()/query_df() convenience operations several of those callers
already reimplement ad hoc.

Does NOT replace direct SQL execution: ADR-024 established DuckDB as the
compute layer for the storage tier and the ml/features, ml/training, and
monitoring modules that run window-function queries — those keep taking
duckdb.DuckDBPyConnection exactly as before via `.connection`. See
ADR-029 for the full rationale on why this repository is scoped to the
connection-creation boundary rather than threaded through every SQL-running
function signature in the codebase.
"""

import duckdb
import pandas as pd

from src.data.cards.storage.base.storage import get_tables
from src.data.db import open_connection


class DuckDBRepository:
    """Wraps one DuckDB connection with common read operations.

    `.connection` is public — callers that need to run arbitrary SQL
    (e.g. src.ml.features.pipeline.build_inference_features, which takes a
    raw duckdb.DuckDBPyConnection) pass `repo.connection` through unchanged.
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


def open_repository(db_path: str, *, read_only: bool) -> DuckDBRepository:
    """Open a DuckDB connection and wrap it in a DuckDBRepository.

    Centralizes what were previously 5 separate duckdb.connect(...) call
    sites (app/main.py, health.py x3, scripts/check_and_retrain.py,
    scripts/train_model.py), each independently handling read_only/path
    resolution with no shared error-wrapping or logging.
    """
    return DuckDBRepository(open_connection(db_path, read_only=read_only))
