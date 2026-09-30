"""Build DuckDB fixture tables with the production schema.

Test data is written as pandas frames with ISO date strings because that is
readable, but ``CREATE TABLE … AS SELECT * FROM df`` then produces VARCHAR
date columns (or INTEGER, for an empty frame) — a schema production has not
had since the 0.2.0 migration. The guard in ``tests/conftest.py`` fails any
test that leaves such a table behind; this helper is the one-line way to
build a table that passes it.
"""

import duckdb
import pandas as pd

from src.data.cards.storage.schema import DATE_COLUMNS


def create_table_from_df(
    con: duckdb.DuckDBPyConnection,
    table_name: str,
    df: pd.DataFrame,
    if_not_exists: bool = False,
) -> None:
    """Create *table_name* from *df*, with every production date column typed DATE."""
    clause = "IF NOT EXISTS " if if_not_exists else ""
    con.register("_seed_df", df)
    try:
        con.execute(f"CREATE TABLE {clause}{table_name} AS SELECT * FROM _seed_df")
    finally:
        con.unregister("_seed_df")
    for column in DATE_COLUMNS:
        if column in df.columns:
            # Via VARCHAR so every source type casts: ISO strings, all-NULL
            # columns DuckDB inferred as INTEGER, and datetime64 → TIMESTAMP.
            con.execute(
                f"ALTER TABLE {table_name} ALTER COLUMN {column} TYPE DATE "
                f"USING CAST(CAST({column} AS VARCHAR) AS DATE)"
            )
