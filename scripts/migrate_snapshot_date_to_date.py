"""One-off migration: snapshot_date / tournament_date VARCHAR → DATE.

Every history table across all three tiers stores its date column as VARCHAR.
Nothing is broken by that on its own — DuckDB casts a bound date parameter
against a VARCHAR column — but it leaks in ways that cost real time:

- ``WHERE snapshot_date BETWEEN DATE '…' AND DATE '…'`` raises
  ``Binder Error: Cannot mix values of type VARCHAR and DATE``, so ad-hoc
  analysis has to know which tables are which;
- the health-check test fixture declared ``snapshot_date DATE`` while
  production had VARCHAR, so the tests exercised a schema that did not exist;
- string ordering and date ordering agree only because every value happens to
  be zero-padded ISO. Nothing enforces that.

WHY THIS ORDER (columns first, code second):
``DuckDBWriter.append`` inserts with ``INSERT INTO t SELECT s.* FROM _staging``
and never recreates the table, so a staging batch whose column is still
VARCHAR casts cleanly into a DATE column. Migrating the columns first is
therefore safe with the current code in place, and the code change that follows
is a tidy-up rather than a flag day. The reverse order would have been safe too
but pointless — DuckDB would cast the new DATE staging straight back to VARCHAR.

Note this is emphatically NOT true of ``DuckDBWriter.upsert``: on any schema
difference it runs ``CREATE OR REPLACE TABLE … AS SELECT * FROM _staging``,
which keeps only the incoming batch. That is fine for the current-state tables
it serves (they are re-downloaded in full daily) and would be catastrophic for
a history table. No history table uses it; this comment exists so the next
person checks before assuming.

Usage:
    python -m scripts.migrate_snapshot_date_to_date --dry-run
    python -m scripts.migrate_snapshot_date_to_date
"""

import argparse
import sys
import time
from pathlib import Path

import duckdb

from src.data.cards.pipelines import load_config
from src.logger import get_logger, setup_logging

logger = get_logger(__name__)

# (tier config key, column name) → every table carrying a date-typed column.
# Discovered by inspection rather than hard-coded blindly: the script re-checks
# each table's current type and skips anything already migrated, so re-running
# it is a no-op.
_DATE_COLUMNS = ("snapshot_date", "tournament_date")

_TIER_KEYS = (
    "bronze_duckdb_path",
    "silver_duckdb_path",
    "gold_duckdb_path",
)


def find_varchar_date_columns(
    con: duckdb.DuckDBPyConnection,
) -> list[tuple[str, str]]:
    """Return (table, column) pairs still typed VARCHAR that should be DATE."""
    found: list[tuple[str, str]] = []
    tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
    for table in tables:
        for name, dtype, *_ in con.execute(f"DESCRIBE {table}").fetchall():
            if name in _DATE_COLUMNS and dtype == "VARCHAR":
                found.append((table, name))
    return found


def unparseable_values(con: duckdb.DuckDBPyConnection, table: str, column: str) -> int:
    """Count values that would not survive the cast.

    TRY_CAST yields NULL where CAST would raise, so a non-zero result here means
    the ALTER would fail — better to find out before rewriting 37 GB.
    """
    row = con.execute(
        f"SELECT COUNT(*) FROM {table} "
        f"WHERE {column} IS NOT NULL AND TRY_CAST({column} AS DATE) IS NULL"
    ).fetchone()
    return int(row[0]) if row else 0


def migrate_file(path: str, dry_run: bool) -> int:
    """Migrate one DuckDB file. Returns the number of columns changed."""
    if not Path(path).exists():
        logger.warning("%s does not exist — skipping", path)
        return 0

    con = duckdb.connect(path, read_only=dry_run)
    changed = 0
    try:
        targets = find_varchar_date_columns(con)
        if not targets:
            logger.info("%s — nothing to migrate", path)
            return 0

        for table, column in targets:
            bad = unparseable_values(con, table, column)
            if bad:
                raise ValueError(
                    f"{path}: {table}.{column} has {bad} value(s) that do not "
                    "parse as DATE — refusing to migrate. Inspect them first."
                )
            if dry_run:
                logger.info("[dry-run] would ALTER %s.%s VARCHAR → DATE", table, column)
                changed += 1
                continue

            t0 = time.perf_counter()
            con.execute(f"ALTER TABLE {table} ALTER COLUMN {column} TYPE DATE")
            logger.info(
                "ALTERed %s.%s → DATE in %.1fs", table, column, time.perf_counter() - t0
            )
            changed += 1
    finally:
        con.close()
    return changed


def main() -> int:
    setup_logging(log_dir=Path("logs"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Open read-only, report what would change, and validate that every "
        "value parses as a DATE. Changes nothing.",
    )
    parser.add_argument("--config", default="configs/data_sources.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    total = 0
    for key in _TIER_KEYS:
        path = config["storage"][key]
        logger.info("--- %s (%s) ---", key, path)
        total += migrate_file(path, args.dry_run)

    verb = "would change" if args.dry_run else "changed"
    logger.info("Done — %s %d column(s).", verb, total)
    return 0


if __name__ == "__main__":
    sys.exit(main())
