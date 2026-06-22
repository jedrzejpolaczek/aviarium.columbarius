import datetime
from dataclasses import dataclass
from typing import Literal

import duckdb

from src.data.cards.storage.base.storage import get_tables
from src.logger import get_logger

logger = get_logger(__name__)


@dataclass
class CheckResult:
    name: str
    layer: str
    status: Literal["PASS", "FAIL"]
    detail: str


def _check_table_has_rows(
    con: duckdb.DuckDBPyConnection, layer: str, table: str
) -> CheckResult:
    if table not in get_tables(con):
        return CheckResult(f"{table} exists", layer, "FAIL", f"table {table!r} not found")
    count: int = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]  # type: ignore[index]
    if count == 0:
        return CheckResult(f"{table} rows", layer, "FAIL", "0 rows")
    return CheckResult(f"{table} rows", layer, "PASS", f"{count} rows")


def _check_snapshot_date_today(
    con: duckdb.DuckDBPyConnection, table: str, today: datetime.date
) -> CheckResult:
    count: int = con.execute(
        f"SELECT COUNT(*) FROM {table} WHERE snapshot_date = ?", [today]
    ).fetchone()[0]  # type: ignore[index]
    if count == 0:
        return CheckResult(
            f"{table} freshness", "silver", "FAIL", f"no rows for {today}"
        )
    return CheckResult(
        f"{table} freshness", "silver", "PASS", f"{count} rows for {today}"
    )


def _check_no_nulls(
    con: duckdb.DuckDBPyConnection, layer: str, table: str, column: str
) -> CheckResult:
    count: int = con.execute(
        f"SELECT COUNT(*) FROM {table} WHERE {column} IS NULL"
    ).fetchone()[0]  # type: ignore[index]
    if count > 0:
        return CheckResult(
            f"{table}.{column} nulls", layer, "FAIL", f"{count} NULL values"
        )
    return CheckResult(f"{table}.{column} nulls", layer, "PASS", "no NULLs")


def _check_no_duplicate_canonical_uuid(
    con: duckdb.DuckDBPyConnection,
) -> CheckResult:
    count: int = con.execute("""
        SELECT COUNT(*) FROM (
            SELECT canonical_uuid
            FROM silver_cards
            WHERE uuid IS NOT NULL AND uuid = canonical_uuid
            GROUP BY canonical_uuid
            HAVING COUNT(*) > 1
        ) t
    """).fetchone()[0]  # type: ignore[index]
    if count > 0:
        return CheckResult(
            "silver_cards duplicate canonical_uuid",
            "silver",
            "FAIL",
            f"{count} duplicated canonical_uuid values",
        )
    return CheckResult(
        "silver_cards duplicate canonical_uuid", "silver", "PASS", "no duplicates"
    )


_ORACLE_ID_CONFLICT_THRESHOLD = 20


def _check_oracle_id_conflicts(con: duckdb.DuckDBPyConnection) -> CheckResult:
    count: int = con.execute("""
        SELECT COUNT(*) FROM (
            SELECT name
            FROM silver_cards
            WHERE oracle_id IS NOT NULL
            GROUP BY name
            HAVING COUNT(DISTINCT oracle_id) > 1
        ) t
    """).fetchone()[0]  # type: ignore[index]
    if count > _ORACLE_ID_CONFLICT_THRESHOLD:
        return CheckResult(
            "silver_cards oracle_id conflicts",
            "silver",
            "FAIL",
            f"{count} names map to multiple oracle_ids (threshold: {_ORACLE_ID_CONFLICT_THRESHOLD})",
        )
    return CheckResult(
        "silver_cards oracle_id conflicts",
        "silver",
        "PASS",
        f"{count} conflicts (within threshold of {_ORACLE_ID_CONFLICT_THRESHOLD})",
    )
