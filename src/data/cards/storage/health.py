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
