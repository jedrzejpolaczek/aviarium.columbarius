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


def _check_silver_prices_no_negative_eur(
    con: duckdb.DuckDBPyConnection, today: datetime.date
) -> CheckResult:
    count: int = con.execute(
        "SELECT COUNT(*) FROM silver_prices_history"
        " WHERE snapshot_date = ? AND eur <= 0",
        [today],
    ).fetchone()[0]  # type: ignore[index]
    if count > 0:
        return CheckResult(
            "silver_prices_history EUR <= 0",
            "silver",
            "FAIL",
            f"{count} rows with EUR <= 0 for {today}",
        )
    return CheckResult(
        "silver_prices_history EUR <= 0",
        "silver",
        "PASS",
        f"no invalid EUR prices for {today}",
    )


def _check_gold_ml_dataset_has_target(con: duckdb.DuckDBPyConnection) -> CheckResult:
    count: int = con.execute(
        "SELECT COUNT(*) FROM gold_ml_dataset WHERE target_price_7d IS NOT NULL"
    ).fetchone()[0]  # type: ignore[index]
    if count == 0:
        return CheckResult(
            "gold_ml_dataset target_price_7d",
            "gold",
            "FAIL",
            "target_price_7d is 100% NULL — no usable training rows",
        )
    return CheckResult(
        "gold_ml_dataset target_price_7d",
        "gold",
        "PASS",
        f"{count} rows with non-NULL target_price_7d",
    )


_BRONZE_TABLES = [
    "bronze_scryfall_cards",
    "bronze_mtgjson_cards",
    "bronze_mtgjson_prices_history",
]

_SILVER_TABLES = [
    "silver_cards",
    "silver_prices_history",
    "silver_language_prices_history",
    "silver_meta_history",
    "silver_format_staples_history",
    "silver_tournament_results_history",
]

_GOLD_TABLES = [
    "gold_card_features",
    "gold_price_features",
    "gold_language_premiums",
    "gold_demand_signals",
    "gold_format_staples",
    "gold_tournament_signals",
    "gold_ml_dataset",
]

_SILVER_FRESHNESS_TABLES = [
    "silver_prices_history",
    "silver_language_prices_history",
    "silver_format_staples_history",
]

_SILVER_QUALITY_NULL_COLUMNS = ["name", "set_code", "collector_number"]


def run_health_checks(
    bronze_path: str,
    silver_path: str,
    gold_path: str,
    today: datetime.date,
) -> list[CheckResult]:
    results: list[CheckResult] = []

    bronze_con = duckdb.connect(str(bronze_path), read_only=True)
    silver_con = duckdb.connect(str(silver_path), read_only=True)
    gold_con = duckdb.connect(str(gold_path), read_only=True)

    try:
        bronze_structure = [
            _check_table_has_rows(bronze_con, "bronze", t) for t in _BRONZE_TABLES
        ]
        results.extend(bronze_structure)

        silver_structure = [
            _check_table_has_rows(silver_con, "silver", t) for t in _SILVER_TABLES
        ]
        results.extend(silver_structure)

        gold_structure = [
            _check_table_has_rows(gold_con, "gold", t) for t in _GOLD_TABLES
        ]
        results.extend(gold_structure)

        if all(r.status == "PASS" for r in silver_structure):
            for t in _SILVER_FRESHNESS_TABLES:
                results.append(_check_snapshot_date_today(silver_con, t, today))
            for col in _SILVER_QUALITY_NULL_COLUMNS:
                results.append(_check_no_nulls(silver_con, "silver", "silver_cards", col))
            results.append(_check_no_duplicate_canonical_uuid(silver_con))
            results.append(_check_oracle_id_conflicts(silver_con))
            results.append(_check_silver_prices_no_negative_eur(silver_con, today))

        if all(r.status == "PASS" for r in gold_structure):
            results.append(_check_gold_ml_dataset_has_target(gold_con))

    finally:
        bronze_con.close()
        silver_con.close()
        gold_con.close()

    for r in results:
        msg = f"[{r.status}] {r.layer} | {r.name} — {r.detail}"
        if r.status == "PASS":
            logger.info(msg)
        else:
            logger.error(msg)

    failed = sum(1 for r in results if r.status == "FAIL")
    passed = sum(1 for r in results if r.status == "PASS")
    logger.info("Health check complete: %d passed, %d failed", passed, failed)

    if failed:
        raise SystemExit(1)

    return results
