"""One-time migration: replace JSON price columns with scalar FLOAT columns.

Usage:
    python scripts/migrate_bronze_prices.py \\
        --source data/bronze/cards_copy.duckdb \\
        --target data/bronze/cards.duckdb

The old tables in `target` are atomically replaced. `source` is opened
read-only and is not modified. Run this EXACTLY ONCE after deploying the
Bronze ingestion changes (Tasks 1–6) but BEFORE deploying Silver changes
(Tasks 9–12).
"""

import argparse
import json
import sys
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).parents[1]))
from src.data.cards.storage.bronze.storage import (
    _MTGJSON_PRICE_MAP,
    _extract_mtgjson_scalar_prices,
)


def migrate_mtgjson_prices(source_path: str, target_path: str) -> int:
    """Migrate bronze_mtgjson_prices_history from JSON paper column to scalar columns.

    Returns:
        Number of rows migrated.
    """
    src = duckdb.connect(source_path, read_only=True)
    tgt = duckdb.connect(target_path, read_only=False)

    try:
        rows = src.execute(
            "SELECT uuid, snapshot_date, paper FROM bronze_mtgjson_prices_history"
        ).fetchall()

        scalar_col_defs = ", ".join(f"{col} FLOAT" for col in _MTGJSON_PRICE_MAP)
        tgt.execute(f"""
            CREATE TABLE bronze_mtgjson_prices_history_new (
                uuid          VARCHAR NOT NULL,
                snapshot_date VARCHAR NOT NULL,
                {scalar_col_defs}
            )
        """)

        col_names = list(_MTGJSON_PRICE_MAP.keys())
        placeholders = ", ".join(["?"] * (2 + len(col_names)))
        insert_sql = (
            f"INSERT INTO bronze_mtgjson_prices_history_new"
            f" (uuid, snapshot_date, {', '.join(col_names)})"
            f" VALUES ({placeholders})"
        )

        batch = []
        for uuid, snapshot_date, paper_json in rows:
            paper = json.loads(paper_json) if isinstance(paper_json, str) else paper_json
            scalars = _extract_mtgjson_scalar_prices(paper, str(snapshot_date))
            batch.append([uuid, str(snapshot_date)] + [scalars[col] for col in col_names])

        if batch:
            tgt.executemany(insert_sql, batch)

        tgt.execute("DROP TABLE IF EXISTS bronze_mtgjson_prices_history")
        tgt.execute(
            "ALTER TABLE bronze_mtgjson_prices_history_new"
            " RENAME TO bronze_mtgjson_prices_history"
        )
        tgt.execute("CHECKPOINT")

        return len(rows)
    finally:
        src.close()
        tgt.close()


def migrate_scryfall_prices(source_path: str, target_path: str) -> int:
    """Migrate bronze_scryfall_prices_history from JSON prices column to scalar columns.

    Returns:
        Number of rows migrated.
    """
    src = duckdb.connect(source_path, read_only=True)
    tgt = duckdb.connect(target_path, read_only=False)

    try:
        rows = src.execute(
            "SELECT id, snapshot_date, prices FROM bronze_scryfall_prices_history"
        ).fetchall()

        tgt.execute("""
            CREATE TABLE bronze_scryfall_prices_history_new (
                id            VARCHAR NOT NULL,
                snapshot_date VARCHAR NOT NULL,
                eur           FLOAT,
                eur_foil      FLOAT,
                usd           FLOAT,
                usd_foil      FLOAT
            )
        """)

        batch = []
        for scryfall_id, snapshot_date, prices_json in rows:
            if prices_json is None:
                prices: dict = {}
            elif isinstance(prices_json, str):
                prices = json.loads(prices_json)
            else:
                prices = prices_json

            batch.append([
                scryfall_id,
                str(snapshot_date),
                float(prices["eur"]) if prices.get("eur") is not None else None,
                float(prices["eur_foil"]) if prices.get("eur_foil") is not None else None,
                float(prices["usd"]) if prices.get("usd") is not None else None,
                float(prices["usd_foil"]) if prices.get("usd_foil") is not None else None,
            ])

        if batch:
            tgt.executemany(
                "INSERT INTO bronze_scryfall_prices_history_new VALUES (?, ?, ?, ?, ?, ?)",
                batch,
            )

        tgt.execute("DROP TABLE IF EXISTS bronze_scryfall_prices_history")
        tgt.execute(
            "ALTER TABLE bronze_scryfall_prices_history_new"
            " RENAME TO bronze_scryfall_prices_history"
        )
        tgt.execute("CHECKPOINT")

        return len(rows)
    finally:
        src.close()
        tgt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate Bronze price tables to scalar columns")
    parser.add_argument("--source", required=True, help="Path to cards_copy.duckdb (backup)")
    parser.add_argument("--target", required=True, help="Path to live cards.duckdb")
    args = parser.parse_args()

    print(f"Migrating MTGJson prices: {args.source} → {args.target}")
    n = migrate_mtgjson_prices(args.source, args.target)
    print(f"  Migrated {n} rows")

    print(f"Migrating Scryfall prices: {args.source} → {args.target}")
    n = migrate_scryfall_prices(args.source, args.target)
    print(f"  Migrated {n} rows")

    print("Migration complete.")


if __name__ == "__main__":
    main()
