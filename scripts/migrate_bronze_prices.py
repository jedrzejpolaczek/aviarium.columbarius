"""One-time migration: replace JSON price columns with scalar/EAV columns.

Usage:
    python scripts/migrate_bronze_prices.py \\
        --source data/bronze/cards_copy.duckdb \\
        --target data/bronze/cards.duckdb

The old tables in `target` are atomically replaced. `source` is opened
read-only and is not modified. Run this EXACTLY ONCE after deploying the
Bronze ingestion changes (Tasks R1–R5) but BEFORE deploying Silver changes
(Tasks 9–12).
"""

import argparse
import json
import sys
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).parents[1]))


def migrate_mtgjson_prices(source_path: str, target_path: str) -> int:
    """Migrate bronze_mtgjson_prices_history from JSON paper column to EAV rows.

    Source table has columns: uuid, snapshot_date, paper (JSON VARCHAR), mtgo (unused).
    Target table schema: (uuid, snapshot_date, retailer, tx_type, finish, price).

    Uses look-back semantics (max date-key <= snapshot_date) when extracting
    prices from the nested paper dict.

    Returns:
        Total number of EAV rows written.
    """
    src = duckdb.connect(source_path, read_only=True)
    tgt = duckdb.connect(target_path, read_only=False)

    try:
        source_rows = src.execute(
            "SELECT uuid, snapshot_date, paper FROM bronze_mtgjson_prices_history"
        ).fetchall()

        tgt.execute("""
            CREATE TABLE bronze_mtgjson_prices_history_new (
                uuid          VARCHAR NOT NULL,
                snapshot_date VARCHAR NOT NULL,
                retailer      VARCHAR NOT NULL,
                tx_type       VARCHAR NOT NULL,
                finish        VARCHAR NOT NULL,
                price         FLOAT
            )
        """)

        eav_rows: list[list] = []
        for uuid, snapshot_date, paper_json in source_rows:
            paper = json.loads(paper_json) if isinstance(paper_json, str) else (paper_json or {})
            snap_str = str(snapshot_date)
            for retailer, retailer_data in paper.items():
                if not retailer_data:
                    continue
                for tx_type in ("buylist", "retail"):
                    listing = (retailer_data.get(tx_type)) or {}
                    for finish, prices in listing.items():
                        if not isinstance(prices, dict):
                            continue
                        candidates = {k: v for k, v in prices.items() if k <= snap_str}
                        if candidates:
                            eav_rows.append([
                                uuid, snap_str, retailer, tx_type, finish,
                                float(candidates[max(candidates)]),
                            ])

        if eav_rows:
            tgt.executemany(
                "INSERT INTO bronze_mtgjson_prices_history_new"
                " (uuid, snapshot_date, retailer, tx_type, finish, price)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                eav_rows,
            )

        tgt.execute("DROP TABLE IF EXISTS bronze_mtgjson_prices_history")
        tgt.execute(
            "ALTER TABLE bronze_mtgjson_prices_history_new"
            " RENAME TO bronze_mtgjson_prices_history"
        )
        tgt.execute("CHECKPOINT")
        return len(eav_rows)
    finally:
        src.close()
        tgt.close()


def migrate_scryfall_prices(source_path: str, target_path: str) -> int:
    """Migrate bronze_scryfall_prices_history from JSON prices column to scalar columns.

    Source table has columns: id, snapshot_date, prices (JSON VARCHAR).
    Target table schema: (id, snapshot_date, eur, eur_foil, usd, usd_foil, tix).

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
                usd_foil      FLOAT,
                tix           FLOAT
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
                float(prices["eur"])      if prices.get("eur")      is not None else None,
                float(prices["eur_foil"]) if prices.get("eur_foil") is not None else None,
                float(prices["usd"])      if prices.get("usd")      is not None else None,
                float(prices["usd_foil"]) if prices.get("usd_foil") is not None else None,
                float(prices["tix"])      if prices.get("tix")      is not None else None,
            ])

        if batch:
            tgt.executemany(
                "INSERT INTO bronze_scryfall_prices_history_new VALUES (?, ?, ?, ?, ?, ?, ?)",
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
    parser = argparse.ArgumentParser(description="Migrate Bronze price tables to scalar/EAV columns")
    parser.add_argument("--source", required=True, help="Path to cards_copy.duckdb (backup)")
    parser.add_argument("--target", required=True, help="Path to live cards.duckdb")
    args = parser.parse_args()

    print(f"Migrating MTGJson prices: {args.source} → {args.target}")
    n = migrate_mtgjson_prices(args.source, args.target)
    print(f"  Migrated {n} EAV rows")

    print(f"Migrating Scryfall prices: {args.source} → {args.target}")
    n = migrate_scryfall_prices(args.source, args.target)
    print(f"  Migrated {n} rows")

    print("Migration complete.")


if __name__ == "__main__":
    main()
