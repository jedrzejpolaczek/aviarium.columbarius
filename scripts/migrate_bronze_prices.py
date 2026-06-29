"""One-time migration: replace JSON price columns with scalar/EAV columns.

Usage:
    python scripts/migrate_bronze_prices.py \\
        --source data/bronze/cards_copy.duckdb \\
        --target data/bronze/cards.duckdb

The old tables in `target` are atomically replaced. `source` is opened
read-only via ATTACH and is not modified. Run this EXACTLY ONCE after
deploying the Bronze ingestion changes (Tasks R1-R5) but BEFORE deploying
Silver changes (Tasks 9-12).
"""

import argparse
import sys
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).parents[1]))


def migrate_mtgjson_prices(source_path: str, target_path: str) -> int:
    """Migrate bronze_mtgjson_prices_history from JSON paper column to EAV rows.

    Source table columns: uuid, snapshot_date, paper (JSON VARCHAR), mtgo (unused).
    Target table schema: (uuid, snapshot_date, retailer, tx_type, finish, price).

    Uses look-back semantics: for each (retailer, tx_type, finish) selects the
    price whose date key is the closest to snapshot_date without exceeding it.
    The entire transformation runs as a single SQL query inside DuckDB.

    Returns:
        Total number of EAV rows written.
    """
    _CHUNK_SQL = """
        INSERT INTO bronze_mtgjson_prices_history_new
        WITH source AS (
            SELECT uuid,
                   CAST(snapshot_date AS VARCHAR) AS snapshot_date,
                   paper::JSON AS j
            FROM src.bronze_mtgjson_prices_history
            WHERE CAST(snapshot_date AS VARCHAR) = ?
              AND paper IS NOT NULL
              AND paper != 'null'
              AND length(paper) > 2
        ),
        retailers AS (
            SELECT uuid, snapshot_date, j,
                   unnest(json_keys(j)) AS retailer
            FROM source
        ),
        tx_types AS (
            SELECT uuid, snapshot_date, retailer,
                   json_extract(j, '$.' || retailer) AS r_json,
                   unnest(json_keys(json_extract(j, '$.' || retailer))) AS tx_type
            FROM retailers
            WHERE json_type(json_extract(j, '$.' || retailer)) = 'OBJECT'
        ),
        tx_filtered AS (
            SELECT * FROM tx_types WHERE tx_type IN ('retail', 'buylist')
        ),
        finishes AS (
            SELECT uuid, snapshot_date, retailer, tx_type,
                   json_extract(r_json, '$.' || tx_type) AS tx_json,
                   unnest(json_keys(json_extract(r_json, '$.' || tx_type))) AS finish
            FROM tx_filtered
            WHERE json_type(json_extract(r_json, '$.' || tx_type)) = 'OBJECT'
        ),
        best AS (
            SELECT uuid, snapshot_date, retailer, tx_type, finish,
                   json_extract(tx_json, '$.' || finish) AS prices_dict,
                   list_max(
                       list_filter(
                           json_keys(json_extract(tx_json, '$.' || finish)),
                           k -> k <= snapshot_date
                       )
                   ) AS best_date
            FROM finishes
            WHERE json_type(json_extract(tx_json, '$.' || finish)) = 'OBJECT'
        )
        SELECT uuid, snapshot_date, retailer, tx_type, finish,
               TRY_CAST(json_extract_string(prices_dict, '$.' || best_date) AS FLOAT) AS price
        FROM best
        WHERE best_date IS NOT NULL
          AND price IS NOT NULL
    """

    tgt = duckdb.connect(target_path, read_only=False)
    try:
        tgt.execute("SET preserve_insertion_order = false")
        tgt.execute("SET threads = 4")
        tgt.execute("SET temp_directory = 'data/bronze/tmp'")
        tgt.execute(f"ATTACH '{source_path}' AS src (READ_ONLY)")
        tgt.execute("DROP TABLE IF EXISTS bronze_mtgjson_prices_history_new")
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

        dates = tgt.execute(
            "SELECT DISTINCT CAST(snapshot_date AS VARCHAR) FROM src.bronze_mtgjson_prices_history ORDER BY 1"
        ).fetchall()
        total_dates = len(dates)

        total_eav = 0
        for i, (snap_date,) in enumerate(dates, 1):
            tgt.execute(_CHUNK_SQL, [snap_date])
            if i % 10 == 0:
                tgt.execute("CHECKPOINT")
            total_eav = tgt.execute(
                "SELECT COUNT(*) FROM bronze_mtgjson_prices_history_new"
            ).fetchone()[0]
            print(f"\r  [{i}/{total_dates}] {snap_date} — {total_eav:,} EAV rows", end="", flush=True)

        print()

        count = tgt.execute(
            "SELECT COUNT(*) FROM bronze_mtgjson_prices_history_new"
        ).fetchone()[0]

        tgt.execute("DETACH src")
        tgt.execute("DROP TABLE IF EXISTS bronze_mtgjson_prices_history")
        tgt.execute(
            "ALTER TABLE bronze_mtgjson_prices_history_new"
            " RENAME TO bronze_mtgjson_prices_history"
        )
        tgt.execute("CHECKPOINT")
        return count
    finally:
        tgt.close()


def migrate_scryfall_prices(source_path: str, target_path: str) -> int:
    """Migrate bronze_scryfall_prices_history from JSON prices column to scalar columns.

    Source table columns: id, snapshot_date, prices (JSON VARCHAR).
    Target table schema: (id, snapshot_date, eur, eur_foil, usd, usd_foil, tix).

    Returns:
        Number of rows migrated.
    """
    tgt = duckdb.connect(target_path, read_only=False)
    try:
        tgt.execute(f"ATTACH '{source_path}' AS src (READ_ONLY)")
        tgt.execute("DROP TABLE IF EXISTS bronze_scryfall_prices_history_new")
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

        tgt.execute("""
            INSERT INTO bronze_scryfall_prices_history_new
            SELECT
                id,
                CAST(snapshot_date AS VARCHAR) AS snapshot_date,
                TRY_CAST(json_extract_string(CASE WHEN prices IS NOT NULL AND prices != 'null' THEN prices::JSON END, '$.eur')      AS FLOAT) AS eur,
                TRY_CAST(json_extract_string(CASE WHEN prices IS NOT NULL AND prices != 'null' THEN prices::JSON END, '$.eur_foil') AS FLOAT) AS eur_foil,
                TRY_CAST(json_extract_string(CASE WHEN prices IS NOT NULL AND prices != 'null' THEN prices::JSON END, '$.usd')      AS FLOAT) AS usd,
                TRY_CAST(json_extract_string(CASE WHEN prices IS NOT NULL AND prices != 'null' THEN prices::JSON END, '$.usd_foil') AS FLOAT) AS usd_foil,
                TRY_CAST(json_extract_string(CASE WHEN prices IS NOT NULL AND prices != 'null' THEN prices::JSON END, '$.tix')      AS FLOAT) AS tix
            FROM src.bronze_scryfall_prices_history
        """)

        count = tgt.execute(
            "SELECT COUNT(*) FROM bronze_scryfall_prices_history_new"
        ).fetchone()[0]

        tgt.execute("DETACH src")
        tgt.execute("DROP TABLE IF EXISTS bronze_scryfall_prices_history")
        tgt.execute(
            "ALTER TABLE bronze_scryfall_prices_history_new"
            " RENAME TO bronze_scryfall_prices_history"
        )
        tgt.execute("CHECKPOINT")
        return count
    finally:
        tgt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate Bronze price tables to scalar/EAV columns"
    )
    parser.add_argument("--source", required=True, help="Path to cards_copy.duckdb (backup)")
    parser.add_argument("--target", required=True, help="Path to live cards.duckdb")
    args = parser.parse_args()

    print(f"Migrating MTGJson prices: {args.source} → {args.target}")
    n = migrate_mtgjson_prices(args.source, args.target)
    print(f"  Migrated {n:,} EAV rows")

    print(f"Migrating Scryfall prices: {args.source} → {args.target}")
    n = migrate_scryfall_prices(args.source, args.target)
    print(f"  Migrated {n:,} rows")

    print("Migration complete.")


if __name__ == "__main__":
    main()
