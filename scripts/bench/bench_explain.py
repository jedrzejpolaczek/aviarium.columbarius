"""Check DuckDB query plan for silver_cards.sql (no data, just the plan)."""
import duckdb
from pathlib import Path

BRONZE = "data/bronze/cards.duckdb"
SILVER_SQL_RAW = (Path("src/data/cards/storage/silver/sql/silver_cards.sql").read_text()
                  .replace("CREATE OR REPLACE TABLE silver_cards AS\n", "", 1)
                  .strip())
COUNT_SQL = f"SELECT count(*) FROM ({SILVER_SQL_RAW})"

con = duckdb.connect(":memory:")
con.execute(f"ATTACH '{BRONZE}' AS _bronze (READ_ONLY)")

print("=== EXPLAIN (physical plan) ===")
plan = con.execute(f"EXPLAIN {COUNT_SQL}").fetchall()
for row in plan:
    print(row[1].encode("ascii", errors="replace").decode("ascii"))

con.close()
