"""
Benchmark: pinpoint whether the 161-min silver_cards build is SQL or write-to-file.

Tests:
  A. Full silver_cards.sql as COUNT(*) -> measures pure SQL cost (no write)
  B. Full silver_cards.sql -> CREATE TABLE in :memory:   -> measures SQL + memory write
  C. Full silver_cards.sql -> CREATE TABLE in test.duckdb -> measures SQL + file write

All via ATTACH of bronze.duckdb, same as the real pipeline.
"""

import time
import duckdb
from pathlib import Path

BRONZE = "data/bronze/cards.duckdb"
SILVER_SQL = (
    Path("src/data/cards/storage/silver/sql/silver_cards.sql")
    .read_text()
)

# Convert "CREATE OR REPLACE TABLE silver_cards AS" to "SELECT COUNT(*) FROM"
# by wrapping the full SQL in a subquery
COUNT_SQL = SILVER_SQL.replace(
    "CREATE OR REPLACE TABLE silver_cards AS\n",
    "",
    1,
).strip()
# COUNT_SQL is now just the WITH ... SELECT * FROM final
# We wrap it:
COUNT_SQL = f"SELECT count(*) FROM ({COUNT_SQL})"


def fmt(s):
    if s >= 60:
        return f"{s/60:.1f}m"
    return f"{s:.2f}s"


def run(label, sql, db_path=":memory:"):
    print(f"\n--- {label} ---")
    con = duckdb.connect(db_path)
    con.execute(f"ATTACH '{BRONZE}' AS _bronze (READ_ONLY)")
    t0 = time.perf_counter()
    result = con.execute(sql).fetchone()
    elapsed = time.perf_counter() - t0
    val = result[0] if result else "?"
    print(f"  [{fmt(elapsed):>8}]  result={val}")
    con.close()
    return elapsed


# ── Test A: pure SQL cost (COUNT, no write) ────────────────────────────────────
print("=== A: Full SQL as SELECT COUNT(*) ===")
print("(no write — measures join/CTE/window overhead only)")
run("Full silver_cards.sql -> COUNT(*)", COUNT_SQL)

# ── Test B: SQL + write to :memory: ───────────────────────────────────────────
print("\n=== B: Full SQL -> CREATE TABLE in :memory: ===")
print("(measures SQL + in-RAM write cost)")
run("Full silver_cards.sql -> :memory:", SILVER_SQL)

# ── Test C: SQL + write to file ────────────────────────────────────────────────
TEST_DB = "data/silver/_bench_test.duckdb"
Path(TEST_DB).unlink(missing_ok=True)
print(f"\n=== C: Full SQL -> CREATE TABLE in file ({TEST_DB}) ===")
print("(measures SQL + disk write cost)")
run("Full silver_cards.sql -> file", SILVER_SQL, db_path=TEST_DB)
Path(TEST_DB).unlink(missing_ok=True)

print("\nDone.")
