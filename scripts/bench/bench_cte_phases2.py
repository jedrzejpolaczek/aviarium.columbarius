"""Time deduped (ROW_NUMBER) and full SQL phases."""
import time, duckdb
from pathlib import Path

BRONZE = "data/bronze/cards.duckdb"
SQL_RAW = Path("src/data/cards/storage/silver/sql/silver_cards.sql").read_text(encoding="utf-8")
STRIPPED = SQL_RAW.replace("CREATE OR REPLACE TABLE silver_cards AS\n", "", 1).strip()

def timed(label, sql):
    con = duckdb.connect(":memory:")
    con.execute(f"ATTACH '{BRONZE}' AS _bronze (READ_ONLY)")
    t0 = time.perf_counter()
    r = con.execute(sql).fetchone()[0]
    elapsed = time.perf_counter() - t0
    con.close()
    print(f"  {label:50s}  {r:>8,}  {elapsed:7.1f}s")

print("Phase timing (fresh :memory: each):\n")
print(f"  {'Label':50s}  {'rows':>8s}  {'time':>7s}")
print("  " + "-"*70)

# D: deduped only — inject a MATERIALIZED CTE marker to force evaluation
# Use CREATE TEMP TABLE to force DuckDB to fully evaluate the result
con = duckdb.connect(":memory:")
con.execute(f"ATTACH '{BRONZE}' AS _bronze (READ_ONLY)")

# Cut the SQL before "final AS (" and change the trailing ",)" to just ")"
# then SELECT count(*) FROM deduped
lines = STRIPPED.split("\n")
deduped_lines = []
for i, line in enumerate(lines):
    # Stop at "final AS (" (with optional leading whitespace or comment before)
    if line.strip().startswith("final AS ("):
        # Remove trailing comma from previous closing paren
        if deduped_lines and deduped_lines[-1].strip() == "":
            deduped_lines.pop()
        # Remove comment line before final AS
        while deduped_lines and deduped_lines[-1].strip().startswith("--"):
            deduped_lines.pop()
        # Remove trailing comma from the closing paren of deduped
        while deduped_lines and deduped_lines[-1].strip() == "":
            deduped_lines.pop()
        last = deduped_lines[-1].rstrip()
        if last.endswith(","):
            deduped_lines[-1] = last[:-1]
        break
    deduped_lines.append(line)

deduped_sql = "\n".join(deduped_lines) + "\nSELECT count(*) FROM deduped"

t0 = time.perf_counter()
r = con.execute(deduped_sql).fetchone()[0]
t_deduped = time.perf_counter() - t0
print(f"  {'D. deduped (ROW_NUMBER + WHERE scryfall_id IS NOT NULL)':50s}  {r:>8,}  {t_deduped:7.1f}s")
con.close()

# E: full SQL as COUNT*
timed("E. full SQL COUNT* (:memory:)", f"SELECT count(*) FROM ({STRIPPED})")
