"""Test: does file-based silver.duckdb connection slow down the query?

Runs the same full CTE chain (all columns, full join) as COUNT* (no write).
Compares:
  A) :memory: connection (control)
  B) file-based silver.duckdb + ATTACH bronze (mirrors real pipeline)
  C) file-based fresh temp.duckdb + ATTACH bronze (no existing data)
"""
import time, duckdb, os, tempfile
from pathlib import Path

BRONZE = "data/bronze/cards.duckdb"
SILVER = "data/silver/cards.duckdb"

FULL_SQL_PATH = Path("src/data/cards/storage/silver/sql/silver_cards.sql")
FULL_SQL_RAW = FULL_SQL_PATH.read_text(encoding="utf-8")

# Strip "CREATE OR REPLACE TABLE silver_cards AS\n" to get a pure SELECT
STRIPPED = FULL_SQL_RAW.replace("CREATE OR REPLACE TABLE silver_cards AS\n", "", 1).strip()
COUNT_SQL = f"SELECT count(*) FROM ({STRIPPED})"


def run(label, con_fn):
    con = con_fn()
    print(f"  {label}: connecting... ", end="", flush=True)
    t0 = time.perf_counter()
    result = con.execute(COUNT_SQL).fetchone()[0]
    elapsed = time.perf_counter() - t0
    con.close()
    m = elapsed / 60
    print(f"{result} rows in {m:.2f} min ({elapsed:.1f}s)")
    return elapsed


print("Silver cards full SQL — COUNT* only (no write)\n")

# A: :memory: (control)
def memory_con():
    con = duckdb.connect(":memory:")
    con.execute(f"ATTACH '{BRONZE}' AS _bronze (READ_ONLY)")
    return con

# B: file-based silver.duckdb + ATTACH bronze
def silver_file_con():
    con = duckdb.connect(SILVER, read_only=False)
    con.execute(f"ATTACH '{BRONZE}' AS _bronze (READ_ONLY)")
    return con

# C: fresh temp .duckdb + ATTACH bronze
TEMP_PATH = "data/temp_bench.duckdb"
def temp_file_con():
    if os.path.exists(TEMP_PATH):
        os.remove(TEMP_PATH)
    con = duckdb.connect(TEMP_PATH, read_only=False)
    con.execute(f"ATTACH '{BRONZE}' AS _bronze (READ_ONLY)")
    return con

t_mem = run("A) :memory:       ", memory_con)
t_sil = run("B) silver.duckdb  ", silver_file_con)
t_tmp = run("C) temp.duckdb    ", temp_file_con)

print(f"\n  :memory: vs silver.duckdb: {t_sil/t_mem:.1f}x slower")
print(f"  :memory: vs temp.duckdb:   {t_tmp/t_mem:.1f}x slower")

if os.path.exists(TEMP_PATH):
    os.remove(TEMP_PATH)
