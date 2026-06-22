"""Check EXPLAIN for the full silver_cards.sql with the current buggy $.scryfallId.
Specifically: how many NL joins are there? Is the FULL OUTER JOIN also NL?
"""
import duckdb
from pathlib import Path

BRONZE = "data/bronze/cards.duckdb"
SQL_RAW = Path("src/data/cards/storage/silver/sql/silver_cards.sql").read_text(encoding="utf-8")
STRIPPED = SQL_RAW.replace("CREATE OR REPLACE TABLE silver_cards AS\n", "", 1).strip()
COUNT_SQL = f"SELECT count(*) FROM ({STRIPPED})"

con = duckdb.connect(":memory:")
con.execute(f"ATTACH '{BRONZE}' AS _bronze (READ_ONLY)")
plan = con.execute(f"EXPLAIN {COUNT_SQL}").fetchall()
con.close()

safe = lambda s: s.encode("ascii", errors="replace").decode("ascii")
lines = "\n".join(r[1] for r in plan).split("\n")

print(f"Total plan lines: {len(lines)}\n")

# Find all join operators
print("=== All JOIN operators in plan ===")
for i, line in enumerate(lines):
    if any(x in line for x in ("JOIN", "join")):
        ctx = lines[max(0,i-1):i+3]
        for c in ctx:
            if c.strip():
                print(f"  L{i:3d}: {safe(c)}")
        print()

# Print section around BLOCKWISE_NL_JOIN
print("=== BLOCKWISE_NL_JOIN occurrences ===")
for i, line in enumerate(lines):
    if "BLOCKWISE_NL_JOIN" in line:
        ctx = lines[max(0,i-2):i+4]
        for c in ctx:
            if c.strip():
                print(f"  L{i:3d}: {safe(c)}")
        print()
