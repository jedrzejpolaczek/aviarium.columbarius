"""Verify that removing 'uuid IS NULL' from with_canonical ON clause
switches DuckDB from BLOCKWISE_NL_JOIN to HASH_JOIN."""
import sys
import duckdb
from pathlib import Path

BRONZE = "data/bronze/cards.duckdb"

# --- CURRENT (buggy) version ---
CURRENT_WITH_CANONICAL = """
with_canonical AS (
    SELECT
        j.*,
        CASE
            WHEN j.uuid IS NOT NULL THEN j.uuid
            ELSE cm.uuid
        END                                                    AS canonical_uuid
    FROM joined j
    LEFT JOIN canonical_map cm
           ON j.uuid IS NULL
          AND j.set_code        = cm.set_code
          AND j.collector_number = cm.collector_number
)"""

# --- FIXED version: remove j.uuid IS NULL from ON clause ---
FIXED_WITH_CANONICAL = """
with_canonical AS (
    SELECT
        j.*,
        CASE
            WHEN j.uuid IS NOT NULL THEN j.uuid
            ELSE cm.uuid
        END                                                    AS canonical_uuid
    FROM joined j
    LEFT JOIN canonical_map cm
           ON j.set_code        = cm.set_code
          AND j.collector_number = cm.collector_number
)"""

BASE_SQL = (Path("src/data/cards/storage/silver/sql/silver_cards.sql")
            .read_text()
            .replace("CREATE OR REPLACE TABLE silver_cards AS\n", "", 1)
            .strip())


def get_join_type(sql_variant: str, label: str) -> str:
    sql = BASE_SQL.replace(CURRENT_WITH_CANONICAL, sql_variant)
    count_sql = f"SELECT count(*) FROM ({sql})"
    con = duckdb.connect(":memory:")
    con.execute(f"ATTACH '{BRONZE}' AS _bronze (READ_ONLY)")
    plan_rows = con.execute(f"EXPLAIN {count_sql}").fetchall()
    plan = "\n".join(r[1] for r in plan_rows)
    con.close()

    # Find the join type for with_canonical (the LEFT JOIN between joined and canonical_map)
    lines = plan.split("\n")
    join_types = []
    for i, line in enumerate(lines):
        if "NL_JOIN" in line or "HASH_JOIN" in line or "BLOCKWISE" in line:
            # grab the surrounding context
            context = lines[max(0, i-1):i+5]
            join_types.append(" | ".join(l.strip() for l in context if l.strip()))

    safe = lambda s: s.encode("ascii", errors="replace").decode("ascii")
    print(f"\n=== {label} ===")
    for jt in join_types:
        print(f"  JOIN: {safe(jt)}")
    return plan


plan_current = get_join_type(CURRENT_WITH_CANONICAL, "CURRENT (has j.uuid IS NULL in ON)")
plan_fixed   = get_join_type(FIXED_WITH_CANONICAL,   "FIXED   (removed j.uuid IS NULL)")

print("\n=== SUMMARY ===")
if "BLOCKWISE_NL_JOIN" in plan_current and "HASH_JOIN" in plan_fixed:
    print("  CONFIRMED: fix converts BLOCKWISE_NL_JOIN -> HASH_JOIN")
elif "HASH_JOIN" in plan_fixed:
    print("  HASH_JOIN present in fixed version")
else:
    print("  WARNING: unexpected join types, review full plan")
