"""Isolate and time ONLY the with_canonical join step.
Current (NL) vs Fixed (Hash) — same data, same join result, different algorithm.
"""
import time, duckdb

BRONZE = "data/bronze/cards.duckdb"

CURRENT_SQL = """
WITH
mtgjson_filtered AS (
    SELECT * FROM _bronze.bronze_mtgjson_cards
    WHERE COALESCE(is_online_only::BOOLEAN, false) = false
      AND COALESCE(is_funny::BOOLEAN,       false) = false
      AND COALESCE(is_oversized::BOOLEAN,   false) = false
),
mtgjson AS (
    SELECT
        TRIM(uuid)                                             AS uuid,
        json_extract_string(identifiers, '$.scryfall_id')     AS scryfall_id,
        UPPER(TRIM(set_code))                                 AS set_code,
        NULLIF(TRIM(number), '_')                             AS collector_number
    FROM mtgjson_filtered
),
scryfall_filtered AS (
    SELECT id FROM _bronze.bronze_scryfall_cards
    WHERE COALESCE(digital::BOOLEAN,   false) = false
      AND COALESCE(oversized::BOOLEAN, false) = false
      AND layout NOT IN ('token', 'double_faced_token', 'emblem')
),
scryfall AS (SELECT TRIM(id) AS id FROM scryfall_filtered),
joined AS (
    SELECT
        m.uuid,
        COALESCE(m.scryfall_id, s.id) AS scryfall_id,
        m.set_code,
        m.collector_number
    FROM mtgjson m FULL OUTER JOIN scryfall s ON m.scryfall_id = s.id
),
canonical_map AS (
    SELECT set_code, collector_number, MIN(uuid) AS uuid
    FROM mtgjson
    WHERE uuid IS NOT NULL AND set_code IS NOT NULL AND collector_number IS NOT NULL
    GROUP BY set_code, collector_number
),
with_canonical AS (
    SELECT
        j.*,
        CASE WHEN j.uuid IS NOT NULL THEN j.uuid ELSE cm.uuid END AS canonical_uuid
    FROM joined j
    LEFT JOIN canonical_map cm
           ON j.uuid IS NULL                -- <-- NL JOIN trigger
          AND j.set_code        = cm.set_code
          AND j.collector_number = cm.collector_number
)
SELECT count(*), count(canonical_uuid) FROM with_canonical
"""

FIXED_SQL = CURRENT_SQL.replace(
    "ON j.uuid IS NULL                -- <-- NL JOIN trigger\n"
    "          AND j.set_code        = cm.set_code\n"
    "          AND j.collector_number = cm.collector_number",
    "ON j.set_code        = cm.set_code\n"
    "          AND j.collector_number = cm.collector_number"
)

def run(label, sql):
    con = duckdb.connect(":memory:")
    con.execute(f"ATTACH '{BRONZE}' AS _bronze (READ_ONLY)")
    t0 = time.perf_counter()
    rows, canonical = con.execute(sql).fetchone()
    elapsed = time.perf_counter() - t0
    con.close()
    m = elapsed / 60
    print(f"  {label:10s}  {m:6.2f} min  ({rows} rows, {canonical} with canonical_uuid)")
    return elapsed

print("Timing ONLY the with_canonical CTE (no write, no window, no format_count):\n")
t_current = run("CURRENT", CURRENT_SQL)
t_fixed   = run("FIXED",   FIXED_SQL)

ratio = t_current / t_fixed if t_fixed > 0 else float('inf')
print(f"\n  Speedup: {ratio:.0f}x  ({t_current:.1f}s -> {t_fixed:.1f}s)")
