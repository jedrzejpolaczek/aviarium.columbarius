"""
Benchmark: pinpoint which step of _SILVER_CARDS_SQL causes ~161 min runtime.

Strategy: run each CTE layer incrementally on bronze_con (direct, no ATTACH),
measuring time added by each step.

Run while Silver pipeline is in Step 1 -- reads bronze.duckdb READ_ONLY.
"""

import sys
import time
import duckdb

BRONZE = "data/bronze/cards.duckdb"


def fmt(s):
    if s >= 60:
        return f"{s/60:.1f}m"
    return f"{s:.2f}s"


def bench(label, con, sql):
    t0 = time.perf_counter()
    result = con.execute(sql).fetchone()
    elapsed = time.perf_counter() - t0
    val = result[0] if result else "?"
    print(f"  [{fmt(elapsed):>8}]  {label}  ({val} rows)")
    return elapsed


con = duckdb.connect(BRONZE, read_only=True)
print(f"Connected to bronze ({BRONZE})")
print()

# ── Step 1: Raw table sizes ────────────────────────────────────────────────────
print("=== RAW TABLE SCANS ===")
bench("COUNT bronze_scryfall_cards", con, "SELECT count(*) FROM bronze_scryfall_cards")
bench("COUNT bronze_mtgjson_cards",  con, "SELECT count(*) FROM bronze_mtgjson_cards")
print()

# ── Step 2: Single-table CTEs with all transforms ──────────────────────────────
print("=== SINGLE-TABLE CTEs (all column transforms) ===")
bench("mtgjson_filtered + mtgjson CTE", con, """
WITH
mtgjson_filtered AS (
    SELECT * FROM bronze_mtgjson_cards
    WHERE COALESCE(is_online_only::BOOLEAN, false) = false
      AND COALESCE(is_funny::BOOLEAN,       false) = false
      AND COALESCE(is_oversized::BOOLEAN,   false) = false
),
mtgjson AS (
    SELECT
        TRIM(uuid)                                             AS uuid,
        json_extract_string(identifiers, '$.scryfall_id')     AS scryfall_id,
        TRIM(name)                                            AS name,
        COALESCE(TRIM(ascii_name), TRIM(name))                AS ascii_name,
        UPPER(TRIM(set_code))                                 AS set_code,
        NULLIF(TRIM(number), '_')                             AS collector_number,
        upper(left(TRIM(language), 1)) || lower(substr(TRIM(language), 2)) AS language,
        LOWER(TRIM(layout))                                   AS layout,
        NULLIF(UPPER(TRIM(mana_cost)), '_')                   AS mana_cost,
        NULLIF(TRIM(text), '_')                               AS text,
        LOWER(TRIM(rarity))                                   AS rarity,
        LOWER(TRIM(border_color))                             AS border_color,
        COALESCE(is_reprint::BOOLEAN,      false)             AS is_reprint,
        COALESCE(is_reserved::BOOLEAN,     false)             AS is_reserved,
        COALESCE(is_promo::BOOLEAN,        false)             AS is_promo,
        TRY_CAST(mana_value AS DOUBLE)                        AS mana_value,
        TRY_CAST(edhrec_rank AS DOUBLE)                       AS edhrec_rank,
        legalities,
        list_transform(COALESCE(colors,         '[]')::VARCHAR[], x -> UPPER(x))   AS colors,
        list_transform(COALESCE(color_identity, '[]')::VARCHAR[], x -> UPPER(x))   AS color_identity,
        list_transform(COALESCE(keywords,       '[]')::VARCHAR[], x -> upper(left(x,1)) || lower(substr(x,2))) AS keywords,
        list_transform(COALESCE(finishes,       '[]')::VARCHAR[], x -> LOWER(x))   AS finishes,
        list_transform(COALESCE(availability,   '[]')::VARCHAR[], x -> LOWER(x))   AS availability
    FROM mtgjson_filtered
)
SELECT count(*) FROM mtgjson
""")

bench("scryfall_filtered + scryfall CTE", con, """
WITH
scryfall_filtered AS (
    SELECT * FROM bronze_scryfall_cards
    WHERE COALESCE(digital::BOOLEAN,   false) = false
      AND COALESCE(oversized::BOOLEAN, false) = false
      AND COALESCE(layout, '') NOT IN ('token', 'double_faced_token', 'emblem')
),
scryfall AS (
    SELECT
        TRIM(id)                                              AS id,
        TRIM(oracle_id)                                       AS oracle_id,
        TRIM(name)                                            AS name,
        LOWER(TRIM(layout))                                   AS layout,
        TRY_CAST(cmc AS DOUBLE)                               AS cmc,
        TRIM(type_line)                                       AS type_line,
        UPPER(TRIM(set))                                      AS set_code,
        TRIM(set_name)                                        AS set_name,
        LOWER(TRIM(set_type))                                 AS set_type,
        LOWER(TRIM(rarity))                                   AS rarity,
        TRIM(lang)                                            AS lang,
        COALESCE(reserved::BOOLEAN, false)                    AS is_reserved,
        COALESCE(reprint::BOOLEAN,  false)                    AS is_reprint,
        legalities,
        list_transform(COALESCE(colors,         '[]')::VARCHAR[], x -> UPPER(x))   AS colors,
        list_transform(COALESCE(color_identity, '[]')::VARCHAR[], x -> UPPER(x))   AS color_identity,
        list_transform(COALESCE(keywords,       '[]')::VARCHAR[], x -> upper(left(x,1)) || lower(substr(x,2))) AS keywords,
        list_transform(COALESCE(finishes,       '[]')::VARCHAR[], x -> LOWER(x))   AS finishes,
        list_transform(COALESCE(games,          '[]')::VARCHAR[], x -> LOWER(x))   AS games,
        COALESCE(artist_ids,     '[]')::VARCHAR[]                                   AS artist_ids,
        COALESCE(all_parts,      '[]')::VARCHAR[]                                   AS all_parts,
        COALESCE(card_faces,     '[]')::VARCHAR[]                                   AS card_faces
    FROM scryfall_filtered
)
SELECT count(*) FROM scryfall
""")
print()

# ── Step 3: FULL OUTER JOIN ────────────────────────────────────────────────────
print("=== FULL OUTER JOIN ===")
bench("FULL OUTER JOIN (minimal columns)", con, """
WITH
mtgjson_filtered AS (
    SELECT uuid, identifiers, name FROM bronze_mtgjson_cards
    WHERE COALESCE(is_online_only::BOOLEAN, false)=false
      AND COALESCE(is_funny::BOOLEAN,false)=false
      AND COALESCE(is_oversized::BOOLEAN,false)=false
),
mtgjson AS (
    SELECT TRIM(uuid) AS uuid,
           json_extract_string(identifiers,'$.scryfall_id') AS scryfall_id,
           TRIM(name) AS name
    FROM mtgjson_filtered
),
scryfall_filtered AS (
    SELECT id, name FROM bronze_scryfall_cards
    WHERE COALESCE(digital::BOOLEAN,false)=false
      AND COALESCE(oversized::BOOLEAN,false)=false
      AND COALESCE(layout,'') NOT IN ('token','double_faced_token','emblem')
),
scryfall AS (SELECT TRIM(id) AS id, TRIM(name) AS name FROM scryfall_filtered),
joined AS (
    SELECT COALESCE(m.scryfall_id, s.id) AS scryfall_id, m.uuid
    FROM mtgjson m FULL OUTER JOIN scryfall s ON m.scryfall_id = s.id
)
SELECT count(*) FROM joined WHERE scryfall_id IS NOT NULL
""")
print()

# ── Step 4: Window function ────────────────────────────────────────────────────
print("=== ROW_NUMBER WINDOW FUNCTION ===")
bench("ROW_NUMBER OVER (PARTITION BY scryfall_id)", con, """
WITH
mtgjson_filtered AS (
    SELECT uuid, identifiers, name FROM bronze_mtgjson_cards
    WHERE COALESCE(is_online_only::BOOLEAN,false)=false
      AND COALESCE(is_funny::BOOLEAN,false)=false
      AND COALESCE(is_oversized::BOOLEAN,false)=false
),
mtgjson AS (
    SELECT TRIM(uuid) AS uuid,
           json_extract_string(identifiers,'$.scryfall_id') AS scryfall_id
    FROM mtgjson_filtered
),
scryfall_filtered AS (
    SELECT id FROM bronze_scryfall_cards
    WHERE COALESCE(digital::BOOLEAN,false)=false
      AND COALESCE(oversized::BOOLEAN,false)=false
      AND COALESCE(layout,'') NOT IN ('token','double_faced_token','emblem')
),
scryfall AS (SELECT TRIM(id) AS id FROM scryfall_filtered),
joined AS (
    SELECT COALESCE(m.scryfall_id, s.id) AS scryfall_id, m.uuid
    FROM mtgjson m FULL OUTER JOIN scryfall s ON m.scryfall_id = s.id
),
deduped AS (
    SELECT * EXCLUDE rn FROM (
        SELECT *,
               ROW_NUMBER() OVER (
                   PARTITION BY scryfall_id
                   ORDER BY CASE WHEN uuid IS NOT NULL THEN 0 ELSE 1 END, uuid
               ) AS rn
        FROM joined
        WHERE scryfall_id IS NOT NULL
    ) WHERE rn = 1
)
SELECT count(*) FROM deduped
""")
print()

# ── Step 5: format_count (20x json_extract_string) ────────────────────────────
print("=== 20x json_extract_string for format_count ===")
bench("format_count on 530K scryfall rows", con, """
SELECT count(*)
FROM (
    SELECT
        (
          COALESCE((LOWER(json_extract_string(legalities,'$.standard'))    ='legal')::INT, 0) +
          COALESCE((LOWER(json_extract_string(legalities,'$.pioneer'))     ='legal')::INT, 0) +
          COALESCE((LOWER(json_extract_string(legalities,'$.modern'))      ='legal')::INT, 0) +
          COALESCE((LOWER(json_extract_string(legalities,'$.legacy'))      ='legal')::INT, 0) +
          COALESCE((LOWER(json_extract_string(legalities,'$.vintage'))     ='legal')::INT, 0) +
          COALESCE((LOWER(json_extract_string(legalities,'$.commander'))   ='legal')::INT, 0) +
          COALESCE((LOWER(json_extract_string(legalities,'$.oathbreaker')) ='legal')::INT, 0) +
          COALESCE((LOWER(json_extract_string(legalities,'$.brawl'))       ='legal')::INT, 0) +
          COALESCE((LOWER(json_extract_string(legalities,'$.historicbrawl'))='legal')::INT, 0) +
          COALESCE((LOWER(json_extract_string(legalities,'$.alchemy'))     ='legal')::INT, 0) +
          COALESCE((LOWER(json_extract_string(legalities,'$.explorer'))    ='legal')::INT, 0) +
          COALESCE((LOWER(json_extract_string(legalities,'$.historic'))    ='legal')::INT, 0) +
          COALESCE((LOWER(json_extract_string(legalities,'$.timeless'))    ='legal')::INT, 0) +
          COALESCE((LOWER(json_extract_string(legalities,'$.gladiator'))   ='legal')::INT, 0) +
          COALESCE((LOWER(json_extract_string(legalities,'$.penny'))       ='legal')::INT, 0) +
          COALESCE((LOWER(json_extract_string(legalities,'$.paupercommander'))='legal')::INT, 0) +
          COALESCE((LOWER(json_extract_string(legalities,'$.pauper'))      ='legal')::INT, 0) +
          COALESCE((LOWER(json_extract_string(legalities,'$.predh'))       ='legal')::INT, 0) +
          COALESCE((LOWER(json_extract_string(legalities,'$.premodern'))   ='legal')::INT, 0) +
          COALESCE((LOWER(json_extract_string(legalities,'$.oldschool'))   ='legal')::INT, 0)
        ) AS format_count
    FROM bronze_scryfall_cards
    WHERE COALESCE(digital::BOOLEAN,false)=false
      AND COALESCE(oversized::BOOLEAN,false)=false
      AND COALESCE(layout,'') NOT IN ('token','double_faced_token','emblem')
)
WHERE format_count > 0
""")
print()

# ── Step 6: Write cost -- CREATE TABLE from minimal SELECT ─────────────────────
print("=== WRITE COST (needs writable :memory: DB) ===")
con_mem = duckdb.connect(":memory:")
con_mem.execute(f"ATTACH '{BRONZE}' AS _bronze (READ_ONLY)")

bench("CREATE TABLE (minimal) from FULL OUTER JOIN", con_mem, """
CREATE OR REPLACE TABLE test_silver_cards AS
WITH
mtgjson_filtered AS (
    SELECT uuid, identifiers, name, set_code, number, language FROM _bronze.bronze_mtgjson_cards
    WHERE COALESCE(is_online_only::BOOLEAN,false)=false
      AND COALESCE(is_funny::BOOLEAN,false)=false
      AND COALESCE(is_oversized::BOOLEAN,false)=false
),
mtgjson AS (
    SELECT TRIM(uuid) AS uuid,
           json_extract_string(identifiers,'$.scryfall_id') AS scryfall_id,
           TRIM(name) AS name, UPPER(TRIM(set_code)) AS set_code,
           NULLIF(TRIM(number),'_') AS collector_number
    FROM mtgjson_filtered
),
scryfall_filtered AS (
    SELECT id, name, set, rarity, lang FROM _bronze.bronze_scryfall_cards
    WHERE COALESCE(digital::BOOLEAN,false)=false
      AND COALESCE(oversized::BOOLEAN,false)=false
      AND COALESCE(layout,'') NOT IN ('token','double_faced_token','emblem')
),
scryfall AS (SELECT TRIM(id) AS id, TRIM(name) AS name, UPPER(TRIM(set)) AS set_code,
             LOWER(TRIM(rarity)) AS rarity, TRIM(lang) AS lang FROM scryfall_filtered),
joined AS (
    SELECT COALESCE(m.scryfall_id,s.id) AS scryfall_id,
           m.uuid, COALESCE(m.name,s.name) AS name
    FROM mtgjson m FULL OUTER JOIN scryfall s ON m.scryfall_id=s.id
),
deduped AS (
    SELECT * EXCLUDE rn FROM (
        SELECT *, ROW_NUMBER() OVER (PARTITION BY scryfall_id
               ORDER BY CASE WHEN uuid IS NOT NULL THEN 0 ELSE 1 END, uuid) AS rn
        FROM joined WHERE scryfall_id IS NOT NULL
    ) WHERE rn=1
)
SELECT * FROM deduped
""")
bench("COUNT rows in test_silver_cards", con_mem, "SELECT count(*) FROM test_silver_cards")
con_mem.close()
print()

con.close()
print("Done.")
