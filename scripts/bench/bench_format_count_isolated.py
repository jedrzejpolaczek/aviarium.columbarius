"""Time the format_count step in isolation.
Creates a minimal query: just one JOIN + deduped equivalent + 24 json_extract.
Compare with 0 json_extract (just deduped) to isolate json_extract cost.
"""
import time, duckdb

BRONZE = "data/bronze/cards.duckdb"

# We already know deduped = 78.3s.
# Here we measure: deduped_rows + 1 json_extract vs 24 json_extract, using bronze scryfall legalities

con = duckdb.connect(":memory:")
con.execute(f"ATTACH '{BRONZE}' AS _bronze (READ_ONLY)")

# Baseline: count rows of scryfall legalities, no extraction
t0 = time.perf_counter()
n = con.execute("""
    SELECT count(*) FROM (
        SELECT id, legalities
        FROM _bronze.bronze_scryfall_cards
        WHERE COALESCE(digital::BOOLEAN, false) = false
          AND COALESCE(oversized::BOOLEAN, false) = false
          AND layout NOT IN ('token', 'double_faced_token', 'emblem')
    )
""").fetchone()[0]
t_scan = time.perf_counter() - t0
print(f"Scan legalities column ({n:,} rows):    {t_scan:.2f}s")

# 1 json_extract
t0 = time.perf_counter()
con.execute("""
    SELECT count(*), sum((LOWER(json_extract_string(legalities,'$.standard'))='legal')::INT) FROM (
        SELECT id, legalities
        FROM _bronze.bronze_scryfall_cards
        WHERE COALESCE(digital::BOOLEAN, false) = false
          AND COALESCE(oversized::BOOLEAN, false) = false
          AND layout NOT IN ('token', 'double_faced_token', 'emblem')
    )
""").fetchone()
t_1 = time.perf_counter() - t0
print(f"1  json_extract ({n:,} rows):           {t_1:.2f}s")

# 20 json_extract (format_count only, no other columns)
t0 = time.perf_counter()
con.execute("""
    SELECT count(*), sum(
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
    ) FROM (
        SELECT id, legalities
        FROM _bronze.bronze_scryfall_cards
        WHERE COALESCE(digital::BOOLEAN, false) = false
          AND COALESCE(oversized::BOOLEAN, false) = false
          AND layout NOT IN ('token', 'double_faced_token', 'emblem')
    )
""").fetchone()
t_20 = time.perf_counter() - t0
print(f"20 json_extract ({n:,} rows):           {t_20:.2f}s")

# Extrapolate: how long for 24 json_extract (final CTE has 4 direct + 20 in sum)
print(f"\nEstimated cost of 24 json_extract in full pipeline: {t_20 * 24/20:.1f}s")
print(f"Ratio (full SQL 916s vs 24 json_extract estimate):"
      f" {916 / (t_20 * 24/20 + 78.3):.2f}x unexplained")

con.close()
