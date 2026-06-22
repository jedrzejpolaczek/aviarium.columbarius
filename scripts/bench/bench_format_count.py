"""Time just the format_count (20 json_extract) on 515K silver rows."""
import time, duckdb

BRONZE = "data/bronze/cards.duckdb"
SILVER = "data/silver/cards.duckdb"

con = duckdb.connect(SILVER, read_only=True)

n = con.execute("SELECT count(*) FROM silver_cards").fetchone()[0]
print(f"silver_cards rows: {n:,}")

# Time 20 json_extract on the existing silver_cards table (legalities column)
t0 = time.perf_counter()
r = con.execute("""
    SELECT count(*), sum(format_count)
    FROM (
        SELECT (
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
        FROM silver_cards
        WHERE legalities IS NOT NULL
    )
""").fetchone()
t = time.perf_counter() - t0

print(f"format_count on silver_cards: {r[0]:,} rows, sum={r[1]:,} in {t:.1f}s")

# Compare: scryfall legalities (raw from bronze) on same count
con2 = duckdb.connect(BRONZE, read_only=True)
t0 = time.perf_counter()
r2 = con2.execute("""
    SELECT count(*),
           sum(COALESCE((LOWER(json_extract_string(legalities,'$.standard'))='legal')::INT,0))
    FROM bronze_scryfall_cards
    WHERE legalities IS NOT NULL
      AND COALESCE(digital::BOOLEAN, false) = false
      AND COALESCE(oversized::BOOLEAN, false) = false
      AND layout NOT IN ('token', 'double_faced_token', 'emblem')
""").fetchone()
t2 = time.perf_counter() - t0
print(f"1x json_extract on bronze_scryfall_cards: {r2[0]:,} rows in {t2:.1f}s")
print(f"Estimated 20x: {t2*20:.1f}s")
con2.close()
con.close()
