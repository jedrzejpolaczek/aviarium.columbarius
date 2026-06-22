"""Diagnose: why is format_count 160x slower in pipeline vs direct table access?

A) format_count directly on bronze scryfall: 5.24s
B) format_count in full CTE pipeline: ~837s

Test: materialize deduped first, THEN run format_count on temp table.
If temp table format_count is fast -> pipeline emits data in slow format.
"""
import time, duckdb
from pathlib import Path

BRONZE = "data/bronze/cards.duckdb"
SQL_RAW = Path("src/data/cards/storage/silver/sql/silver_cards.sql").read_text(encoding="utf-8")
STRIPPED = SQL_RAW.replace("CREATE OR REPLACE TABLE silver_cards AS\n", "", 1).strip()

FORMAT_COUNT_EXPR = """
    * EXCLUDE has_mtgjson_data,
    LOWER(json_extract_string(legalities,'$.commander'))     = 'legal' AS is_commander_legal,
    LOWER(json_extract_string(legalities,'$.standard'))      = 'legal' AS is_standard_legal,
    LOWER(json_extract_string(legalities,'$.modern'))        = 'legal' AS is_modern_legal,
    LOWER(json_extract_string(legalities,'$.legacy'))        = 'legal' AS is_legacy_legal,
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
"""

con = duckdb.connect(":memory:")
con.execute(f"ATTACH '{BRONZE}' AS _bronze (READ_ONLY)")

# Step 1: Materialize deduped as a CREATE TEMP TABLE
# DuckDB syntax: CREATE TABLE AS WITH ... SELECT  (not WITH ... CREATE TABLE)
# Strategy: find "final AS (" in the CTE chain, cut everything from there,
# strip trailing comments/blanks/commas, then wrap with CREATE ... AS ... SELECT
print("Step 1: Materialize 'deduped' CTE to temp_deduped table...")

idx = STRIPPED.find("\nfinal AS (")
before_final = STRIPPED[:idx]

# Strip trailing comment lines and blank lines, then remove trailing comma
lines = before_final.split("\n")
while lines and (lines[-1].strip() == "" or lines[-1].strip().startswith("--")):
    lines.pop()
if lines and lines[-1].rstrip().endswith(","):
    lines[-1] = lines[-1].rstrip()[:-1]

cte_chain = "\n".join(lines)  # WITH ... deduped AS (...)
create_deduped_sql = "CREATE TEMP TABLE temp_deduped AS\n" + cte_chain + "\nSELECT * FROM deduped"

t0 = time.perf_counter()
con.execute(create_deduped_sql)
t_mat = time.perf_counter() - t0
n = con.execute("SELECT count(*) FROM temp_deduped").fetchone()[0]
print(f"  Materialized {n:,} rows in {t_mat:.1f}s\n")

# Step 2a: format_count on temp_deduped (already in columnar DuckDB storage)
print("Step 2a: format_count on temp_deduped (columnar, in-memory DuckDB table)...")
t0 = time.perf_counter()
r2a = con.execute(f"SELECT count(*) FROM (SELECT {FORMAT_COUNT_EXPR} FROM temp_deduped)").fetchone()[0]
t_fmt_col = time.perf_counter() - t0
print(f"  {r2a:,} rows in {t_fmt_col:.1f}s\n")

# Step 2b: format_count directly on scryfall (columnar, DuckDB ATTACH)
print("Step 2b: format_count on bronze_scryfall_cards (columnar, via ATTACH)...")
t0 = time.perf_counter()
r2b = con.execute("""
    SELECT count(*) FROM (
        SELECT id, legalities,
          COALESCE((LOWER(json_extract_string(legalities,'$.standard'))='legal')::INT,0)+
          COALESCE((LOWER(json_extract_string(legalities,'$.pioneer'))='legal')::INT,0)+
          COALESCE((LOWER(json_extract_string(legalities,'$.modern'))='legal')::INT,0)+
          COALESCE((LOWER(json_extract_string(legalities,'$.legacy'))='legal')::INT,0)+
          COALESCE((LOWER(json_extract_string(legalities,'$.vintage'))='legal')::INT,0)+
          COALESCE((LOWER(json_extract_string(legalities,'$.commander'))='legal')::INT,0)+
          COALESCE((LOWER(json_extract_string(legalities,'$.oathbreaker'))='legal')::INT,0)+
          COALESCE((LOWER(json_extract_string(legalities,'$.brawl'))='legal')::INT,0)+
          COALESCE((LOWER(json_extract_string(legalities,'$.historicbrawl'))='legal')::INT,0)+
          COALESCE((LOWER(json_extract_string(legalities,'$.alchemy'))='legal')::INT,0)+
          COALESCE((LOWER(json_extract_string(legalities,'$.explorer'))='legal')::INT,0)+
          COALESCE((LOWER(json_extract_string(legalities,'$.historic'))='legal')::INT,0)+
          COALESCE((LOWER(json_extract_string(legalities,'$.timeless'))='legal')::INT,0)+
          COALESCE((LOWER(json_extract_string(legalities,'$.gladiator'))='legal')::INT,0)+
          COALESCE((LOWER(json_extract_string(legalities,'$.penny'))='legal')::INT,0)+
          COALESCE((LOWER(json_extract_string(legalities,'$.paupercommander'))='legal')::INT,0)+
          COALESCE((LOWER(json_extract_string(legalities,'$.pauper'))='legal')::INT,0)+
          COALESCE((LOWER(json_extract_string(legalities,'$.predh'))='legal')::INT,0)+
          COALESCE((LOWER(json_extract_string(legalities,'$.premodern'))='legal')::INT,0)+
          COALESCE((LOWER(json_extract_string(legalities,'$.oldschool'))='legal')::INT,0)
          AS format_count
        FROM _bronze.bronze_scryfall_cards
        WHERE COALESCE(digital::BOOLEAN,false)=false
          AND COALESCE(oversized::BOOLEAN,false)=false
          AND layout NOT IN ('token','double_faced_token','emblem')
    )
""").fetchone()[0]
t_fmt_direct = time.perf_counter() - t0
print(f"  {r2b:,} rows in {t_fmt_direct:.1f}s\n")

print("=" * 60)
print(f"  format_count on temp_deduped (materialized):  {t_fmt_col:.1f}s")
print(f"  format_count on bronze_scryfall_cards direct:  {t_fmt_direct:.1f}s")
print(f"  format_count in full pipeline (from phases2):  ~837s")
print(f"  Speedup (materialized vs pipeline):             {837/t_fmt_col:.0f}x")
con.close()
