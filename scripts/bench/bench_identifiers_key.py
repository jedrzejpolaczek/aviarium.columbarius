"""Check what keys actually exist in the identifiers JSON of bronze_mtgjson_cards."""
import duckdb

BRONZE = "data/bronze/cards.duckdb"
con = duckdb.connect(BRONZE, read_only=True)

# Check a few sample identifiers to see the actual JSON keys
samples = con.execute("""
    SELECT identifiers
    FROM bronze_mtgjson_cards
    WHERE identifiers IS NOT NULL
    LIMIT 3
""").fetchall()

print("Sample identifiers JSON:")
for row in samples:
    print(" ", row[0][:200])

# Count how many rows have scryfallId vs scryfall_id
r1 = con.execute("""
    SELECT
        count(*) as total,
        count(json_extract_string(identifiers, '$.scryfallId')) as has_scryfallId,
        count(json_extract_string(identifiers, '$.scryfall_id')) as has_scryfall_id
    FROM bronze_mtgjson_cards
""").fetchone()

print(f"\nTotal rows:              {r1[0]:>7,}")
print(f"has $.scryfallId:        {r1[1]:>7,}")
print(f"has $.scryfall_id:       {r1[2]:>7,}")

con.close()
