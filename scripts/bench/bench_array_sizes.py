"""Quick check: how large are the big array columns in bronze tables."""
import duckdb

con = duckdb.connect("data/bronze/cards.duckdb", read_only=True)

print("=== MTGJson (filtered) ===")
r = con.execute("""
SELECT
    count(*) AS rows,
    sum(length(COALESCE(rulings,        '[]')::VARCHAR)) / 1024 / 1024 AS rulings_mb,
    sum(length(COALESCE(printings,      '[]')::VARCHAR)) / 1024 / 1024 AS printings_mb,
    sum(length(COALESCE(other_face_ids, '[]')::VARCHAR)) / 1024 / 1024 AS other_face_ids_mb,
    sum(length(COALESCE(variations,     '[]')::VARCHAR)) / 1024 / 1024 AS variations_mb,
    sum(length(COALESCE(card_parts,     '[]')::VARCHAR)) / 1024 / 1024 AS card_parts_mb
FROM bronze_mtgjson_cards
WHERE COALESCE(is_online_only::BOOLEAN, false) = false
  AND COALESCE(is_funny::BOOLEAN,       false) = false
  AND COALESCE(is_oversized::BOOLEAN,   false) = false
""").fetchone()
print(f"  rows={r[0]}, rulings={r[1]:.0f}MB, printings={r[2]:.0f}MB, "
      f"other_face_ids={r[3]:.0f}MB, variations={r[4]:.0f}MB, card_parts={r[5]:.0f}MB")

print("\n=== Scryfall (filtered) ===")
r2 = con.execute("""
SELECT
    count(*) AS rows,
    sum(length(COALESCE(card_faces,     '[]')::VARCHAR)) / 1024 / 1024 AS card_faces_mb,
    sum(length(COALESCE(all_parts,      '[]')::VARCHAR)) / 1024 / 1024 AS all_parts_mb,
    sum(length(COALESCE(multiverse_ids, '[]')::VARCHAR)) / 1024 / 1024 AS multiverse_ids_mb,
    sum(length(COALESCE(legalities,     '{}')::VARCHAR)) / 1024 / 1024 AS legalities_mb
FROM bronze_scryfall_cards
WHERE COALESCE(digital::BOOLEAN,   false) = false
  AND COALESCE(oversized::BOOLEAN, false) = false
  AND layout NOT IN ('token', 'double_faced_token', 'emblem')
""").fetchone()
print(f"  rows={r2[0]}, card_faces={r2[1]:.0f}MB, all_parts={r2[2]:.0f}MB, "
      f"multiverse_ids={r2[3]:.0f}MB, legalities={r2[4]:.0f}MB")

print("\n=== DuckDB memory_limit ===")
ml = con.execute("SELECT current_setting('memory_limit')").fetchone()
print(f"  memory_limit = {ml[0]}")

threads = con.execute("SELECT current_setting('threads')").fetchone()
print(f"  threads = {threads[0]}")

con.close()
