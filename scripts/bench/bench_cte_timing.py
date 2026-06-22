"""Isolate which CTE in silver_cards.sql is the slowest.
Each test adds one more CTE and measures cumulative time.
"""
import time, duckdb

BRONZE = "data/bronze/cards.duckdb"

def timed(label, sql, con):
    t0 = time.perf_counter()
    r = con.execute(sql).fetchone()[0]
    return time.perf_counter() - t0, r

def make_con():
    con = duckdb.connect(":memory:")
    con.execute(f"ATTACH '{BRONZE}' AS _bronze (READ_ONLY)")
    return con

print("Cumulative CTE timing (each step adds the next CTE):\n")
con = make_con()

mtgjson_filtered = """
mtgjson_filtered AS (
    SELECT * FROM _bronze.bronze_mtgjson_cards
    WHERE COALESCE(is_online_only::BOOLEAN, false) = false
      AND COALESCE(is_funny::BOOLEAN,       false) = false
      AND COALESCE(is_oversized::BOOLEAN,   false) = false
)"""

mtgjson = """
mtgjson AS (
    SELECT
        TRIM(uuid) AS uuid,
        json_extract_string(identifiers, '$.scryfallId') AS scryfall_id,
        TRIM(name) AS name,
        COALESCE(TRIM(ascii_name), TRIM(name)) AS ascii_name,
        UPPER(TRIM(set_code)) AS set_code,
        NULLIF(TRIM(number), '_') AS collector_number,
        upper(left(TRIM(language), 1)) || lower(substr(TRIM(language), 2)) AS language,
        LOWER(TRIM(layout)) AS layout,
        NULLIF(UPPER(TRIM(mana_cost)), '_') AS mana_cost,
        NULLIF(TRIM(text), '_') AS text,
        NULLIF(TRIM(original_text), '_') AS original_text,
        NULLIF(TRIM(power), '_') AS power,
        NULLIF(TRIM(toughness), '_') AS toughness,
        NULLIF(TRIM(loyalty), '_') AS loyalty,
        NULLIF(TRIM(defense), '_') AS defense,
        LOWER(TRIM(rarity)) AS rarity,
        LOWER(TRIM(border_color)) AS border_color,
        TRIM(frame_version) AS frame_version,
        NULLIF(LOWER(TRIM(watermark)), '_') AS watermark,
        NULLIF(LOWER(TRIM(security_stamp)), '_') AS security_stamp,
        NULLIF(TRIM(flavor_text), '_') AS flavor_text,
        NULLIF(TRIM(flavor_name), '_') AS flavor_name,
        NULLIF(TRIM(artist), '_') AS artist,
        NULLIF(TRIM(printed_name), '_') AS printed_name,
        NULLIF(TRIM(printed_text), '_') AS printed_text,
        NULLIF(TRIM(face_name), '_') AS face_name,
        LOWER(TRIM(side)) AS side,
        TRY_CAST(mana_value AS DOUBLE) AS mana_value,
        TRY_CAST(face_mana_value AS DOUBLE) AS face_mana_value,
        TRY_CAST(edhrec_rank AS DOUBLE) AS edhrec_rank,
        TRY_CAST(edhrec_saltiness AS DOUBLE) AS edhrec_saltiness,
        COALESCE(is_reprint::BOOLEAN, false) AS is_reprint,
        COALESCE(is_reserved::BOOLEAN, false) AS is_reserved,
        COALESCE(is_promo::BOOLEAN, false) AS is_promo,
        COALESCE(is_full_art::BOOLEAN, false) AS is_full_art,
        COALESCE(is_textless::BOOLEAN, false) AS is_textless,
        COALESCE(is_alternative::BOOLEAN, false) AS is_alternative,
        COALESCE(is_story_spotlight::BOOLEAN, false) AS is_story_spotlight,
        COALESCE(is_timeshifted::BOOLEAN, false) AS is_timeshifted,
        COALESCE(is_rebalanced::BOOLEAN, false) AS is_rebalanced,
        COALESCE(is_game_changer::BOOLEAN, false) AS is_game_changer,
        COALESCE(has_alternative_deck_limit::BOOLEAN, false) AS has_alternative_deck_limit,
        COALESCE(has_content_warning::BOOLEAN, false) AS has_content_warning,
        (original_text IS NOT NULL AND TRIM(COALESCE(text,'')) != TRIM(original_text)) AS errata,
        legalities,
        list_transform(COALESCE(colors,'[]')::VARCHAR[], x -> UPPER(x)) AS colors,
        list_transform(COALESCE(color_identity,'[]')::VARCHAR[], x -> UPPER(x)) AS color_identity,
        list_transform(COALESCE(color_indicator,'[]')::VARCHAR[], x -> UPPER(x)) AS color_indicator,
        list_transform(COALESCE(produced_mana,'[]')::VARCHAR[], x -> UPPER(x)) AS produced_mana,
        list_transform(COALESCE(printings,'[]')::VARCHAR[], x -> UPPER(x)) AS printings,
        list_transform(COALESCE(keywords,'[]')::VARCHAR[], x -> upper(left(x,1)) || lower(substr(x,2))) AS keywords,
        list_transform(COALESCE(finishes,'[]')::VARCHAR[], x -> LOWER(x)) AS finishes,
        list_transform(COALESCE(availability,'[]')::VARCHAR[], x -> LOWER(x)) AS availability,
        list_transform(COALESCE(frame_effects,'[]')::VARCHAR[], x -> LOWER(x)) AS frame_effects,
        list_transform(COALESCE(booster_types,'[]')::VARCHAR[], x -> LOWER(x)) AS booster_types,
        list_transform(COALESCE(promo_types,'[]')::VARCHAR[], x -> LOWER(x)) AS promo_types,
        COALESCE(rulings,'[]')::VARCHAR[] AS rulings,
        COALESCE(artist_ids,'[]')::VARCHAR[] AS artist_ids,
        COALESCE(other_face_ids,'[]')::VARCHAR[] AS other_face_ids,
        COALESCE(card_parts,'[]')::VARCHAR[] AS card_parts,
        COALESCE(variations,'[]')::VARCHAR[] AS variations,
        list_filter(string_split(split_part(COALESCE(TRIM(original_type),''), ' ? ', 1), ' '),
            x -> list_contains(['Legendary','Basic','Snow','World','Elite','Host'], x)) AS original_supertypes,
        list_filter(string_split(split_part(COALESCE(TRIM(original_type),''), ' ? ', 1), ' '),
            x -> list_contains(['Creature','Instant','Sorcery','Artifact','Enchantment',
                 'Planeswalker','Land','Battle','Tribal','Conspiracy','Dungeon'], x)) AS original_types,
        CASE WHEN POSITION(' ? ' IN COALESCE(original_type,'')) > 0
             THEN string_split(TRIM(split_part(original_type, ' ? ', 2)), ' ')
             ELSE []::VARCHAR[] END AS original_subtypes
    FROM mtgjson_filtered
)"""

scryfall_filtered = """
scryfall_filtered AS (
    SELECT * FROM _bronze.bronze_scryfall_cards
    WHERE COALESCE(digital::BOOLEAN, false) = false
      AND COALESCE(oversized::BOOLEAN, false) = false
      AND layout NOT IN ('token', 'double_faced_token', 'emblem')
)"""

scryfall = """
scryfall AS (
    SELECT
        TRIM(id) AS id, TRIM(oracle_id) AS oracle_id,
        TRIM(name) AS name, LOWER(TRIM(layout)) AS layout,
        NULLIF(UPPER(TRIM(mana_cost)),'_') AS mana_cost,
        TRY_CAST(cmc AS DOUBLE) AS cmc, NULLIF(TRIM(oracle_text),'_') AS oracle_text,
        TRIM(type_line) AS type_line, NULLIF(TRIM(power),'_') AS power,
        NULLIF(TRIM(toughness),'_') AS toughness, NULLIF(TRIM(loyalty),'_') AS loyalty,
        NULLIF(TRIM(defense),'_') AS defense, NULLIF(TRIM(artist),'_') AS artist,
        TRIM(illustration_id) AS illustration_id, LOWER(TRIM(border_color)) AS border_color,
        TRIM(collector_number) AS collector_number, NULLIF(TRIM(flavor_name),'_') AS flavor_name,
        NULLIF(TRIM(flavor_text),'_') AS flavor_text, TRIM(frame) AS frame,
        NULLIF(TRIM(printed_name),'_') AS printed_name, NULLIF(TRIM(printed_text),'_') AS printed_text,
        NULLIF(TRIM(printed_type_line),'_') AS printed_type_line, LOWER(TRIM(rarity)) AS rarity,
        UPPER(TRIM(set)) AS set_code, TRIM(set_id) AS set_id, TRIM(set_name) AS set_name,
        LOWER(TRIM(set_type)) AS set_type, NULLIF(LOWER(TRIM(security_stamp)),'_') AS security_stamp,
        NULLIF(LOWER(TRIM(watermark)),'_') AS watermark, TRIM(scryfall_uri) AS scryfall_uri,
        TRIM(lang) AS lang,
        CASE TRIM(lang) WHEN 'en' THEN 'English' WHEN 'es' THEN 'Spanish' WHEN 'fr' THEN 'French'
            WHEN 'de' THEN 'German' WHEN 'it' THEN 'Italian' WHEN 'pt' THEN 'Portuguese'
            WHEN 'ja' THEN 'Japanese' WHEN 'ko' THEN 'Korean' WHEN 'ru' THEN 'Russian'
            WHEN 'zhs' THEN 'Chinese Simplified' WHEN 'zht' THEN 'Chinese Traditional'
            ELSE TRIM(lang) END AS language,
        TRY_CAST(tcgplayer_id AS DOUBLE) AS tcgplayer_id,
        TRY_CAST(cardmarket_id AS DOUBLE) AS cardmarket_id,
        TRY_CAST(edhrec_rank AS DOUBLE) AS edhrec_rank,
        TRY_CAST(penny_rank AS DOUBLE) AS penny_rank,
        COALESCE(reserved::BOOLEAN,false) AS is_reserved, COALESCE(reprint::BOOLEAN,false) AS is_reprint,
        COALESCE(promo::BOOLEAN,false) AS is_promo, COALESCE(full_art::BOOLEAN,false) AS is_full_art,
        COALESCE(textless::BOOLEAN,false) AS is_textless, COALESCE(variation::BOOLEAN,false) AS is_variation,
        COALESCE(booster::BOOLEAN,false) AS is_booster,
        COALESCE(story_spotlight::BOOLEAN,false) AS is_story_spotlight,
        COALESCE(game_changer::BOOLEAN,false) AS game_changer,
        legalities,
        list_transform(COALESCE(color_identity,'[]')::VARCHAR[], x -> UPPER(x)) AS color_identity,
        list_transform(COALESCE(color_indicator,'[]')::VARCHAR[], x -> UPPER(x)) AS color_indicator,
        list_transform(COALESCE(colors,'[]')::VARCHAR[], x -> UPPER(x)) AS colors,
        list_transform(COALESCE(produced_mana,'[]')::VARCHAR[], x -> UPPER(x)) AS produced_mana,
        list_transform(COALESCE(keywords,'[]')::VARCHAR[], x -> upper(left(x,1)) || lower(substr(x,2))) AS keywords,
        list_transform(COALESCE(finishes,'[]')::VARCHAR[], x -> LOWER(x)) AS finishes,
        list_transform(COALESCE(frame_effects,'[]')::VARCHAR[], x -> LOWER(x)) AS frame_effects,
        list_transform(COALESCE(games,'[]')::VARCHAR[], x -> LOWER(x)) AS games,
        list_transform(COALESCE(promo_types,'[]')::VARCHAR[], x -> LOWER(x)) AS promo_types,
        COALESCE(artist_ids,'[]')::VARCHAR[] AS artist_ids,
        COALESCE(multiverse_ids,'[]')::VARCHAR[] AS multiverse_ids,
        COALESCE(all_parts,'[]')::VARCHAR[] AS all_parts,
        COALESCE(card_faces,'[]')::VARCHAR[] AS card_faces
    FROM scryfall_filtered
)"""

steps = [
    ("1. mtgjson_filtered", f"WITH {mtgjson_filtered} SELECT count(*) FROM mtgjson_filtered"),
    ("2. mtgjson (full cols)", f"WITH {mtgjson_filtered},{mtgjson} SELECT count(*) FROM mtgjson"),
    ("3. scryfall_filtered", f"WITH {scryfall_filtered} SELECT count(*) FROM scryfall_filtered"),
    ("4. scryfall (full cols)", f"WITH {scryfall_filtered},{scryfall} SELECT count(*) FROM scryfall"),
]

for label, sql in steps:
    t, n = timed(label, sql, con)
    print(f"  {label:35s}  {n:>7,} rows  {t:6.2f}s")

con.close()
