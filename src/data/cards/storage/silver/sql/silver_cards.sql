CREATE OR REPLACE TABLE silver_cards AS
WITH

-- ── MTGJson: filter rows ─────────────────────────────────────────────────────
mtgjson_filtered AS (
    SELECT m.*
    FROM _bronze.bronze_mtgjson_cards m
    LEFT JOIN (
        SELECT id, LOWER(TRIM(set_type)) AS set_type
        FROM _bronze.bronze_scryfall_cards
    ) sc ON TRIM(json_extract_string(m.identifiers, '$.scryfall_id')) = TRIM(sc.id)
    WHERE COALESCE(m.is_online_only::BOOLEAN, false) = false
      AND COALESCE(m.is_funny::BOOLEAN,       false) = false
      AND COALESCE(m.is_oversized::BOOLEAN,   false) = false
      AND COALESCE(sc.set_type, '') NOT IN ('funny', 'memorabilia')
),

-- ── MTGJson: clean all columns ───────────────────────────────────────────────
mtgjson AS (
    SELECT
        TRIM(uuid)                                             AS uuid,
        TRIM(json_extract_string(identifiers, '$.scryfall_id')) AS scryfall_id,
        TRIM(name)                                             AS name,
        COALESCE(TRIM(ascii_name), TRIM(name))                 AS ascii_name,
        UPPER(TRIM(set_code))                                  AS set_code,
        NULLIF(TRIM(number), '_')                              AS collector_number,
        upper(left(TRIM(language), 1)) || lower(substr(TRIM(language), 2)) AS language,
        LOWER(TRIM(layout))                                    AS layout,
        NULLIF(UPPER(TRIM(mana_cost)), '_')                    AS mana_cost,
        NULLIF(TRIM(text), '_')                                AS text,
        NULLIF(TRIM(original_text), '_')                       AS original_text,
        NULLIF(TRIM(power), '_')                               AS power,
        NULLIF(TRIM(toughness), '_')                           AS toughness,
        NULLIF(TRIM(loyalty), '_')                             AS loyalty,
        NULLIF(TRIM(defense), '_')                             AS defense,
        LOWER(TRIM(rarity))                                    AS rarity,
        LOWER(TRIM(border_color))                              AS border_color,
        TRIM(frame_version)                                    AS frame_version,
        NULLIF(LOWER(TRIM(watermark)), '_')                    AS watermark,
        NULLIF(LOWER(TRIM(security_stamp)), '_')               AS security_stamp,
        NULLIF(TRIM(flavor_text), '_')                         AS flavor_text,
        NULLIF(TRIM(flavor_name), '_')                         AS flavor_name,
        NULLIF(TRIM(artist), '_')                              AS artist,
        NULLIF(TRIM(printed_name), '_')                        AS printed_name,
        NULLIF(TRIM(printed_text), '_')                        AS printed_text,
        NULLIF(TRIM(face_name), '_')                           AS face_name,
        LOWER(TRIM(side))                                      AS side,
        TRY_CAST(mana_value       AS DOUBLE)                   AS mana_value,
        TRY_CAST(face_mana_value  AS DOUBLE)                   AS face_mana_value,
        TRY_CAST(edhrec_rank      AS DOUBLE)                   AS edhrec_rank,
        TRY_CAST(edhrec_saltiness AS DOUBLE)                   AS edhrec_saltiness,
        COALESCE(is_reprint::BOOLEAN,             false)       AS is_reprint,
        COALESCE(is_reserved::BOOLEAN,            false)       AS is_reserved,
        COALESCE(is_promo::BOOLEAN,               false)       AS is_promo,
        COALESCE(is_full_art::BOOLEAN,            false)       AS is_full_art,
        COALESCE(is_textless::BOOLEAN,            false)       AS is_textless,
        COALESCE(is_alternative::BOOLEAN,         false)       AS is_alternative,
        COALESCE(is_story_spotlight::BOOLEAN,     false)       AS is_story_spotlight,
        COALESCE(is_timeshifted::BOOLEAN,         false)       AS is_timeshifted,
        COALESCE(is_rebalanced::BOOLEAN,          false)       AS is_rebalanced,
        COALESCE(is_game_changer::BOOLEAN,        false)       AS is_game_changer,
        COALESCE(has_alternative_deck_limit::BOOLEAN, false)   AS has_alternative_deck_limit,
        COALESCE(has_content_warning::BOOLEAN,    false)       AS has_content_warning,
        (original_text IS NOT NULL
         AND TRIM(COALESCE(text, '')) != TRIM(original_text))  AS errata,
        legalities,
        list_transform(COALESCE(colors,          '[]')::VARCHAR[], x -> UPPER(x))    AS colors,
        list_transform(COALESCE(color_identity,  '[]')::VARCHAR[], x -> UPPER(x))    AS color_identity,
        list_transform(COALESCE(color_indicator, '[]')::VARCHAR[], x -> UPPER(x))    AS color_indicator,
        list_transform(COALESCE(produced_mana,   '[]')::VARCHAR[], x -> UPPER(x))    AS produced_mana,
        list_transform(COALESCE(printings,       '[]')::VARCHAR[], x -> UPPER(x))    AS printings,
        list_transform(COALESCE(keywords,        '[]')::VARCHAR[], x -> upper(left(x, 1)) || lower(substr(x, 2)))  AS keywords,
        list_transform(COALESCE(finishes,        '[]')::VARCHAR[], x -> LOWER(x))    AS finishes,
        list_transform(COALESCE(availability,    '[]')::VARCHAR[], x -> LOWER(x))    AS availability,
        list_transform(COALESCE(frame_effects,   '[]')::VARCHAR[], x -> LOWER(x))    AS frame_effects,
        list_transform(COALESCE(booster_types,   '[]')::VARCHAR[], x -> LOWER(x))    AS booster_types,
        list_transform(COALESCE(promo_types,     '[]')::VARCHAR[], x -> LOWER(x))    AS promo_types,
        COALESCE(rulings,        '[]')::VARCHAR[]                                     AS rulings,
        COALESCE(artist_ids,     '[]')::VARCHAR[]                                     AS artist_ids,
        COALESCE(other_face_ids, '[]')::VARCHAR[]                                     AS other_face_ids,
        COALESCE(card_parts,     '[]')::VARCHAR[]                                     AS card_parts,
        COALESCE(variations,     '[]')::VARCHAR[]                                     AS variations,
        list_filter(
            string_split(split_part(COALESCE(TRIM(original_type),''), ' — ', 1), ' '),
            x -> list_contains(
                ['Legendary','Basic','Snow','World','Elite','Host'], x)
        )                                                       AS original_supertypes,
        list_filter(
            string_split(split_part(COALESCE(TRIM(original_type),''), ' — ', 1), ' '),
            x -> list_contains(
                ['Creature','Instant','Sorcery','Artifact','Enchantment',
                 'Planeswalker','Land','Battle','Tribal','Conspiracy','Dungeon'], x)
        )                                                       AS original_types,
        CASE
            WHEN POSITION(' — ' IN COALESCE(original_type,'')) > 0
            THEN string_split(TRIM(split_part(original_type, ' — ', 2)), ' ')
            ELSE []::VARCHAR[]
        END AS original_subtypes
    FROM mtgjson_filtered
),

-- ── Scryfall: filter rows ────────────────────────────────────────────────────
scryfall_filtered AS (
    SELECT * FROM _bronze.bronze_scryfall_cards
    WHERE COALESCE(digital::BOOLEAN,   false) = false
      AND COALESCE(oversized::BOOLEAN, false) = false
      AND COALESCE(layout, '') NOT IN ('token', 'double_faced_token', 'emblem')
      AND COALESCE(LOWER(TRIM(set_type)), '') NOT IN ('funny', 'memorabilia')
),

-- ── Scryfall: clean all columns ──────────────────────────────────────────────
scryfall AS (
    SELECT
        TRIM(id)                                               AS id,
        TRIM(oracle_id)                                        AS oracle_id,
        TRIM(name)                                             AS name,
        LOWER(TRIM(layout))                                    AS layout,
        NULLIF(UPPER(TRIM(mana_cost)), '_')                    AS mana_cost,
        TRY_CAST(cmc AS DOUBLE)                                AS cmc,
        NULLIF(TRIM(oracle_text), '_')                         AS oracle_text,
        TRIM(type_line)                                        AS type_line,
        NULLIF(TRIM(power), '_')                               AS power,
        NULLIF(TRIM(toughness), '_')                           AS toughness,
        NULLIF(TRIM(loyalty), '_')                             AS loyalty,
        NULLIF(TRIM(defense), '_')                             AS defense,
        NULLIF(TRIM(artist), '_')                              AS artist,
        TRIM(illustration_id)                                  AS illustration_id,
        LOWER(TRIM(border_color))                              AS border_color,
        TRIM(collector_number)                                 AS collector_number,
        NULLIF(TRIM(flavor_name), '_')                         AS flavor_name,
        NULLIF(TRIM(flavor_text), '_')                         AS flavor_text,
        TRIM(frame)                                            AS frame,
        NULLIF(TRIM(printed_name), '_')                        AS printed_name,
        NULLIF(TRIM(printed_text), '_')                        AS printed_text,
        NULLIF(TRIM(printed_type_line), '_')                   AS printed_type_line,
        LOWER(TRIM(rarity))                                    AS rarity,
        UPPER(TRIM(set))                                       AS set_code,
        TRIM(set_id)                                           AS set_id,
        TRIM(set_name)                                         AS set_name,
        LOWER(TRIM(set_type))                                  AS set_type,
        NULLIF(LOWER(TRIM(security_stamp)), '_')               AS security_stamp,
        NULLIF(LOWER(TRIM(watermark)), '_')                    AS watermark,
        TRIM(scryfall_uri)                                     AS scryfall_uri,
        TRIM(lang)                                             AS lang,
        CASE TRIM(lang)
            WHEN 'en'  THEN 'English'           WHEN 'es'  THEN 'Spanish'
            WHEN 'fr'  THEN 'French'            WHEN 'de'  THEN 'German'
            WHEN 'it'  THEN 'Italian'           WHEN 'pt'  THEN 'Portuguese'
            WHEN 'ja'  THEN 'Japanese'          WHEN 'ko'  THEN 'Korean'
            WHEN 'ru'  THEN 'Russian'           WHEN 'zhs' THEN 'Chinese Simplified'
            WHEN 'zht' THEN 'Chinese Traditional' WHEN 'he' THEN 'Hebrew'
            WHEN 'la'  THEN 'Latin'             WHEN 'grc' THEN 'Ancient Greek'
            WHEN 'ar'  THEN 'Arabic'            WHEN 'sa'  THEN 'Sanskrit'
            WHEN 'ph'  THEN 'Phyrexian'         WHEN 'qya' THEN 'Quenya'
            ELSE TRIM(lang)
        END                                                    AS language,
        TRY_CAST(tcgplayer_id  AS DOUBLE)                     AS tcgplayer_id,
        TRY_CAST(cardmarket_id AS DOUBLE)                      AS cardmarket_id,
        TRY_CAST(edhrec_rank   AS DOUBLE)                      AS edhrec_rank,
        TRY_CAST(penny_rank    AS DOUBLE)                      AS penny_rank,
        COALESCE(reserved::BOOLEAN,        false)              AS is_reserved,
        COALESCE(reprint::BOOLEAN,         false)              AS is_reprint,
        COALESCE(promo::BOOLEAN,           false)              AS is_promo,
        COALESCE(full_art::BOOLEAN,        false)              AS is_full_art,
        COALESCE(textless::BOOLEAN,        false)              AS is_textless,
        COALESCE(variation::BOOLEAN,       false)              AS is_variation,
        COALESCE(booster::BOOLEAN,         false)              AS is_booster,
        COALESCE(story_spotlight::BOOLEAN, false)              AS is_story_spotlight,
        COALESCE(game_changer::BOOLEAN,    false)              AS game_changer,
        legalities,
        list_transform(COALESCE(color_identity,  '[]')::VARCHAR[], x -> UPPER(x))   AS color_identity,
        list_transform(COALESCE(color_indicator, '[]')::VARCHAR[], x -> UPPER(x))   AS color_indicator,
        list_transform(COALESCE(colors,          '[]')::VARCHAR[], x -> UPPER(x))   AS colors,
        list_transform(COALESCE(produced_mana,   '[]')::VARCHAR[], x -> UPPER(x))   AS produced_mana,
        list_transform(COALESCE(keywords,        '[]')::VARCHAR[], x -> upper(left(x, 1)) || lower(substr(x, 2))) AS keywords,
        list_transform(COALESCE(finishes,        '[]')::VARCHAR[], x -> LOWER(x))   AS finishes,
        list_transform(COALESCE(frame_effects,   '[]')::VARCHAR[], x -> LOWER(x))   AS frame_effects,
        list_transform(COALESCE(games,           '[]')::VARCHAR[], x -> LOWER(x))   AS games,
        list_transform(COALESCE(promo_types,     '[]')::VARCHAR[], x -> LOWER(x))   AS promo_types,
        COALESCE(artist_ids,     '[]')::VARCHAR[]                                    AS artist_ids,
        COALESCE(multiverse_ids, '[]')::VARCHAR[]                                    AS multiverse_ids,
        COALESCE(all_parts,      '[]')::VARCHAR[]                                    AS all_parts,
        COALESCE(card_faces,     '[]')::VARCHAR[]                                    AS card_faces
    FROM scryfall_filtered
),

-- ── FULL OUTER JOIN: MTGJson values preferred; Scryfall fills for Scryfall-only rows ──
joined AS (
    SELECT
        m.uuid,
        COALESCE(m.scryfall_id, s.id)                         AS scryfall_id,
        s.oracle_id,
        s.type_line,
        s.illustration_id,
        s.set_id,
        s.set_name,
        s.set_type,
        s.scryfall_uri,
        s.lang,
        s.tcgplayer_id,
        s.cardmarket_id,
        s.penny_rank,
        s.is_variation,
        s.is_booster,
        s.game_changer,
        s.all_parts,
        s.card_faces,
        s.multiverse_ids,
        s.games,
        COALESCE(m.name,               s.name)                AS name,
        COALESCE(m.ascii_name,         s.name)                AS ascii_name,
        COALESCE(m.set_code,           s.set_code)            AS set_code,
        COALESCE(m.collector_number,   s.collector_number)    AS collector_number,
        COALESCE(m.language,           s.language)            AS language,
        COALESCE(m.layout,             s.layout)              AS layout,
        COALESCE(m.mana_cost,          s.mana_cost)           AS mana_cost,
        COALESCE(m.mana_value,         s.cmc)                 AS mana_value,
        COALESCE(m.text,               s.oracle_text)         AS text,
        m.original_text,
        m.errata,
        COALESCE(m.power,              s.power)               AS power,
        COALESCE(m.toughness,          s.toughness)           AS toughness,
        COALESCE(m.loyalty,            s.loyalty)             AS loyalty,
        COALESCE(m.defense,            s.defense)             AS defense,
        COALESCE(m.legalities,         s.legalities)          AS legalities,
        COALESCE(m.rarity,             s.rarity)              AS rarity,
        COALESCE(m.border_color,       s.border_color)        AS border_color,
        COALESCE(m.frame_version,      s.frame)               AS frame_version,
        COALESCE(m.frame_effects,      s.frame_effects)       AS frame_effects,
        COALESCE(m.finishes,           s.finishes)            AS finishes,
        COALESCE(m.watermark,          s.watermark)           AS watermark,
        COALESCE(m.security_stamp,     s.security_stamp)      AS security_stamp,
        COALESCE(m.flavor_text,        s.flavor_text)         AS flavor_text,
        COALESCE(m.flavor_name,        s.flavor_name)         AS flavor_name,
        COALESCE(m.artist,             s.artist)              AS artist,
        COALESCE(m.is_reprint,         s.is_reprint)          AS is_reprint,
        COALESCE(m.is_reserved,        s.is_reserved)         AS is_reserved,
        COALESCE(m.is_promo,           s.is_promo)            AS is_promo,
        COALESCE(m.is_full_art,        s.is_full_art)         AS is_full_art,
        COALESCE(m.is_textless,        s.is_textless)         AS is_textless,
        COALESCE(m.is_story_spotlight, s.is_story_spotlight)  AS is_story_spotlight,
        COALESCE(m.colors,             s.colors)              AS colors,
        COALESCE(m.color_identity,     s.color_identity)      AS color_identity,
        COALESCE(m.color_indicator,    s.color_indicator)     AS color_indicator,
        COALESCE(m.produced_mana,      s.produced_mana)       AS produced_mana,
        COALESCE(m.keywords,           s.keywords)            AS keywords,
        COALESCE(m.promo_types,        s.promo_types)         AS promo_types,
        COALESCE(m.artist_ids,         s.artist_ids)          AS artist_ids,
        m.face_mana_value,
        m.edhrec_rank,
        m.edhrec_saltiness,
        m.is_alternative,
        m.is_timeshifted,
        m.is_rebalanced,
        m.is_game_changer,
        m.has_alternative_deck_limit,
        m.has_content_warning,
        m.printings,
        m.rulings,
        m.other_face_ids,
        m.card_parts,
        m.variations,
        m.original_supertypes,
        m.original_types,
        m.original_subtypes,
        (m.uuid IS NOT NULL)                                   AS has_mtgjson_data
    FROM mtgjson m
    FULL OUTER JOIN scryfall s ON m.scryfall_id = s.id
),

-- ── Resolve canonical_uuid for Scryfall-only language-variant rows ───────────
canonical_map AS (
    SELECT set_code, collector_number, MIN(uuid) AS uuid
    FROM mtgjson
    WHERE uuid IS NOT NULL
      AND set_code IS NOT NULL
      AND collector_number IS NOT NULL
    GROUP BY set_code, collector_number
),
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
),

-- ── Dedup DFC multi-face rows: keep front face (smallest uuid), then Scryfall-only ──
deduped AS (
    SELECT * EXCLUDE rn
    FROM (
        SELECT *,
               ROW_NUMBER() OVER (
                   PARTITION BY scryfall_id
                   ORDER BY
                       CASE WHEN uuid IS NOT NULL THEN 0 ELSE 1 END,
                       uuid
               ) AS rn
        FROM with_canonical
        WHERE scryfall_id IS NOT NULL
    )
    WHERE rn = 1
),

-- ── Extract scalar legality columns ──────────────────────────────────────────
final AS (
    SELECT
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
        )                                                        AS format_count
    FROM deduped
)

SELECT * FROM final
