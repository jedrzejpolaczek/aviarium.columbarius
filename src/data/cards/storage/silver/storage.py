"""DuckDB persistence layer for the Silver (cleaned) tier.

Exposes SilverStorage, a context-manager class that reads raw data from Bronze
DuckDB tables, applies config-driven transformations, and writes clean data to
Silver DuckDB tables.

Transformation rules for each source table are declared in silver_config.json —
adding a new source requires only a new config entry and no code changes.

Typical usage:
    with SilverStorage(
        "data/bronze/cards.duckdb",
        "data/silver/cards.duckdb",
        "configs/silver_config.json",
    ) as storage:
        storage.populate()   # initial load
        # or
        storage.update()     # incremental daily run
"""

import datetime
import json
from pathlib import Path

import duckdb
import pandas as pd

from src.data.cards.storage.base import TransformStorage, get_tables
from src.data.cards.storage.errors import StorageConnectionError, StorageWriteError
from src.data.cards.storage.silver.card_join import SilverCardJoin
from src.data.cards.storage.silver.persistence import SilverWriter
from src.data.cards.storage.silver.prices import SilverPriceBuilder
from src.data.cards.storage.silver.report import write_report
from src.data.cards.storage.silver.transforms import SilverTransforms
from src.logger import get_logger

logger = get_logger(__name__)

_SILVER_CARDS_SQL = """
CREATE OR REPLACE TABLE silver_cards AS
WITH

-- ── MTGJson: filter rows ─────────────────────────────────────────────────────
mtgjson_filtered AS (
    SELECT * FROM _bronze.bronze_mtgjson_cards
    WHERE COALESCE(is_online_only::BOOLEAN, false) = false
      AND COALESCE(is_funny::BOOLEAN,       false) = false
      AND COALESCE(is_oversized::BOOLEAN,   false) = false
),

-- ── MTGJson: clean all columns ───────────────────────────────────────────────
mtgjson AS (
    SELECT
        TRIM(uuid)                                              AS uuid,
        json_extract_string(identifiers, '$.scryfallId')       AS scryfall_id,
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
        END                                                     AS original_subtypes
    FROM mtgjson_filtered
),

-- ── Scryfall: filter rows ────────────────────────────────────────────────────
scryfall_filtered AS (
    SELECT * FROM _bronze.bronze_scryfall_cards
    WHERE COALESCE(digital::BOOLEAN,   false) = false
      AND COALESCE(oversized::BOOLEAN, false) = false
      AND COALESCE(layout, '') NOT IN ('token', 'double_faced_token', 'emblem')
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
           ON j.uuid IS NULL
          AND j.set_code        = cm.set_code
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
          (LOWER(json_extract_string(legalities,'$.standard'))    ='legal')::INT +
          (LOWER(json_extract_string(legalities,'$.pioneer'))     ='legal')::INT +
          (LOWER(json_extract_string(legalities,'$.modern'))      ='legal')::INT +
          (LOWER(json_extract_string(legalities,'$.legacy'))      ='legal')::INT +
          (LOWER(json_extract_string(legalities,'$.vintage'))     ='legal')::INT +
          (LOWER(json_extract_string(legalities,'$.commander'))   ='legal')::INT +
          (LOWER(json_extract_string(legalities,'$.oathbreaker')) ='legal')::INT +
          (LOWER(json_extract_string(legalities,'$.brawl'))       ='legal')::INT +
          (LOWER(json_extract_string(legalities,'$.historicbrawl'))='legal')::INT +
          (LOWER(json_extract_string(legalities,'$.alchemy'))     ='legal')::INT +
          (LOWER(json_extract_string(legalities,'$.explorer'))    ='legal')::INT +
          (LOWER(json_extract_string(legalities,'$.historic'))    ='legal')::INT +
          (LOWER(json_extract_string(legalities,'$.timeless'))    ='legal')::INT +
          (LOWER(json_extract_string(legalities,'$.gladiator'))   ='legal')::INT +
          (LOWER(json_extract_string(legalities,'$.penny'))       ='legal')::INT +
          (LOWER(json_extract_string(legalities,'$.paupercommander'))='legal')::INT +
          (LOWER(json_extract_string(legalities,'$.pauper'))      ='legal')::INT +
          (LOWER(json_extract_string(legalities,'$.predh'))       ='legal')::INT +
          (LOWER(json_extract_string(legalities,'$.premodern'))   ='legal')::INT +
          (LOWER(json_extract_string(legalities,'$.oldschool'))   ='legal')::INT
        )                                                        AS format_count
    FROM deduped
)

SELECT * FROM final
"""


class SilverStorage(TransformStorage):
    """Persistence layer for the Silver (cleaned) tier.

    Inherits connection management and the populate()/update() entry points
    from TransformStorage. Reads raw DataFrames from Bronze DuckDB, applies a
    config-driven transformation pipeline (row filtering, type coercion,
    normalization, column renames), and writes the results to Silver DuckDB.

    Transformation rules for each source are declared in silver_config.json.
    Adding a new source requires only a new entry there — no code changes needed.

    Composition:
        _card_join  (SilverCardJoin)    — MTGJson × Scryfall merge logic
        _prices     (SilverPriceBuilder)— price extraction, join, and forward-fill
        _writer     (DuckDBWriter)      — DuckDB append / full-load / upsert helpers

    Usage:
        with SilverStorage(
            "data/bronze/cards.duckdb",
            "data/silver/cards.duckdb",
            "configs/silver_config.json",
        ) as storage:
            storage.populate()     # initial load or full rebuild
            storage.update()       # incremental daily run

    Raises:
        StorageConnectionError: If either DuckDB connection cannot be opened.
    """

    def __init__(
        self, bronze_db_path: str, silver_db_path: str, config_path: str
    ) -> None:
        """Open Bronze (read-only) and Silver (read-write) DuckDB connections.

        Args:
            bronze_db_path: Path to the Bronze DuckDB file.
            silver_db_path: Path to the Silver DuckDB file (created if it does not exist).
            config_path: Path to the silver_config.json file.

        Raises:
            StorageConnectionError: If either connection cannot be established.
        """
        self._bronze_db_path = bronze_db_path
        self._bronze_con = self._open_connection(bronze_db_path, read_only=True)
        self._silver_con = self._open_connection(silver_db_path, read_only=False)

        try:
            self._config = json.loads(Path(config_path).read_text())
        except FileNotFoundError:
            raise StorageConnectionError(
                f"Silver config not found: {config_path}"
            ) from None
        except json.JSONDecodeError as e:
            raise StorageConnectionError(
                f"Invalid JSON in silver config {config_path}: {e}"
            ) from e

        self._transforms = SilverTransforms(
            language_map=self._config["language_map"],
            legality_map=self._config["legality_map"],
            supertypes=self._config["supertypes"],
            card_types=self._config["card_types"],
        )
        self._card_join = SilverCardJoin(language_map=self._config["language_map"])
        self._writer = SilverWriter(self._silver_con)
        self._prices = SilverPriceBuilder(self._bronze_con, self._silver_con)

    def close(self) -> None:
        """Close both Bronze and Silver DuckDB connections."""
        self._bronze_con.close()
        self._silver_con.close()
        logger.progress("Closed SilverStorage connections")

    # ------------------------------------------------------------------
    # Bronze loading
    # ------------------------------------------------------------------

    def _load_bronze_data(self) -> dict[str, pd.DataFrame]:
        """Load Bronze tables referenced in the config into DataFrames.

        Only tables declared as sources in silver_config.json are loaded,
        avoiding the cost of reading history tables with millions of rows.
        """
        needed = {f"bronze_{name}" for name in self._config["sources"]}
        existing = get_tables(self._bronze_con)
        to_load = needed & existing
        logger.progress("Loading %d Bronze tables", len(to_load))
        return {
            name: self._bronze_con.execute(f"SELECT * FROM {name}").df()
            for name in to_load
        }

    # ------------------------------------------------------------------
    # History appenders
    # ------------------------------------------------------------------

    def _append_tournament_results_history(self) -> None:
        """Append tournament results from Bronze to silver_tournament_results_history.

        Reads bronze_tournament_results, normalises card_name, then joins to
        silver_cards on name to resolve oracle_id and one representative
        scryfall_id. Unmatched cards are kept with oracle_id = NULL so no
        Bronze data is silently lost.
        """
        bronze_tables = get_tables(self._bronze_con)
        if "bronze_tournament_results" not in bronze_tables:
            logger.warning(
                "bronze_tournament_results not found — skipping tournament results history"
            )
            return

        df = self._bronze_con.execute("SELECT * FROM bronze_tournament_results").df()
        if df.empty:
            logger.warning("bronze_tournament_results is empty — skipping")
            return

        df["card_name"] = (
            df["card_name"].str.strip().str.replace(" / ", " // ", regex=False)
        )

        silver_tables = get_tables(self._silver_con)
        if "silver_cards" in silver_tables:
            card_map = (
                self._silver_con.execute(
                    "SELECT DISTINCT name, oracle_id, scryfall_id "
                    "FROM silver_cards WHERE name IS NOT NULL"
                )
                .df()
                .drop_duplicates(subset=["name"], keep="first")
            )
            df = df.merge(card_map, left_on="card_name", right_on="name", how="left")
            df = df.drop(columns=["name"])
        else:
            logger.warning(
                "silver_cards not yet available — oracle_id/scryfall_id will be NULL"
            )
            df["oracle_id"] = None
            df["scryfall_id"] = None

        df["snapshot_date"] = df["tournament_date"]
        self._writer.append(df, "silver_tournament_results_history", key_column="id")

    def _append_format_staples_history(self) -> None:
        """Append today's format-staples snapshot from Bronze to silver_format_staples_history.

        Safe to call multiple times per day — duplicate (id, snapshot_date) pairs
        are skipped.
        """
        bronze_tables = get_tables(self._bronze_con)
        if "bronze_format_staples_history" not in bronze_tables:
            logger.warning(
                "bronze_format_staples_history not found — skipping format staples history"
            )
            return
        df = self._bronze_con.execute(
            "SELECT * FROM bronze_format_staples_history"
        ).df()
        self._writer.append(df, "silver_format_staples_history", key_column="id")

    def _build_silver_cards_sql(self) -> None:
        """Build silver_cards entirely in DuckDB SQL via ATTACH of the Bronze file.

        Executes _SILVER_CARDS_SQL: a multi-CTE CREATE OR REPLACE TABLE that filters,
        cleans, joins MTGJson × Scryfall, resolves canonical_uuid, deduplicates
        multi-face DFC rows, and extracts scalar legality columns — all in one
        DuckDB query. No pandas DataFrame is allocated.

        silver_cards is always fully rebuilt (CREATE OR REPLACE) because Scryfall
        delivers a complete daily snapshot and orphan rows from previous runs must
        not persist (identical semantics to the previous full_load call).
        """
        bronze_tables = get_tables(self._bronze_con)
        missing = [
            t for t in ("bronze_mtgjson_cards", "bronze_scryfall_cards")
            if t not in bronze_tables
        ]
        if missing:
            logger.warning(
                "Missing Bronze tables %s — skipping silver_cards build", missing
            )
            return
        try:
            self._silver_con.execute(
                f"ATTACH '{self._bronze_db_path}' AS _bronze (READ_ONLY)"
            )
            self._silver_con.execute(_SILVER_CARDS_SQL)
            count = self._silver_con.execute(
                "SELECT count(*) FROM silver_cards"
            ).fetchone()
            logger.info(
                "Built silver_cards via SQL path: %d rows",
                count[0] if count else 0,
            )
        except duckdb.Error as e:
            logger.error("Failed to build silver_cards via SQL: %s", e)
            raise StorageWriteError(f"Failed to build silver_cards: {e}") from e
        finally:
            try:
                self._silver_con.execute("DETACH _bronze")
            except duckdb.Error:
                pass

    def _append_meta_history_sql(self) -> None:
        """Append Bronze scryfall_meta_history to Silver via DuckDB SQL.

        ATTACHes the Bronze file read-only, transforms with TRIM/TRY_CAST/COALESCE/
        lower, filters via INNER JOIN silver_cards (when available), and INSERTs via
        anti-join dedup — same contract as DuckDBWriter.append().

        legalities is passed through unchanged: Scryfall already stores values
        lowercase and Gold reads them via json_extract_string.
        """
        bronze_tables = get_tables(self._bronze_con)
        if "bronze_scryfall_meta_history" not in bronze_tables:
            logger.warning(
                "bronze_scryfall_meta_history not found — skipping silver_meta_history"
            )
            return

        silver_tables = get_tables(self._silver_con)
        has_silver_cards = "silver_cards" in silver_tables
        join_clause = (
            "INNER JOIN silver_cards sc ON sc.scryfall_id = b.id"
            if has_silver_cards
            else ""
        )
        if not has_silver_cards:
            logger.warning(
                "silver_cards not available — writing all meta_history rows unfiltered"
            )

        transform_sql = f"""
            SELECT
                TRIM(b.id)            AS id,
                TRIM(b.snapshot_date) AS snapshot_date,
                b.legalities,
                TRY_CAST(b.edhrec_rank AS INTEGER)   AS edhrec_rank,
                COALESCE(b.reserved::BOOLEAN, false)  AS is_reserved,
                COALESCE(lower(b.promo_types), '[]')  AS promo_types,
                COALESCE(lower(b.finishes),    '[]')  AS finishes
            FROM _bronze.bronze_scryfall_meta_history b
            {join_clause}
        """
        try:
            self._silver_con.execute(
                f"ATTACH '{self._bronze_db_path}' AS _bronze (READ_ONLY)"
            )
            if "silver_meta_history" not in silver_tables:
                self._silver_con.execute(
                    f"CREATE TABLE silver_meta_history AS {transform_sql}"
                )
                logger.info("Created silver_meta_history via SQL path")
            else:
                self._silver_con.execute(f"""
                    INSERT INTO silver_meta_history
                    SELECT src.*
                    FROM ({transform_sql}) src
                    LEFT JOIN silver_meta_history t
                        ON  t.id            = src.id
                        AND t.snapshot_date = src.snapshot_date
                    WHERE t.id IS NULL
                """)
                logger.info("Appended to silver_meta_history via SQL path")
        except duckdb.Error as e:
            logger.error("Failed to append silver_meta_history via SQL: %s", e)
            raise StorageWriteError(f"Failed to append silver_meta_history: {e}") from e
        finally:
            try:
                self._silver_con.execute("DETACH _bronze")
            except duckdb.Error:
                pass

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    def _pipeline(
        self, update: bool, report_path: str = "data/silver/transform_report.json"
    ) -> None:
        """Run the full Bronze → Silver transformation pipeline.

        After writing silver_cards, runs an oracle ID name conflict check
        (EDA-01 §7): logs a warning if any card name maps to more than one
        oracle_id, which signals a split card handling regression.

        Args:
            update: Unused — silver_cards always does a full_load (Scryfall is a
                complete daily snapshot, so orphan rows from previous runs must
                not persist). History tables use append() regardless.
            report_path: File path where the transformation report will be written.
        """
        all_issues: list[dict[str, object]] = []
        bronze_data = self._load_bronze_data()
        transformed: dict[str, pd.DataFrame] = {}

        for source_name, source_config in self._config["sources"].items():
            bronze_table = f"bronze_{source_name}"
            if bronze_table not in bronze_data:
                logger.warning("Bronze table %r not found — skipping", bronze_table)
                continue
            logger.progress("Transforming %r", bronze_table)
            df, issues = self._transforms.transform(
                bronze_data[bronze_table], source_config
            )
            all_issues.extend(issues)
            transformed[source_name] = df

        # Cards — delegate to _join_cards wrapper for patchability in tests
        cards_df = self._join_cards(transformed)
        cards_df = self._transforms._extract_legality_features(cards_df, all_issues)
        null_oracle = (
            int(cards_df["oracle_id"].isna().sum())
            if "oracle_id" in cards_df.columns
            else 0
        )
        logger.debug(
            "silver_cards: %d cards joined (%d without oracle_id)",
            len(cards_df),
            null_oracle,
        )
        # Scryfall delivers a complete daily snapshot, so silver_cards is always
        # rebuilt from scratch.  An upsert would leave orphan rows (e.g. tokens
        # from older runs before the layout filter was added, or cards removed
        # from Scryfall) that silently corrupt oracle_id uniqueness checks.
        self._writer.full_load(cards_df, "silver_cards")

        # Oracle ID name conflict check (EDA-01 §7). A card name mapping to more
        # than one oracle_id signals a split card handling regression — the front and
        # back faces of a split card should share one oracle_id, not create two.
        if (
            not cards_df.empty
            and "name" in cards_df.columns
            and "oracle_id" in cards_df.columns
        ):
            conflicts = (
                cards_df[cards_df["oracle_id"].notna() & cards_df["name"].notna()]
                .groupby("name")["oracle_id"]
                .nunique()
            )
            conflicts = conflicts[conflicts > 1]
            if conflicts.empty:
                logger.info("Oracle ID conflict check: 0 conflicts")
            else:
                logger.warning(
                    "Oracle ID conflict check: %d name(s) map to multiple oracle_ids"
                    " — split card handling may have regressed. Examples: %s",
                    len(conflicts),
                    list(conflicts.head(5).index),
                )

        # Meta history — DuckDB SQL path: ATTACHes Bronze, transforms, and
        # dedup-inserts; filtered by silver_cards when available.
        self._append_meta_history_sql()
        self._silver_con.execute("CHECKPOINT")

        # Prices history (canonical / English cards)
        today = datetime.date.today().isoformat()
        prices_df = self._prices.build(today)
        logger.debug(
            "silver_prices_history: %d price records for %s", len(prices_df), today
        )
        self._writer.append(
            prices_df, "silver_prices_history", key_column="scryfall_id"
        )

        # Language variant prices (non-English Scryfall cards linked to canonical UUID)
        lang_prices_df = self._prices.build_language_prices(today)
        logger.debug(
            "silver_language_prices_history: %d language price records for %s",
            len(lang_prices_df),
            today,
        )
        self._writer.append(
            lang_prices_df, "silver_language_prices_history", key_column="scryfall_id"
        )

        self._append_format_staples_history()
        self._append_tournament_results_history()

        write_report(all_issues, report_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _join_cards(self, cards: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Guard-and-delegate wrapper around SilverCardJoin.join.

        Checks that both required sources are present before delegating;
        returns an empty DataFrame with a warning if either is missing.
        Called by _pipeline and patchable in tests that need a controlled
        cards_df without running the full join.

        Args:
            cards: Transformed DataFrames keyed by source name.

        Returns:
            Merged silver_cards DataFrame, or an empty DataFrame if either
            source is absent.
        """
        if "mtgjson_cards" not in cards or "scryfall_cards" not in cards:
            missing = [s for s in ("mtgjson_cards", "scryfall_cards") if s not in cards]
            logger.warning("Cannot join cards — missing sources: %s", missing)
            return pd.DataFrame()
        return self._card_join.join(cards["mtgjson_cards"], cards["scryfall_cards"])
