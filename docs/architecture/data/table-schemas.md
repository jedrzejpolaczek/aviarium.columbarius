# Table Schemas

Per-table column reference for all 22 DuckDB tables across Bronze, Silver, and Gold layers.

**Conventions:**
- *Grain* — what one row represents
- *Updated* — `full_load` (drop+replace), `append` (daily rows added), `upsert` (delete+insert by key)
- Types are DuckDB types as stored; `VARCHAR` means a raw JSON string where noted
- `?` after a type = nullable

For data flow between tables, see [data-lineage.md](data-lineage.md).
For term definitions, see [glossary.md](glossary.md).

---

## Bronze Layer

Raw ingestion from external sources. Bronze tables store Pydantic model dumps exactly as received; only Pydantic validation (type coercion, required-field checks) is applied. Columns marked "JSON" are stored as VARCHAR containing serialised JSON.

### bronze_scryfall_cards

**Grain:** 1 row per Scryfall card printing (unique `id`)
**Updated:** upsert on `id`
**Source:** Scryfall bulk API — all-cards endpoint

| Column | Type | Description |
|--------|------|-------------|
| id | VARCHAR | Scryfall UUID for this printing. Primary key. |
| lang | VARCHAR | Language code, e.g. `en`, `ja`, `de` |
| name | VARCHAR | Full card name; multi-face cards use ` // ` separator |
| oracle_id | VARCHAR? | Oracle identity UUID; shared across all reprints of the same oracle card |
| layout | VARCHAR | Card layout: `normal`, `transform`, `split`, `flip`, `meld`, `saga`, etc. |
| set | VARCHAR | Set code, e.g. `10E`, `blb` |
| set_type | VARCHAR | Set category: `core`, `expansion`, `masters`, `draft_innovation`, etc. |
| rarity | VARCHAR | `common`, `uncommon`, `rare`, `mythic`, `special`, `bonus` |
| collector_number | VARCHAR | Collector number within set; may contain non-numeric characters |
| type_line | VARCHAR? | Full type line as printed, e.g. `Legendary Creature — Wizard` |
| mana_cost | VARCHAR? | Mana cost string, e.g. `{3}{U}{U}`. Absent for some cards. |
| cmc | FLOAT? | Converted mana cost (mana value). Absent for reversible_card layout. |
| colors | VARCHAR | JSON array of colour codes: `["W","U"]`. Empty array for colorless. |
| color_identity | VARCHAR | JSON array of Commander colour identity codes |
| legalities | VARCHAR | JSON object mapping format names to legality strings (`legal`, `banned`, `not_legal`, `restricted`) |
| finishes | VARCHAR | JSON array of available finishes: `nonfoil`, `foil`, `etched` |
| prices | VARCHAR | JSON object (ScryfallPrices): `{eur, eur_foil, usd, usd_foil, tix, …}` |
| edhrec_rank | INTEGER? | EDHREC Commander popularity rank at time of ingest. Snapshot copy — use `silver_meta_history` for time-series. |
| reserved | BOOLEAN | True if this card is on the Reserved List |
| digital | BOOLEAN | True if this was a digital-only release (Arena/MTGO) |
| oversized | BOOLEAN | True if this is an oversized card |
| promo | BOOLEAN | True if this is a promotional printing |
| reprint | BOOLEAN | True if this card has appeared in a prior set |
| full_art | BOOLEAN | True if this card has full-art treatment |
| textless | BOOLEAN | True if printed without rules text |
| promo_types | VARCHAR? | JSON array of promo categories, e.g. `["prerelease","datestamped"]` |
| booster | BOOLEAN | True if found in booster packs |
| variation | BOOLEAN | True if this is a variant of another printing in the same set |
| variation_of | VARCHAR? | Scryfall UUID of the printing this card is a variation of |

> **Note:** `bronze_scryfall_cards` stores the full `ScryfallCard` Pydantic model dump (~50 columns total). The table above documents only the columns used by downstream Silver transformations. See `src/data/dataclasses/scryfall.py` for the complete model.

---

### bronze_scryfall_meta_history

**Grain:** 1 row per (scryfall_id, snapshot_date)
**Updated:** append — deduplication on (id, snapshot_date)
**Source:** Daily snapshot of selected fields from `bronze_scryfall_cards`

| Column | Type | Description |
|--------|------|-------------|
| id | VARCHAR | Scryfall UUID (foreign key → `bronze_scryfall_cards.id`) |
| snapshot_date | DATE | Date this snapshot was taken (ISO 8601) |
| legalities | VARCHAR | JSON object: format→legality mapping |
| edhrec_rank | INTEGER? | EDHREC rank on this date |
| reserved | BOOLEAN | Reserved List status on this date |
| promo_types | VARCHAR? | JSON array of promo categories |
| finishes | VARCHAR | JSON array of available finishes |

---

### bronze_scryfall_prices_history

**Grain:** 1 row per (id, snapshot_date)
**Updated:** append — deduplication on (id, snapshot_date)
**Source:** Daily snapshot of the `prices` field from `bronze_scryfall_cards`

| Column | Type | Description |
|--------|------|-------------|
| id | VARCHAR | Scryfall UUID |
| snapshot_date | VARCHAR | Date this price snapshot was taken |
| eur | FLOAT? | EUR non-foil price |
| eur_foil | FLOAT? | EUR foil price |
| usd | FLOAT? | USD non-foil price |
| usd_foil | FLOAT? | USD foil price |
| tix | FLOAT? | MTGO ticket price (captured in Bronze; not propagated to Silver) |

---

### bronze_mtgjson_cards

**Grain:** 1 row per MTGJson card UUID
**Updated:** full_load (replace on each populate/update)
**Source:** MTGJson AllPrintings.json

| Column | Type | Description |
|--------|------|-------------|
| uuid | VARCHAR | MTGJson UUID for this printing. Primary key. |
| name | VARCHAR | Full card name |
| set_code | VARCHAR | Set code, e.g. `10E`, `BLB` |
| number | VARCHAR | Collector number within set |
| language | VARCHAR | Language name, e.g. `English`, `Japanese` |
| layout | VARCHAR | Card layout |
| mana_value | FLOAT | Numeric converted mana cost |
| rarity | VARCHAR | Rarity: `common`, `uncommon`, `rare`, `mythic`, `special`, `bonus` |
| types | VARCHAR | JSON array of card types, e.g. `["Legendary","Creature"]` |
| supertypes | VARCHAR | JSON array of supertypes, e.g. `["Legendary","Snow"]` |
| subtypes | VARCHAR | JSON array of subtypes, e.g. `["Wizard","Forest"]` |
| colors | VARCHAR | JSON array of colour codes |
| color_identity | VARCHAR | JSON array of Commander colour identity codes |
| legalities | VARCHAR | JSON object: format→legality (MTGJson uses title-case: `Legal`, `Banned`) |
| finishes | VARCHAR | JSON array of available finishes |
| identifiers | VARCHAR | JSON object (MtgjsonIdentifiers): cross-site IDs including `scryfall_id` |
| variations | VARCHAR? | JSON array of MTGJson UUIDs of other printings that are variants within the same set |
| is_reserved | BOOLEAN? | True if on the Reserved List |
| is_reprint | BOOLEAN? | True if printed in a prior set |
| is_promo | BOOLEAN? | True if promotional printing |
| is_full_art | BOOLEAN? | True if full-art treatment |
| is_textless | BOOLEAN? | True if printed without rules text |
| edhrec_rank | INTEGER? | EDHREC rank at time of ingest |
| edhrec_saltiness | FLOAT? | EDHREC saltiness score (0.0–1.0+) |

> **Note:** `bronze_mtgjson_cards` stores the full `MtgjsonCard` Pydantic model dump (~60 columns total). See `src/data/dataclasses/mtgjson.py` for the complete model.

---

### bronze_mtgjson_prices_history

**Grain:** 1 row per (uuid, snapshot_date, retailer, tx_type, finish)
**Updated:** append — deduplication on (uuid, snapshot_date, retailer, tx_type, finish)
**Source:** MTGJson AllPricesToday.json (daily snapshot) + AllPrices.json (one-time 90-day seed)

EAV schema — one row per price point. All retailers present in the MTGJson feed are captured
without pre-selection. Silver pivots to wide columns via `CASE WHEN` SQL.

| Column | Type | Description |
|--------|------|-------------|
| uuid | VARCHAR | MTGJson UUID |
| snapshot_date | VARCHAR | Date this price was recorded |
| retailer | VARCHAR | Source retailer (cardmarket, tcgplayer, …) |
| tx_type | VARCHAR | Transaction type: `retail` or `buylist` |
| finish | VARCHAR | Card finish: `normal`, `foil`, `etched` |
| price | FLOAT? | Price in retailer's native currency |

---

### bronze_format_staples_history

**Grain:** 1 row per (composite_id, snapshot_date) where composite_id = `{card_name}__{format}`
**Updated:** append — deduplication on (id, snapshot_date)
**Source:** Daily snapshot of `FormatStaple` records scraped from MTGGoldfish

| Column | Type | Description |
|--------|------|-------------|
| id | VARCHAR | Composite key: `{card_name}__{format}` |
| snapshot_date | DATE | Date this staple snapshot was taken |
| card_name | VARCHAR | Card name as displayed on MTGGoldfish |
| format | VARCHAR | Format: `commander`, `modern`, `legacy`, `vintage`, `standard` |
| deck_pct | FLOAT | Percentage of competitive decks running this card |
| percentage_in_decks | INTEGER | Integer version of deck_pct (e.g. 42 for 42%) |
| played | FLOAT | Average number of copies played per deck that includes it |
| top | INTEGER | Rank within the format staple list (1 = most played) |

---

### bronze_tournament_results

**Grain:** 1 row per (tournament_id, card_name, is_sideboard)
**Updated:** upsert on `id`
**Source:** MTGTop8 top-8 tournament decklists

| Column | Type | Description |
|--------|------|-------------|
| id | VARCHAR | Composite PK: `{tournament_id}__{card_name}__{is_sideboard}` |
| tournament_id | VARCHAR | Tournament identifier, e.g. `mtgtop8_99999` |
| tournament_date | VARCHAR | Tournament date as ISO string (YYYY-MM-DD) |
| format | VARCHAR | Format: `modern`, `legacy`, `vintage`, `standard`, `pioneer` |
| event_name | VARCHAR | Name of the tournament event |
| placement | INTEGER | Top-8 placement (1–8) |
| player | VARCHAR | Player name |
| deck_name | VARCHAR | Archetype name, e.g. `Rhinos`, `Murktide` |
| card_name | VARCHAR | Card face name as listed in the decklist |
| copies | INTEGER | Number of copies in the deck |
| is_sideboard | BOOLEAN | True if in the sideboard, False if in the main deck |

---

## Silver Layer

Cleaned and merged data. `silver_cards` is the central card reference table (MTGJson + Scryfall joined). History tables accumulate one row per card per day via deduplication append.

### silver_cards

**Grain:** 1 row per card printing
**Updated:** full_load (always a full rebuild — Scryfall is a daily snapshot)
**Source:** Merge of `bronze_mtgjson_cards` + `bronze_scryfall_cards`

| Column | Type | Description |
|--------|------|-------------|
| uuid | VARCHAR? | MTGJson UUID. NULL for Scryfall-only cards (digital, oversized, language variants without a direct MTGJson match) |
| scryfall_id | VARCHAR? | Scryfall UUID |
| oracle_id | VARCHAR? | Scryfall oracle UUID (shared across all reprints of the same oracle card) |
| name | VARCHAR | Card name |
| set_code | VARCHAR? | Set code |
| language | VARCHAR? | Language name, e.g. `English`, `Japanese` |
| canonical_uuid | VARCHAR? | MTGJson UUID of the English printing. Used as fallback join key for language variants (uuid IS NULL) |
| rarity | VARCHAR? | Rarity |
| mana_value | FLOAT? | Numeric converted mana cost (capped at 20 in Gold) |
| finishes | VARCHAR[]? | DuckDB native array of finish strings |
| colors | VARCHAR[]? | DuckDB native array of colour codes |
| color_identity | VARCHAR[]? | DuckDB native array of Commander colour identity codes |
| variations | VARCHAR[]? | DuckDB native array of UUID strings for variant printings |
| original_supertypes | VARCHAR[]? | DuckDB native array of supertypes from MTGJson; used to derive `is_legendary` in Gold |
| is_reserved | BOOLEAN? | Reserved List membership |
| is_reprint | BOOLEAN? | Whether this is a reprint |
| is_promo | BOOLEAN? | Whether this is a promo printing |
| is_full_art | BOOLEAN? | Full-art treatment |
| is_textless | BOOLEAN? | Printed without rules text |
| edhrec_saltiness | FLOAT? | EDHREC saltiness score from MTGJson |
| set_type | VARCHAR? | Set type from Scryfall |
| is_commander_legal | BOOLEAN? | Computed from legalities: True if `commander = 'Legal'` |
| is_standard_legal | BOOLEAN? | Computed from legalities: True if `standard = 'Legal'` |
| is_modern_legal | BOOLEAN? | Computed from legalities: True if `modern = 'Legal'` |
| is_legacy_legal | BOOLEAN? | Computed from legalities: True if `legacy = 'Legal'` |
| format_count | INTEGER? | Count of formats where this card is legal (out of 5 tracked) |

---

### silver_meta_history

**Grain:** 1 row per (scryfall_id, snapshot_date)
**Updated:** append — deduplication on (id, snapshot_date); filtered to IDs present in `silver_cards`
**Source:** `bronze_scryfall_meta_history` filtered to paper non-digital cards

| Column | Type | Description |
|--------|------|-------------|
| id | VARCHAR | Scryfall UUID |
| snapshot_date | DATE | Snapshot date |
| legalities | VARCHAR | JSON object: format→legality |
| edhrec_rank | INTEGER? | EDHREC rank on this date |
| reserved | BOOLEAN | Reserved List status |
| promo_types | VARCHAR? | JSON array of promo categories |
| finishes | VARCHAR | JSON array of finishes |

---

### silver_prices_history

**Grain:** 1 row per (uuid, snapshot_date)
**Updated:** append — deduplication on (uuid, snapshot_date); forward-filled for all-NULL price rows
**Source:** `bronze_scryfall_prices_history` (EUR/USD prices) + `bronze_mtgjson_prices_history` (Cardmarket/TCGPlayer/Card Kingdom/ManaPool prices); joined to `silver_cards` for UUID resolution

| Column | Type | Description |
|--------|------|-------------|
| uuid | VARCHAR | MTGJson UUID (COALESCE of direct match and canonical_uuid fallback) |
| scryfall_id | VARCHAR | Scryfall UUID |
| snapshot_date | DATE | Price snapshot date |
| eur | FLOAT? | Scryfall non-foil EUR price |
| eur_foil | FLOAT? | Scryfall foil EUR price |
| usd | FLOAT? | Scryfall non-foil USD price |
| usd_foil | FLOAT? | Scryfall foil USD price |
| cardmarket_eur | FLOAT? | Cardmarket retail non-foil EUR (from MTGJson) |
| cardmarket_eur_foil | FLOAT? | Cardmarket retail foil EUR (from MTGJson) |
| cardmarket_buylist_eur | FLOAT? | Cardmarket buylist EUR (from MTGJson; currently 100% NULL) |
| tcgplayer_usd | FLOAT? | TCGPlayer retail non-foil USD (from MTGJson) |
| tcgplayer_usd_foil | FLOAT? | TCGPlayer retail foil USD (from MTGJson) |
| tcgplayer_buylist_usd | FLOAT? | TCGPlayer buylist USD (from MTGJson; currently 100% NULL) |
| cardkingdom_usd | FLOAT? | Card Kingdom retail non-foil USD (from MTGJson) |
| cardkingdom_usd_foil | FLOAT? | Card Kingdom retail foil USD (from MTGJson) |
| cardkingdom_buylist_usd | FLOAT? | Card Kingdom buylist non-foil USD (from MTGJson) |
| cardkingdom_buylist_usd_foil | FLOAT? | Card Kingdom buylist foil USD (from MTGJson) |
| manapool_usd | FLOAT? | ManaPool retail non-foil USD (from MTGJson; no buylist source) |
| manapool_usd_foil | FLOAT? | ManaPool retail foil USD (from MTGJson; no buylist source) |

---

### silver_language_prices_history

**Grain:** 1 row per (scryfall_id, snapshot_date) for non-English language variants
**Updated:** append — deduplication via forward-fill logic
**Source:** `bronze_scryfall_prices_history` filtered to `silver_cards` rows where `uuid IS NULL AND canonical_uuid IS NOT NULL`

| Column | Type | Description |
|--------|------|-------------|
| scryfall_id | VARCHAR | Scryfall UUID of the language variant |
| canonical_uuid | VARCHAR | MTGJson UUID of the English printing |
| lang | VARCHAR | Language code from Scryfall (e.g. `ja`, `de`) |
| snapshot_date | DATE | Price snapshot date |
| eur | FLOAT? | Non-foil EUR price |
| eur_foil | FLOAT? | Foil EUR price |
| usd | FLOAT? | Non-foil USD price |
| usd_foil | FLOAT? | Foil USD price |

---

### silver_format_staples_history

**Grain:** 1 row per (composite_id, snapshot_date)
**Updated:** append — same columns as `bronze_format_staples_history`
**Source:** Direct copy of `bronze_format_staples_history`

| Column | Type | Description |
|--------|------|-------------|
| id | VARCHAR | Composite key: `{card_name}__{format}` |
| snapshot_date | DATE | Snapshot date |
| card_name | VARCHAR | Card name as on MTGGoldfish |
| format | VARCHAR | Format |
| deck_pct | FLOAT | % of decks running this card |
| percentage_in_decks | INTEGER | Integer % |
| played | FLOAT | Average copies per including deck |
| top | INTEGER | Rank in the format staple list |

---

### silver_tournament_results_history

**Grain:** 1 row per tournament result row
**Updated:** append
**Source:** `bronze_tournament_results` with `oracle_id` and `scryfall_id` added via join to `silver_cards` on normalized card name

| Column | Type | Description |
|--------|------|-------------|
| id | VARCHAR | Composite PK from Bronze |
| tournament_id | VARCHAR | Tournament identifier |
| tournament_date | VARCHAR | ISO date string |
| format | VARCHAR | Format |
| event_name | VARCHAR | Tournament event name |
| placement | INTEGER | Top-8 placement (1–8) |
| player | VARCHAR | Player name |
| deck_name | VARCHAR | Archetype name |
| card_name | VARCHAR | Card name |
| copies | INTEGER | Copies in deck |
| is_sideboard | BOOLEAN | True if sideboard |
| oracle_id | VARCHAR? | Oracle UUID resolved via `silver_cards` join |
| scryfall_id | VARCHAR? | Scryfall UUID resolved via `silver_cards` join |

---

## Gold Layer

ML-ready feature tables. Built entirely from Silver data using window functions, joins, and aggregations. **All Gold tables are fully rebuilt on every pipeline run** (both `populate()` and `update()`) because window features span the full price history and cannot be patched incrementally.

### gold_card_features

**Grain:** 1 row per card printing (uuid)
**Source:** `silver_cards` WHERE uuid IS NOT NULL

| Column | Type | Description |
|--------|------|-------------|
| uuid | VARCHAR | MTGJson UUID. Primary key. |
| scryfall_id | VARCHAR? | Scryfall UUID |
| oracle_id | VARCHAR? | Oracle UUID |
| name | VARCHAR | Card name |
| set_code | VARCHAR | Set code |
| rarity | VARCHAR? | Rarity |
| mana_value | FLOAT? | Mana value, capped at 20 (raw values above 20 indicate corrupted Bronze entries) |
| is_reserved | BOOLEAN? | Reserved List membership |
| is_reprint | BOOLEAN? | Reprint flag |
| is_promo | BOOLEAN? | Promo flag |
| is_full_art | BOOLEAN? | Full-art flag |
| is_textless | BOOLEAN? | Textless flag |
| edhrec_saltiness | FLOAT? | EDHREC saltiness |
| set_type | VARCHAR? | Set type |
| finish_count | INTEGER | Number of available finishes (1–3) |
| has_etched_finish | BOOLEAN | True if etched finish is available |
| color_count | INTEGER | Number of colours |
| color_identity_count | INTEGER | Size of Commander colour identity |
| variation_count | INTEGER | Number of variant printings in the same set |
| is_legendary | BOOLEAN | Derived from `original_supertypes`: True if "Legendary" is in supertypes |
| is_commander_legal | BOOLEAN? | Legal in Commander |
| is_standard_legal | BOOLEAN? | Legal in Standard |
| is_modern_legal | BOOLEAN? | Legal in Modern |
| is_legacy_legal | BOOLEAN? | Legal in Legacy |
| format_count | INTEGER? | Count of legal formats (0–5) |
| print_count | INTEGER | Number of distinct printings of this oracle card |

---

### gold_price_features

**Grain:** 1 row per (uuid, snapshot_date)
**Source:** `silver_prices_history` + optional LEFT JOIN `silver_meta_history` for EDHREC rank

| Column | Type | Description |
|--------|------|-------------|
| uuid | VARCHAR | MTGJson UUID |
| scryfall_id | VARCHAR? | Scryfall UUID |
| snapshot_date | DATE | Price snapshot date |
| eur | FLOAT? | Non-foil EUR price |
| eur_foil | FLOAT? | Foil EUR price |
| usd | FLOAT? | Non-foil USD price |
| usd_foil | FLOAT? | Foil USD price |
| cardmarket_eur | FLOAT? | Cardmarket EUR |
| cardmarket_eur_foil | FLOAT? | Cardmarket foil EUR |
| tcgplayer_usd | FLOAT? | TCGPlayer USD |
| tcgplayer_usd_foil | FLOAT? | TCGPlayer foil USD |
| edhrec_rank | INTEGER? | EDHREC rank on this date (time-aligned from `silver_meta_history`) |
| price_7d_avg | FLOAT? | 7-day rolling average of EUR price (row-based window) |
| price_30d_avg | FLOAT? | 30-day rolling average of EUR price |
| price_change_1d_abs | FLOAT? | EUR change vs previous row |
| price_change_7d_abs | FLOAT? | EUR change vs 7 rows ago |
| price_change_30d_abs | FLOAT? | EUR change vs 30 rows ago |
| price_change_1d_pct | FLOAT? | 1-day EUR percentage change |
| price_change_7d_pct | FLOAT? | 7-day EUR percentage change |
| price_change_30d_pct | FLOAT? | 30-day EUR percentage change |
| price_volatility_30d | FLOAT? | Standard deviation of EUR over 30 rows |
| foil_premium | FLOAT? | eur_foil / eur ratio |
| price_ath | FLOAT? | All-time high EUR price (bounded: only past + current rows) |
| price_atl | FLOAT? | All-time low EUR price (bounded) |
| days_with_price | INTEGER? | Count of rows with a non-NULL EUR price up to and including this row |
| days_since_last_real_price | INTEGER? | Days since the last non-NULL EUR snapshot |
| price_rank_global | INTEGER? | EUR rank among all cards on this date (rank 1 = most expensive) |
| is_price_spike | BOOLEAN? | True if day-over-day EUR change exceeds 300% |

---

### gold_language_premiums

**Grain:** 1 row per (scryfall_id, snapshot_date) for non-English language variants
**Source:** `silver_language_prices_history` INNER JOIN `silver_prices_history` on (canonical_uuid, snapshot_date)

| Column | Type | Description |
|--------|------|-------------|
| scryfall_id | VARCHAR | Scryfall UUID of the language variant |
| canonical_uuid | VARCHAR | MTGJson UUID of the English printing |
| lang | VARCHAR | Language code |
| snapshot_date | DATE | Snapshot date |
| lang_eur | FLOAT? | Language variant non-foil EUR price |
| lang_eur_foil | FLOAT? | Language variant foil EUR price |
| lang_usd | FLOAT? | Language variant non-foil USD price |
| lang_usd_foil | FLOAT? | Language variant foil USD price |
| canonical_eur | FLOAT? | English printing non-foil EUR price |
| canonical_eur_foil | FLOAT? | English printing foil EUR price |
| eur_lang_premium | FLOAT? | lang_eur / canonical_eur. NULL when either price is 0 or absent. |
| eur_foil_lang_premium | FLOAT? | lang_eur_foil / canonical_eur_foil |

---

### gold_demand_signals

**Grain:** 1 row per (scryfall_id, snapshot_date)
**Source:** `silver_meta_history` — legality transitions detected by comparing consecutive rows

| Column | Type | Description |
|--------|------|-------------|
| id | VARCHAR | Scryfall UUID |
| snapshot_date | DATE | Snapshot date |
| edhrec_rank | INTEGER? | EDHREC rank on this date |
| edhrec_rank_change | FLOAT? | edhrec_rank minus previous row's rank. NULL on first row per card. Negative = rank improved. |
| commander_legality | VARCHAR? | Raw legality string for Commander |
| standard_legality | VARCHAR? | Raw legality string for Standard |
| modern_legality | VARCHAR? | Raw legality string for Modern |
| legacy_legality | VARCHAR? | Raw legality string for Legacy |
| vintage_legality | VARCHAR? | Raw legality string for Vintage |
| commander_banned | BOOLEAN | True on the date a Commander ban took effect (legal→banned transition) |
| standard_banned | BOOLEAN | True on the date a Standard ban took effect |
| modern_banned | BOOLEAN | True on the date a Modern ban took effect |
| legacy_banned | BOOLEAN | True on the date a Legacy ban took effect |
| vintage_banned | BOOLEAN | True on the date a Vintage ban took effect |
| commander_unbanned | BOOLEAN | True on the date a Commander unban took effect (banned→legal transition) |
| standard_unbanned | BOOLEAN | True on the date a Standard unban took effect |
| modern_unbanned | BOOLEAN | True on the date a Modern unban took effect |
| legacy_unbanned | BOOLEAN | True on the date a Legacy unban took effect |
| vintage_unbanned | BOOLEAN | True on the date a Vintage unban took effect |

---

### gold_events

**Grain:** 1 row per (event_date, format, event_type)
**Source:** `silver_meta_history` — aggregated legality transitions across all cards

| Column | Type | Description |
|--------|------|-------------|
| event_date | DATE | Date the ban or unban took effect |
| format | VARCHAR | Affected format |
| event_type | VARCHAR | `'ban'` or `'unban'` |
| card_count | INTEGER | Number of cards affected by this event |

---

### gold_format_staples

**Grain:** 1 row per (card, format, snapshot_date)
**Source:** `silver_format_staples_history` — window functions over full history

| Column | Type | Description |
|--------|------|-------------|
| id | VARCHAR | Composite key: `{card_name}__{format}` |
| card_name | VARCHAR | Card name (MTGGoldfish display name) |
| format | VARCHAR | Format |
| snapshot_date | DATE | Snapshot date |
| deck_pct | FLOAT? | % of decks running this card on this date |
| played | FLOAT? | Average copies per including deck |
| top | INTEGER? | Rank within format staple list |
| deck_pct_7d_avg | FLOAT? | 7-row rolling average of deck_pct |
| deck_pct_30d_avg | FLOAT? | 30-row rolling average of deck_pct |
| deck_pct_change_7d | FLOAT? | deck_pct minus value 7 rows ago |
| deck_pct_change_30d | FLOAT? | deck_pct minus value 30 rows ago |

---

### gold_ban_price_impact

**Grain:** 1 row per (scryfall_id, format, event_type, event_date)
**Source:** `silver_meta_history` (ban/unban events) + `silver_prices_history` (price windows)

| Column | Type | Description |
|--------|------|-------------|
| scryfall_id | VARCHAR | Scryfall UUID |
| format | VARCHAR | Format of the ban/unban event |
| event_type | VARCHAR | `'ban'` or `'unban'` |
| event_date | DATE | Date the ban/unban took effect |
| price_30d_before | FLOAT? | Average EUR price in the 30 days before the event. NULL if fewer than 30 days of history exist. |
| price_7d_before | FLOAT? | Average EUR price in the 7 days before the event |
| price_at_event | FLOAT? | EUR price on the event date |
| price_7d_after | FLOAT? | Average EUR price in the 7 days after the event |
| price_30d_after | FLOAT? | Average EUR price in the 30 days after the event |
| price_change_7d_pct | FLOAT? | (price_7d_after − price_7d_before) / price_7d_before |
| price_change_30d_pct | FLOAT? | (price_30d_after − price_30d_before) / price_30d_before |

---

### gold_tournament_signals

**Grain:** 1 row per (oracle_id, format)
**Source:** `silver_tournament_results_history` — aggregated over 30-day and 90-day windows from current date

| Column | Type | Description |
|--------|------|-------------|
| oracle_id | VARCHAR | Oracle UUID |
| scryfall_id | VARCHAR? | Scryfall UUID (MIN per oracle_id group — any representative printing) |
| format | VARCHAR | Format |
| top8_appearances_30d | INTEGER | Distinct main-deck tournaments in the last 30 days |
| top8_appearances_90d | INTEGER | Distinct main-deck tournaments in the last 90 days |
| top8_copies_avg | FLOAT? | Average main-deck copies per including tournament |
| sideboard_appearances_30d | INTEGER | Distinct sideboard tournaments in the last 30 days |
| main_deck_pct | FLOAT? | % of appearances in the last 90 days where the card was in the main deck |
| last_top8_date | DATE? | Date of most recent main-deck top-8 appearance |

---

### gold_ml_dataset

**Grain:** 1 row per (uuid, snapshot_date) — same spine as `gold_price_features`
**Source:** `gold_price_features` spine + LEFT JOINs to `gold_card_features`, `gold_demand_signals`, `gold_tournament_signals` (aggregated), `gold_format_staples` (pivoted by format)

Contains all columns from `gold_price_features`, plus:

**Price targets (computed via self-join on gold_price_features):**

| Column | Type | Description |
|--------|------|-------------|
| target_price_7d | FLOAT? | EUR price 7 days after this row's snapshot_date |
| target_price_30d | FLOAT? | EUR price 30 days after this row's snapshot_date |
| target_change_30d | VARCHAR? | `'up'` (>+20%), `'down'` (<-20%), or `'flat'` (otherwise). NULL if target_price_30d is NULL. |

**From gold_card_features (static per printing):**

| Column | Type |
|--------|------|
| rarity | VARCHAR? |
| mana_value | FLOAT? |
| is_reserved | BOOLEAN? |
| is_reprint | BOOLEAN? |
| color_count | INTEGER? |
| color_identity_count | INTEGER? |
| is_commander_legal | BOOLEAN? |
| is_modern_legal | BOOLEAN? |
| is_legacy_legal | BOOLEAN? |
| is_standard_legal | BOOLEAN? |
| format_count | INTEGER? |
| print_count | INTEGER? |
| finish_count | INTEGER? |
| has_etched_finish | BOOLEAN? |
| edhrec_saltiness | FLOAT? |
| set_type | VARCHAR? |

**From gold_demand_signals (time-aligned):**

| Column | Type |
|--------|------|
| commander_banned | BOOLEAN? |
| modern_banned | BOOLEAN? |
| legacy_banned | BOOLEAN? |
| standard_banned | BOOLEAN? |
| commander_unbanned | BOOLEAN? |
| modern_unbanned | BOOLEAN? |
| edhrec_rank_change | FLOAT? |

**From gold_tournament_signals (cross-sectional, no date dimension):**

| Column | Type | Description |
|--------|------|-------------|
| top8_30d_total | INTEGER? | SUM(top8_appearances_30d) across all formats |
| top8_90d_total | INTEGER? | SUM(top8_appearances_90d) across all formats |
| top8_copies_avg | FLOAT? | AVG(top8_copies_avg) across all formats |

**From gold_format_staples (pivoted, time-aligned):**

| Column | Type | Description |
|--------|------|-------------|
| staple_pct_commander | FLOAT? | Commander deck_pct on this date |
| staple_7d_commander | FLOAT? | Commander deck_pct_7d_avg on this date |
| staple_pct_modern | FLOAT? | Modern deck_pct on this date |
| staple_pct_legacy | FLOAT? | Legacy deck_pct on this date |
| staple_pct_vintage | FLOAT? | Vintage deck_pct on this date |
| staple_7d_vintage | FLOAT? | Vintage deck_pct_7d_avg on this date |
