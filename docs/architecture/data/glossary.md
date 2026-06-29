# Data Glossary

Alphabetical definitions for domain terms used in table names, column names, ADRs, and code comments.

---

### Ban event
A legality transition where a card moves from `legal` to `banned` in a format between two consecutive daily snapshots. Detected by comparing `{format}_legality` in consecutive `silver_meta_history` rows.

**In the data:** `gold_demand_signals.{format}_banned` (BOOLEAN flag per card per day) · `gold_events.event_type = 'ban'` (aggregate count per format per day)

---

### Bronze layer
The raw ingestion layer. Tables store data exactly as received from external sources (Scryfall, MTGJson, MTGGoldfish, MTGTop8), with Pydantic validation applied but no business-logic transformations.

**In the data:** All `bronze_*` tables in `bronze.duckdb`

---

### Canonical UUID
The MTGJson `uuid` of the English paper printing of a card. Used as the primary join key between Scryfall and MTGJson data. Non-English language variants (uuid=NULL in `silver_cards`) link back to their English printing via `canonical_uuid`.

**In the data:** `silver_cards.canonical_uuid` · `silver_language_prices_history.canonical_uuid`

---

### Demand signal
A proxy metric for market interest in a card. Composed of ban/unban event flags (which cause price movements) and EDHREC rank change (which indicates growing/declining Commander community interest).

**In the data:** `gold_demand_signals` table — one row per (scryfall_id, snapshot_date)

---

### EDHREC rank
A daily integer rank representing a card's popularity in Commander (EDH) decks, as reported by Scryfall from EDHREC.com. Lower rank = more popular. Not all cards are ranked (NULL for many). Snapshotted daily in `silver_meta_history` and time-aligned into `gold_price_features`.

**In the data:** `silver_meta_history.edhrec_rank` · `gold_price_features.edhrec_rank` · `gold_demand_signals.edhrec_rank_change`

---

### EDHREC saltiness
A float score (0.0–1.0+) measuring how controversial a card is among Commander players, from EDHREC.com. Higher = more disliked by opponents. Stored as a static property of the printing.

**In the data:** `silver_cards.edhrec_saltiness` · `gold_card_features.edhrec_saltiness`

---

### Event type
The direction of a format legality change. `'ban'` = card was made illegal (legal→banned). `'unban'` = card was made legal again (banned→legal).

**In the data:** `gold_events.event_type` · `gold_ban_price_impact.event_type`

---

### Foil premium
The ratio of foil EUR price to non-foil EUR price for the same card on the same date. Values > 1.0 mean the foil trades at a premium. NULL when either price is absent.

**In the data:** `gold_price_features.foil_premium`

---

### Format
One of five competitive Magic formats tracked by the pipeline: `commander`, `standard`, `modern`, `legacy`, `vintage`. Format determines which cards are legal to play and influences demand signals and staple metrics.

**In the data:** `gold_demand_signals.{format}_banned/unbanned` · `gold_format_staples.format` · `gold_tournament_signals.format`

---

### Format staple
A card appearing in a significant percentage of competitive decks in a given format, as tracked by MTGGoldfish. The `deck_pct` column is the primary metric — what percentage of tournament-viable decks include this card on a given day.

**In the data:** `silver_format_staples_history` · `gold_format_staples`

---

### Gold layer
The ML-ready feature layer. Tables are built from Silver data using window functions, joins, and aggregations. All Gold tables are fully rebuilt on every pipeline run (no incremental updates) because window features span the full price history.

**In the data:** All `gold_*` tables in `gold.duckdb`

---

### Language premium
The ratio of a non-English language variant's EUR price to the canonical English printing's EUR price on the same date. Values > 1.0 mean the language variant trades at a premium (e.g., Japanese foils often trade above 1.0).

**In the data:** `gold_language_premiums.eur_lang_premium`

---

### Language variant
A non-English printing of a card. In `silver_cards`, language variants have `uuid = NULL` and `canonical_uuid` pointing to the English printing's MTGJson UUID. Their prices are tracked separately from the English canonical price.

**In the data:** `silver_cards` rows where `uuid IS NULL AND canonical_uuid IS NOT NULL` · `silver_language_prices_history`

---

### Oracle card
The abstract card identity, shared across all printings and reprints. Two cards with the same oracle text and name share an `oracle_id` even if they have different art, set codes, or collector numbers.

**In the data:** `silver_cards.oracle_id` · `gold_card_features.oracle_id` · `gold_tournament_signals.oracle_id`

---

### Price snapshot
One row of price data for a card on a specific date. Prices come from Scryfall (scalar `eur`, `eur_foil`, `usd`, `usd_foil` float columns in `bronze_scryfall_prices_history`) and MTGJson (EAV rows in `bronze_mtgjson_prices_history` pivoted to wide Cardmarket/TCGPlayer columns at the Silver layer via CASE WHEN SQL). Missing prices on a given day are forward-filled from the most recent prior snapshot.

**In the data:** `silver_prices_history` — grain: 1 row per (uuid, snapshot_date)

---

### Printing
A specific physical release of an oracle card in a set. Identified by the MTGJson `uuid` (primary key in the ML pipeline). The same oracle card printed in 10 sets = 10 printings, each with its own UUID and potentially different prices.

**In the data:** `silver_cards` · `gold_card_features` — keyed by `uuid`

---

### Silver layer
The cleaned and merged data layer. Raw Bronze data is deduplicated, normalized, and joined (MTGJson + Scryfall card catalogs merged into `silver_cards`). History tables accumulate one appended row per card per day with deduplication on (key, snapshot_date).

**In the data:** All `silver_*` tables in `silver.duckdb`

---

### Unban event
A legality transition where a card moves from `banned` to `legal` in a format between two consecutive daily snapshots. The reverse of a ban event.

**In the data:** `gold_demand_signals.{format}_unbanned` · `gold_events.event_type = 'unban'`
