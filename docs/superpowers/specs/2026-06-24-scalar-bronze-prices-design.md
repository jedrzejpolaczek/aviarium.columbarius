# Design: Scalar Bronze Price Tables

**Date:** 2026-06-24
**Status:** Approved, pending implementation

---

## Problem

`bronze_scryfall_prices_history` stores prices as a JSON VARCHAR column (`prices`).
`bronze_mtgjson_prices_history` stores prices as a JSON VARCHAR column (`paper`) with a nested `{retailer → tx_type → finish → date → value}` structure, plus a `mtgo` column that is never read downstream (excluded by ADR-012).

Silver's `SilverPriceBuilder` is forced to:
- Parse `prices` JSON with DuckDB `json_extract_string` (Scryfall — acceptable, but inconsistent)
- Parse `paper` JSON in a Python list comprehension (`_extract_all_prices`) using dynamic date-key lookup (MTGJson — violates ADR-024)

The inconsistency between two Bronze price tables with different storage strategies creates confusion and leaves a Python JSON parsing path in Silver where DuckDB should be the compute layer.

---

## Decision

Scalarise both Bronze price history tables at ingestion time. Silver reads scalar FLOAT columns directly — no JSON parsing in Silver for price data.

This extends ADR-024 (DuckDB compute layer) to the Bronze price ingestion boundary: price extraction happens once at ingest, not repeatedly at every Silver build.

---

## New Schemas

### `bronze_scryfall_prices_history`

**Grain:** 1 row per (scryfall_id, snapshot_date)

| Column        | Type    | Description                        |
|---------------|---------|------------------------------------|
| id            | VARCHAR | Scryfall UUID                      |
| snapshot_date | DATE    | Date this snapshot was taken       |
| eur           | FLOAT?  | EUR price (non-foil)               |
| eur_foil      | FLOAT?  | EUR price (foil)                   |
| usd           | FLOAT?  | USD price (non-foil)               |
| usd_foil      | FLOAT?  | USD price (foil)                   |

`tix` excluded per ADR-012 (physical cards only). `usd_etched` excluded — not consumed by Silver or Gold (YAGNI).

### `bronze_mtgjson_prices_history`

**Grain:** 1 row per (uuid, snapshot_date)

| Column                 | Type    | Description                                |
|------------------------|---------|--------------------------------------------|
| uuid                   | VARCHAR | MTGJson UUID                               |
| snapshot_date          | DATE    | Date this snapshot was taken               |
| cardmarket_eur         | FLOAT?  | Cardmarket retail normal EUR               |
| cardmarket_eur_foil    | FLOAT?  | Cardmarket retail foil EUR                 |
| cardmarket_buylist_eur | FLOAT?  | Cardmarket buylist normal EUR              |
| tcgplayer_usd          | FLOAT?  | TCGPlayer retail normal USD                |
| tcgplayer_usd_foil     | FLOAT?  | TCGPlayer retail foil USD                  |
| tcgplayer_buylist_usd  | FLOAT?  | TCGPlayer buylist normal USD               |

`mtgo` column dropped — never consumed by Silver or Gold, explicitly out of scope per ADR-012.

---

## Migration

### Strategy: Atomic table replacement from backup

A one-off migration script `scripts/migrate_bronze_prices.py` migrates both tables using the backup `bronze/cards_copy.duckdb` as source. The live database is never left in a partial state.

**Algorithm (per table):**
1. Open `cards_copy.duckdb` read-only; open live `bronze/cards.duckdb` read-write
2. Verify backup has the expected source tables
3. Create `<table>_new` with the scalar schema in the live DB
4. Read source rows in batches of 10 000; extract scalars in Python; INSERT into `<table>_new`
5. `DROP TABLE <table>` (old); `ALTER TABLE <table>_new RENAME TO <table>`
6. `CHECKPOINT` on live DB

If the script fails during step 4, the old table is untouched. `cards_copy.duckdb` is not deleted — it remains as backup until manually removed.

**Scalar extraction for MTGJson:**
Each `paper` JSON blob already contains only one date's prices per row (enforced by `_filter_prices_to_date` at seed time and by `AllPricesToday.json` structure at daily snapshot time). Extraction: find max date-key ≤ `snapshot_date` in the nested dict (same logic as the current `_extract_all_prices`).

**Scalar extraction for Scryfall:**
`prices` JSON is flat: `{"eur": "1.50", "eur_foil": null, "usd": "1.95", ...}`. Extract directly: `float(prices["eur"]) if prices.get("eur") else None`.

---

## Bronze Ingestion Changes (`src/data/cards/storage/bronze/`)

### `STORAGE_CONFIG` (`config.py`)

Remove the `SnapshotConfig("bronze_scryfall_prices_history", fields=["prices"])` entry from `scryfall`. Remove `SnapshotConfig("bronze_mtgjson_prices_history")` from `mtgjson_prices`. Both are now handled by dedicated methods, not the generic `_snapshot`.

```python
"scryfall": SourceStorageConfig(
    table="bronze_scryfall_cards",
    key="id",
    snapshots=[
        # bronze_scryfall_prices_history → handled by _snapshot_scryfall_prices
        SnapshotConfig("bronze_scryfall_meta_history", fields=[...]),
    ],
),
"mtgjson_prices": SourceStorageConfig(
    table=None,
    key="uuid",
    snapshots=[],  # handled by _snapshot_mtgjson_prices
),
```

### New methods in `BronzeStorage` (`storage.py`)

**`_snapshot_scryfall_prices(records)`**
Iterates `ScryfallCard` records. For each: extracts `eur`, `eur_foil`, `usd`, `usd_foil` as FLOAT from `record.prices` dict. Builds DataFrame, calls `self._writer.append(df, "bronze_scryfall_prices_history", "id")`.

**`_snapshot_mtgjson_prices(records)`**
Iterates `MtgjsonCardPrices` records. For each: extracts 6 scalar FLOAT columns from `record.paper` using max-date-key ≤ today logic (shared with `seed_historical_prices`). Builds DataFrame, calls `self._writer.append(df, "bronze_mtgjson_prices_history", "uuid")`.

Extraction logic shared via private module-level function `_extract_mtgjson_scalar_prices(paper_dict, target_date) -> dict[str, float | None]`. `_MTGJSON_PRICE_MAP` moves from `silver/prices.py` to `bronze/storage.py`.

### `_snapshot` (generic)

Unchanged. Continues to handle `bronze_scryfall_meta_history` and `bronze_format_staples_history` via `_process_sources`.

### `daily_update`

Explicitly calls both new methods after `_process_sources`:
```python
scryfall_records, _ = results.get("scryfall", ([], []))
self._snapshot_scryfall_prices(scryfall_records)

mtgjson_records, _ = results.get("mtgjson_prices", ([], []))
self._snapshot_mtgjson_prices(mtgjson_records)
```

### `seed_historical_prices`

Rewritten to use `_extract_mtgjson_scalar_prices` instead of `_filter_prices_to_date`. Builds rows with 6 scalar columns. `_filter_prices_to_date` deleted.

---

## Silver Changes (`src/data/cards/storage/silver/`)

### `_build_scryfall_base` (`prices.py`)

Query simplified: `eur`, `eur_foil`, `usd`, `usd_foil` are now direct FLOAT columns, no `json_extract_string`. SQL moved to `silver/sql/scryfall_prices_base.sql`.

### `_join_mtgjson_prices` (`prices.py`)

Python list comprehension + `pd.concat` + `_extract_all_prices` replaced by:
```python
SELECT uuid, snapshot_date, cardmarket_eur, cardmarket_eur_foil,
       cardmarket_buylist_eur, tcgplayer_usd, tcgplayer_usd_foil, tcgplayer_buylist_usd
FROM bronze_mtgjson_prices_history
WHERE snapshot_date = ?
```
Followed by pandas LEFT merge on `(uuid, snapshot_date)`. SQL moved to `silver/sql/mtgjson_prices_daily.sql`.

Fallback when table missing: iterate 6 column names directly instead of `_MTGJSON_PRICE_MAP`.

### Dead code removed

- `_extract_all_prices` static method
- `_MTGJSON_PRICE_MAP` module-level constant
- `import json`

### `build_language_prices` (`prices.py`)

Contains an identical `json_extract_string(prices, '$.eur')` query against `bronze_scryfall_prices_history` (lines 329–344). Must be simplified to direct column access alongside `_build_scryfall_base`. SQL moved to `silver/sql/scryfall_language_prices_base.sql`.

### Unchanged

- `_PRICE_COLS` — still used by `_fill_price_history`
- All forward-fill logic (`_fill_from_history`, `_fill_price_history`, `_fill_language_price_history`)
- `build_language_prices` business logic — only the inner SQL query changes

---

## Health Checks (`src/data/cards/storage/health.py`)

### New check

`_check_bronze_prices_coverage(bronze_con, today)` — verifies that at least one card has a non-NULL `eur` in `bronze_scryfall_prices_history` for today, and at least one card has a non-NULL `cardmarket_eur` in `bronze_mtgjson_prices_history` for today. FAIL if 0 rows with prices — indicates ingestion did not extract any prices.

### Unchanged

All existing checks (`_check_table_has_rows`, `_BRONZE_TABLES`, silver/gold checks) unchanged.

---

## Tests

### Deleted

- `TestFilterPricesToDate` (entire class) — function removed
- `test_paper_filtered_to_snapshot_date_only` — tests JSON column that no longer exists
- `test_dates_collected_from_mtgo_platform` — mtgo column removed

### Updated

- `test_calls_snapshot_for_configured_sources` — `_snapshot` call count changes from 4 to 2; add assertions for `_snapshot_scryfall_prices` and `_snapshot_mtgjson_prices` being called
- `test_daily_update_calls_snapshot` — same update
- `TestSeedHistoricalPrices` — fixture rows checked for scalar columns (`cardmarket_eur` etc.) instead of `paper` JSON; `test_row_contains_uuid_and_snapshot_date` unchanged; `test_one_row_per_unique_date` unchanged

### New

**`TestSnapshotScryfallPrices`** (`test_bronze.py`):
- extracts eur/eur_foil/usd/usd_foil as FLOAT from prices dict
- None/missing keys → NULL column
- tix key ignored
- idempotent on duplicate (id, snapshot_date)

**`TestSnapshotMtgjsonPrices`** (`test_bronze.py`):
- extracts 6 scalar columns from paper dict
- max-date-key ≤ today selection
- paper=None → all columns NULL
- idempotent on duplicate (uuid, snapshot_date)

**Silver `test_silver.py`**:
- `_build_scryfall_base` fixture tables use scalar columns instead of `prices` JSON
- `_join_mtgjson_prices` tests use scalar Bronze fixture; verify DuckDB SELECT + merge
- `build_language_prices` fixture tables use scalar columns instead of `prices` JSON

**`tests/scripts/test_migrate_bronze_prices.py`** (new file):
- migration produces correct scalar values from JSON rows
- old table untouched if INSERT fails midway
- empty source table → no rows in target, no error

**`test_health.py`**:
- `_check_bronze_prices_coverage` PASS when rows with non-NULL prices exist for today
- `_check_bronze_prices_coverage` FAIL when all prices NULL

---

## Documentation Updates

| File | Change |
|------|--------|
| `docs/adr/ADR-003-medallion-architecture.md` | Update Bronze table schemas for both price tables |
| `docs/adr/ADR-012-physical-cards-only.md` | Note that `mtgo` column is removed from Bronze as a consequence |
| `docs/adr/ADR-025-scalar-bronze-prices.md` | New ADR — decision, motivation, consequences |
| `docs/architecture/c4/bronze-storage.md` | Update snapshot mechanism description for price sources |
| `docs/architecture/c4/silver-storage.md` | Step 6: "reads scalar columns" instead of "extracts from JSON" |
| `docs/architecture/data/table-schemas.md` | New schemas for both Bronze price tables |
| `docs/architecture/data/data-lineage.md` | Steps 1, 3 updated; join keys table updated |
| `docs/architecture/data/glossary.md` | "Price snapshot" entry — remove JSON references |
| `docs/architecture/c1/system-context.md` | MTGJson description: "(paper only)" instead of "(paper/MTGO)" |

---

## Scope Boundaries

**In scope:**
- Both Bronze price history tables → scalar columns
- Migration script from `cards_copy.duckdb`
- Bronze ingestion methods
- Silver price builder simplification
- Health checks
- Tests
- Documentation

**Out of scope (not touched):**
- Gold layer — no changes needed
- `bronze_scryfall_meta_history` — JSON structure unrelated to prices
- `bronze_format_staples_history` — unrelated
- Silver forward-fill logic — unaffected
- `build_language_prices` — reads Scryfall prices (scalar after migration), no logic change needed
