# ADR-025: Scalar Bronze Price Tables with EAV for MTGJson

**Date:** 2026-06-24
**Status:** Accepted

## Context

Bronze price history tables previously stored prices as JSON blobs (`paper`, `prices` columns).
Silver was forced to parse JSON in Python (violating ADR-024) and pre-select specific retailers
at Bronze ingestion time (violating the medallion architecture principle that Bronze = raw,
Silver = semantic selection).

## Decision

1. **MTGJson Bronze → EAV**: `bronze_mtgjson_prices_history` stores one row per price point:
   `(uuid, snapshot_date, retailer, tx_type, finish, price)`. All retailers present in the
   MTGJson feed are captured without pre-selection. Silver pivots to wide columns via CASE WHEN SQL.

2. **Scryfall Bronze → scalar + tix**: `bronze_scryfall_prices_history` stores scalar FLOAT columns
   including `tix`. Silver does not select `tix`; the exclusion decision lives in Silver SQL.

3. **Bronze = structural normalization only. Silver = semantic selection only.**

## Consequences

- New retailers appearing in MTGJson (e.g. `cardkingdom`) are captured automatically in Bronze
  without code changes. Silver's CASE WHEN SQL is the single place that decides what downstream
  consumers see.
- `_MTGJSON_PRICE_MAP` lives exclusively in `SilverPriceBuilder`, exported as the public
  `MTGJSON_PRICE_COMBOS` module constant for consumers like `health.py`. Bronze has no
  concept of which combinations matter.
- Schema drift is detected post-pipeline by `_check_bronze_prices_schema_drift` (health.py),
  which produces WARN (not FAIL) when new or missing combinations are observed.
- One-time migration via `scripts/migrate_bronze_prices.py` from `cards_copy.duckdb`.
