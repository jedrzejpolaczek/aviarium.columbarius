-- Selects Scryfall's already-scalar price columns for one snapshot date.
-- No pivot needed — Scryfall Bronze stores prices as scalar columns (unlike
-- MTGJson's EAV format), see ADR-025.
SELECT
    id           AS scryfall_id,
    snapshot_date,
    eur,
    eur_foil,
    usd,
    usd_foil
FROM bronze_scryfall_prices_history
WHERE snapshot_date = ?
