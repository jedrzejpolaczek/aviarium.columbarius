SELECT
    id           AS scryfall_id,
    snapshot_date,
    eur,
    eur_foil,
    usd,
    usd_foil
FROM bronze_scryfall_prices_history
WHERE snapshot_date = ?
