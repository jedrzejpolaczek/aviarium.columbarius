SELECT
    uuid,
    snapshot_date,
    MAX(CASE WHEN retailer = 'cardmarket'  AND tx_type = 'retail'  AND finish = 'normal' THEN price END) AS cardmarket_eur,
    MAX(CASE WHEN retailer = 'cardmarket'  AND tx_type = 'retail'  AND finish = 'foil'   THEN price END) AS cardmarket_eur_foil,
    MAX(CASE WHEN retailer = 'cardmarket'  AND tx_type = 'buylist' AND finish = 'normal' THEN price END) AS cardmarket_buylist_eur,
    MAX(CASE WHEN retailer = 'tcgplayer'   AND tx_type = 'retail'  AND finish = 'normal' THEN price END) AS tcgplayer_usd,
    MAX(CASE WHEN retailer = 'tcgplayer'   AND tx_type = 'retail'  AND finish = 'foil'   THEN price END) AS tcgplayer_usd_foil,
    MAX(CASE WHEN retailer = 'tcgplayer'   AND tx_type = 'buylist' AND finish = 'normal' THEN price END) AS tcgplayer_buylist_usd,
    MAX(CASE WHEN retailer = 'cardkingdom' AND tx_type = 'retail'  AND finish = 'normal' THEN price END) AS cardkingdom_usd,
    MAX(CASE WHEN retailer = 'cardkingdom' AND tx_type = 'retail'  AND finish = 'foil'   THEN price END) AS cardkingdom_usd_foil,
    MAX(CASE WHEN retailer = 'cardkingdom' AND tx_type = 'buylist' AND finish = 'normal' THEN price END) AS cardkingdom_buylist_usd,
    MAX(CASE WHEN retailer = 'cardkingdom' AND tx_type = 'buylist' AND finish = 'foil'   THEN price END) AS cardkingdom_buylist_usd_foil,
    MAX(CASE WHEN retailer = 'manapool'    AND tx_type = 'retail'  AND finish = 'normal' THEN price END) AS manapool_usd,
    MAX(CASE WHEN retailer = 'manapool'    AND tx_type = 'retail'  AND finish = 'foil'   THEN price END) AS manapool_usd_foil
FROM bronze_mtgjson_prices_history
WHERE snapshot_date = ?
GROUP BY uuid, snapshot_date
