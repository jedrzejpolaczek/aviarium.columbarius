SELECT
    lp.scryfall_id,
    lp.canonical_uuid,
    lp.lang,
    lp.snapshot_date,
    lp.eur                                  AS lang_eur,
    lp.eur_foil                             AS lang_eur_foil,
    lp.usd                                  AS lang_usd,
    lp.usd_foil                             AS lang_usd_foil,
    ep.eur                                  AS canonical_eur,
    ep.eur_foil                             AS canonical_eur_foil,
    lp.eur      / NULLIF(ep.eur,      0)   AS eur_lang_premium,
    lp.eur_foil / NULLIF(ep.eur_foil, 0)   AS eur_foil_lang_premium
FROM silver_language_prices_history lp
JOIN silver_prices_history ep
    ON  lp.canonical_uuid = ep.uuid
    AND lp.snapshot_date  = ep.snapshot_date
ORDER BY lp.canonical_uuid, lp.lang, lp.snapshot_date
