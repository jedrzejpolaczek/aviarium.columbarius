-- Cleans one snapshot's worth of Scryfall metadata (legalities, edhrec_rank,
-- reserved flag, promo_types, finishes) for append to silver_meta_history.
-- {{join_clause}} is a Python-side placeholder for an optional join (check
-- the calling code in silver/storage.py for what it's conditioned on).
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
