-- Top-8 tournament appearance counts per (card, format): 30d/90d totals,
-- average copies played, sideboard-vs-maindeck split, from
-- silver_tournament_results_history.
WITH base AS (
    SELECT *,
        CAST(tournament_date AS DATE) AS tournament_dt,
        (CURRENT_DATE - CAST(tournament_date AS DATE)) AS days_ago
    FROM silver_tournament_results_history
    WHERE oracle_id IS NOT NULL
)
SELECT
    oracle_id,
    MIN(scryfall_id) AS scryfall_id,
    format,
    COUNT(DISTINCT CASE
        WHEN days_ago <= 30 AND NOT is_sideboard THEN tournament_id
    END) AS top8_appearances_30d,
    COUNT(DISTINCT CASE
        WHEN days_ago <= 90 AND NOT is_sideboard THEN tournament_id
    END) AS top8_appearances_90d,
    AVG(CASE
        WHEN NOT is_sideboard THEN CAST(copies AS FLOAT)
    END) AS top8_copies_avg,
    COUNT(DISTINCT CASE
        WHEN days_ago <= 30 AND is_sideboard THEN tournament_id
    END) AS sideboard_appearances_30d,
    COUNT(DISTINCT CASE
        WHEN days_ago <= 90 AND NOT is_sideboard THEN tournament_id
    END) * 100.0
        / NULLIF(COUNT(DISTINCT CASE
            WHEN days_ago <= 90 THEN tournament_id
        END), 0) AS main_deck_pct,
    MAX(CASE
        WHEN NOT is_sideboard THEN tournament_date
    END) AS last_top8_date
FROM base
GROUP BY oracle_id, format
ORDER BY oracle_id, format
