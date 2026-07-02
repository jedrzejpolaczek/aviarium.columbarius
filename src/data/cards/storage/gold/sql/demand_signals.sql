WITH lagged AS (
    SELECT
        id,
        snapshot_date,
        edhrec_rank,
        LAG(edhrec_rank) OVER w                                          AS prev_rank,
        json_extract_string(legalities, '$.commander')                   AS commander_legality,
        LAG(json_extract_string(legalities, '$.commander')) OVER w       AS prev_commander,
        json_extract_string(legalities, '$.standard')                    AS standard_legality,
        LAG(json_extract_string(legalities, '$.standard'))  OVER w       AS prev_standard,
        json_extract_string(legalities, '$.modern')                      AS modern_legality,
        LAG(json_extract_string(legalities, '$.modern'))    OVER w       AS prev_modern,
        json_extract_string(legalities, '$.legacy')                      AS legacy_legality,
        LAG(json_extract_string(legalities, '$.legacy'))    OVER w       AS prev_legacy,
        json_extract_string(legalities, '$.vintage')                     AS vintage_legality,
        LAG(json_extract_string(legalities, '$.vintage'))   OVER w       AS prev_vintage
    FROM silver_meta_history
    WINDOW w AS (PARTITION BY id ORDER BY snapshot_date)
)
SELECT
    id,
    snapshot_date,
    edhrec_rank,
    edhrec_rank - prev_rank                                              AS edhrec_rank_change,
    commander_legality,
    standard_legality,
    modern_legality,
    legacy_legality,
    vintage_legality,
    COALESCE(prev_commander = 'legal'  AND commander_legality = 'banned', FALSE) AS commander_banned,
    COALESCE(prev_commander = 'banned' AND commander_legality = 'legal',  FALSE) AS commander_unbanned,
    COALESCE(prev_standard  = 'legal'  AND standard_legality  = 'banned', FALSE) AS standard_banned,
    COALESCE(prev_standard  = 'banned' AND standard_legality  = 'legal',  FALSE) AS standard_unbanned,
    COALESCE(prev_modern    = 'legal'  AND modern_legality    = 'banned', FALSE) AS modern_banned,
    COALESCE(prev_modern    = 'banned' AND modern_legality    = 'legal',  FALSE) AS modern_unbanned,
    COALESCE(prev_legacy    = 'legal'  AND legacy_legality    = 'banned', FALSE) AS legacy_banned,
    COALESCE(prev_legacy    = 'banned' AND legacy_legality    = 'legal',  FALSE) AS legacy_unbanned,
    COALESCE(prev_vintage   = 'legal'  AND vintage_legality   = 'banned', FALSE) AS vintage_banned,
    COALESCE(prev_vintage   = 'banned' AND vintage_legality   = 'legal',  FALSE) AS vintage_unbanned
FROM lagged
ORDER BY id, snapshot_date
