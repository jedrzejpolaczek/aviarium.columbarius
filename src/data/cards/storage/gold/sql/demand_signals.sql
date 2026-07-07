-- {{legality_lag}} is substituted by build_demand_signals() with
-- sql/_legality_lag_cte.sql's "lagged AS (...)" CTE body (see signals.py).
WITH {legality_lag}
SELECT
    id,
    snapshot_date,
    edhrec_rank,
    edhrec_rank - prev_rank                                              AS edhrec_rank_change,
    curr_commander                                                       AS commander_legality,
    curr_standard                                                        AS standard_legality,
    curr_modern                                                          AS modern_legality,
    curr_legacy                                                          AS legacy_legality,
    curr_vintage                                                         AS vintage_legality,
    COALESCE(prev_commander = 'legal'  AND curr_commander = 'banned', FALSE) AS commander_banned,
    COALESCE(prev_commander = 'banned' AND curr_commander = 'legal',  FALSE) AS commander_unbanned,
    COALESCE(prev_standard  = 'legal'  AND curr_standard  = 'banned', FALSE) AS standard_banned,
    COALESCE(prev_standard  = 'banned' AND curr_standard  = 'legal',  FALSE) AS standard_unbanned,
    COALESCE(prev_modern    = 'legal'  AND curr_modern    = 'banned', FALSE) AS modern_banned,
    COALESCE(prev_modern    = 'banned' AND curr_modern    = 'legal',  FALSE) AS modern_unbanned,
    COALESCE(prev_legacy    = 'legal'  AND curr_legacy    = 'banned', FALSE) AS legacy_banned,
    COALESCE(prev_legacy    = 'banned' AND curr_legacy    = 'legal',  FALSE) AS legacy_unbanned,
    COALESCE(prev_vintage   = 'legal'  AND curr_vintage   = 'banned', FALSE) AS vintage_banned,
    COALESCE(prev_vintage   = 'banned' AND curr_vintage   = 'legal',  FALSE) AS vintage_unbanned
FROM lagged
ORDER BY id, snapshot_date
