-- edhrec_rank/prev_rank are consumed only by demand_signals.sql;
-- unused (harmless no-op) for events.sql/ban_price_impact_events.sql.
lagged AS (
    SELECT
        id,
        snapshot_date,
        edhrec_rank,
        LAG(edhrec_rank) OVER w                                          AS prev_rank,
        json_extract_string(legalities, '$.commander') AS curr_commander,
        LAG(json_extract_string(legalities, '$.commander')) OVER w AS prev_commander,
        json_extract_string(legalities, '$.standard')  AS curr_standard,
        LAG(json_extract_string(legalities, '$.standard'))  OVER w AS prev_standard,
        json_extract_string(legalities, '$.modern')    AS curr_modern,
        LAG(json_extract_string(legalities, '$.modern'))    OVER w AS prev_modern,
        json_extract_string(legalities, '$.legacy')    AS curr_legacy,
        LAG(json_extract_string(legalities, '$.legacy'))    OVER w AS prev_legacy,
        json_extract_string(legalities, '$.vintage')   AS curr_vintage,
        LAG(json_extract_string(legalities, '$.vintage'))   OVER w AS prev_vintage
    FROM silver_meta_history
    WINDOW w AS (PARTITION BY id ORDER BY snapshot_date)
)
