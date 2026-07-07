{legality_lag},
transitions AS (
    SELECT id, snapshot_date, 'commander' AS format,
           CASE WHEN prev_commander = 'legal'  AND curr_commander = 'banned' THEN 'ban'
                WHEN prev_commander = 'banned' AND curr_commander = 'legal'  THEN 'unban'
           END AS event_type
    FROM lagged
    WHERE prev_commander IS NOT NULL
      AND (   (prev_commander = 'legal'  AND curr_commander = 'banned')
           OR (prev_commander = 'banned' AND curr_commander = 'legal'))
    UNION ALL
    SELECT id, snapshot_date, 'standard',
           CASE WHEN prev_standard = 'legal'  AND curr_standard = 'banned' THEN 'ban'
                WHEN prev_standard = 'banned' AND curr_standard = 'legal'  THEN 'unban'
           END
    FROM lagged
    WHERE prev_standard IS NOT NULL
      AND (   (prev_standard = 'legal'  AND curr_standard = 'banned')
           OR (prev_standard = 'banned' AND curr_standard = 'legal'))
    UNION ALL
    SELECT id, snapshot_date, 'modern',
           CASE WHEN prev_modern = 'legal'  AND curr_modern = 'banned' THEN 'ban'
                WHEN prev_modern = 'banned' AND curr_modern = 'legal'  THEN 'unban'
           END
    FROM lagged
    WHERE prev_modern IS NOT NULL
      AND (   (prev_modern = 'legal'  AND curr_modern = 'banned')
           OR (prev_modern = 'banned' AND curr_modern = 'legal'))
    UNION ALL
    SELECT id, snapshot_date, 'legacy',
           CASE WHEN prev_legacy = 'legal'  AND curr_legacy = 'banned' THEN 'ban'
                WHEN prev_legacy = 'banned' AND curr_legacy = 'legal'  THEN 'unban'
           END
    FROM lagged
    WHERE prev_legacy IS NOT NULL
      AND (   (prev_legacy = 'legal'  AND curr_legacy = 'banned')
           OR (prev_legacy = 'banned' AND curr_legacy = 'legal'))
    UNION ALL
    SELECT id, snapshot_date, 'vintage',
           CASE WHEN prev_vintage = 'legal'  AND curr_vintage = 'banned' THEN 'ban'
                WHEN prev_vintage = 'banned' AND curr_vintage = 'legal'  THEN 'unban'
           END
    FROM lagged
    WHERE prev_vintage IS NOT NULL
      AND (   (prev_vintage = 'legal'  AND curr_vintage = 'banned')
           OR (prev_vintage = 'banned' AND curr_vintage = 'legal'))
)
