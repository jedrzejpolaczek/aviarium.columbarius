WITH {transitions_cte}
SELECT snapshot_date AS event_date, format, event_type, COUNT(*) AS card_count
FROM transitions
GROUP BY snapshot_date, format, event_type
ORDER BY event_date, format, event_type
