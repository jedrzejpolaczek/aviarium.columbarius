WITH {transitions_cte}
SELECT id AS scryfall_id, snapshot_date AS event_date, format, event_type
FROM transitions
