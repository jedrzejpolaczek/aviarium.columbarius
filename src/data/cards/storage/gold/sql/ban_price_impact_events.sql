-- {{transitions_cte}} is substituted by build_ban_price_impact() with
-- GoldSignalBuilders._TRANSITIONS_CTE (see signals.py).
WITH {transitions_cte}
SELECT id AS scryfall_id, snapshot_date AS event_date, format, event_type
FROM transitions
