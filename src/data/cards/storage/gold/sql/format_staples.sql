-- Rolling EDHREC format-staple percentages: 7d/30d trailing averages and
-- 7d/30d deltas per (card, format), from silver_format_staples_history.
SELECT
    id,
    card_name,
    format,
    snapshot_date,
    deck_pct,
    played,
    top,
    AVG(deck_pct) OVER w7  AS deck_pct_7d_avg,
    AVG(deck_pct) OVER w30 AS deck_pct_30d_avg,
    deck_pct - LAG(deck_pct, 7)  OVER (PARTITION BY id ORDER BY snapshot_date)
        AS deck_pct_change_7d,
    deck_pct - LAG(deck_pct, 30) OVER (PARTITION BY id ORDER BY snapshot_date)
        AS deck_pct_change_30d
FROM silver_format_staples_history
WINDOW
    {rolling_7_30}
ORDER BY id, snapshot_date
