-- Training frame for the price model: joins gold_price_features (spine)
-- with its own t+7/t+30 future prices to compute regression targets
-- (target_price_7d, target_price_30d) plus a 3-class label
-- (target_change_30d) for exploratory classification. {{cols}}/{{joins}} are
-- Python-side placeholders (see GoldMLDatasetBuilder.build_ml_dataset) for
-- optional joins to gold_card_features/gold_demand_signals/etc. when those
-- tables exist.
WITH spine AS (
    SELECT * FROM gold_price_features
),
with_targets AS (
    SELECT
        s.*,
        t7.eur  AS target_price_7d,
        t30.eur AS target_price_30d
    FROM spine s
    LEFT JOIN gold_price_features t7
        ON s.uuid = t7.uuid
        AND CAST(t7.snapshot_date AS DATE)
            = CAST(s.snapshot_date AS DATE) + INTERVAL '7 days'
    LEFT JOIN gold_price_features t30
        ON s.uuid = t30.uuid
        AND CAST(t30.snapshot_date AS DATE)
            = CAST(s.snapshot_date AS DATE) + INTERVAL '30 days'
),
with_change_label AS (
    SELECT *,
        -- +/-20% over 30 days is the up/down threshold for the exploratory
        -- 3-class label; chosen as a round, interpretable cutoff, not derived
        -- from a statistical test — target_price_30d/target_price_7d (the
        -- regression targets) are the primary model outputs.
        CASE
            WHEN target_price_30d IS NULL        THEN NULL
            WHEN target_price_30d > eur * 1.2   THEN 'up'
            WHEN target_price_30d < eur * 0.8   THEN 'down'
            ELSE 'flat'
        END AS target_change_30d
    FROM with_targets
    WHERE eur IS NOT NULL
)
SELECT
    wc.*,
    {cols}
FROM with_change_label wc
{joins}
ORDER BY wc.uuid, wc.snapshot_date
