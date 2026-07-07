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
