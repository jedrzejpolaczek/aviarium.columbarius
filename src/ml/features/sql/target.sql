WITH t0 AS (
    SELECT uuid, eur AS eur_t0
    FROM gold_price_features
    WHERE snapshot_date = ?
),
t7 AS (
    SELECT uuid, eur AS eur_t7
    FROM gold_price_features
    WHERE snapshot_date = CAST(? AS DATE) + INTERVAL 7 DAY
)
SELECT
    t0.uuid,
    LN(1 + t7.eur_t7) - LN(1 + t0.eur_t0) AS log_return_7d
FROM t0
JOIN t7 ON t0.uuid = t7.uuid
