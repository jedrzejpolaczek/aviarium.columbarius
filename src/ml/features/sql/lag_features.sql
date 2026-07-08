-- Time-based lag/rolling features for one snapshot date: lag_1d/7d/14d/30d,
-- rolling_mean_7d, rolling_std_14d, rolling_min_30d/max_30d, and momentum_7d,
-- all computed from gold_price_features with backward-only window functions
-- (no future leakage). See src/ml/features/lag.py's module docstring for
-- the full leakage-safety rationale.
WITH lagged AS (
    SELECT
        uuid,
        snapshot_date,
        eur,
        edhrec_rank,
        foil_premium,
        LAG(eur,  1) OVER w AS lag_1d,
        LAG(eur,  7) OVER w AS lag_7d,
        LAG(eur, 14) OVER w AS lag_14d,
        LAG(eur, 30) OVER w AS lag_30d,
        AVG(eur) OVER (
            PARTITION BY uuid ORDER BY snapshot_date
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ) AS rolling_mean_7d,
        STDDEV(eur) OVER (
            PARTITION BY uuid ORDER BY snapshot_date
            ROWS BETWEEN 13 PRECEDING AND CURRENT ROW
        ) AS rolling_std_14d,
        MIN(eur) OVER (
            PARTITION BY uuid ORDER BY snapshot_date
            ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
        ) AS rolling_min_30d,
        MAX(eur) OVER (
            PARTITION BY uuid ORDER BY snapshot_date
            ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
        ) AS rolling_max_30d
    FROM gold_price_features
    WINDOW w AS (PARTITION BY uuid ORDER BY snapshot_date)
)
SELECT
    *,
    (eur - lag_7d) / NULLIF(lag_7d, 0) AS momentum_7d
FROM lagged
WHERE snapshot_date = ?
