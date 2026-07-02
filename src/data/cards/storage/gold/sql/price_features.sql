WITH price_lags AS (
    SELECT
        *,
        LAG(eur,  1) OVER (PARTITION BY uuid ORDER BY snapshot_date) AS lag_1d,
        LAG(eur,  7) OVER (PARTITION BY uuid ORDER BY snapshot_date) AS lag_7d,
        LAG(eur, 30) OVER (PARTITION BY uuid ORDER BY snapshot_date) AS lag_30d
    FROM silver_prices_history
    WHERE uuid IS NOT NULL
)
SELECT
    p.uuid,
    p.scryfall_id,
    p.snapshot_date,
    p.eur,
    p.eur_foil,
    p.usd,
    p.usd_foil,
    p.cardmarket_eur,
    p.cardmarket_eur_foil,
    -- cardmarket_buylist_eur and tcgplayer_buylist_usd omitted:
    -- 100 % NULL in current data (buylist source not yet ingested).
    p.tcgplayer_usd,
    p.tcgplayer_usd_foil,

    {edhrec_col}

    AVG(p.eur) OVER w7  AS price_7d_avg,
    AVG(p.eur) OVER w30 AS price_30d_avg,

    p.eur - p.lag_1d  AS price_change_1d_abs,
    p.eur - p.lag_7d  AS price_change_7d_abs,
    p.eur - p.lag_30d AS price_change_30d_abs,

    (p.eur - p.lag_1d)  / NULLIF(p.lag_1d,  0) AS price_change_1d_pct,
    (p.eur - p.lag_7d)  / NULLIF(p.lag_7d,  0) AS price_change_7d_pct,
    (p.eur - p.lag_30d) / NULLIF(p.lag_30d, 0) AS price_change_30d_pct,

    STDDEV(p.eur) OVER w30 AS price_volatility_30d,

    p.eur_foil / NULLIF(p.eur, 0) AS foil_premium,

    -- Bounded historical windows: ROWS BETWEEN UNBOUNDED PRECEDING AND
    -- CURRENT ROW ensures only past + current data is used, eliminating
    -- the future-leakage that an unordered PARTITION BY uuid produced.
    MAX(p.eur) OVER w_hist   AS price_ath,
    MIN(p.eur) OVER w_hist   AS price_atl,
    COUNT(p.eur) OVER w_hist AS days_with_price,
    DATEDIFF('day',
        MAX(CASE WHEN p.eur IS NOT NULL THEN p.snapshot_date::DATE END)
            OVER w_hist,
        p.snapshot_date::DATE
    ) AS days_since_last_real_price,

    RANK() OVER (
        PARTITION BY p.snapshot_date ORDER BY p.eur DESC NULLS LAST
    ) AS price_rank_global,

    ABS((p.eur - p.lag_1d) / NULLIF(p.lag_1d, 0)) > 3.0 AS is_price_spike

FROM price_lags p
{meta_join}
WINDOW
    w7     AS (PARTITION BY p.uuid ORDER BY p.snapshot_date
                ROWS BETWEEN 6 PRECEDING AND CURRENT ROW),
    w30    AS (PARTITION BY p.uuid ORDER BY p.snapshot_date
                ROWS BETWEEN 29 PRECEDING AND CURRENT ROW),
    w_hist AS (PARTITION BY p.uuid ORDER BY p.snapshot_date
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
ORDER BY p.uuid, p.snapshot_date
