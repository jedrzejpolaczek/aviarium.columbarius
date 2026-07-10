-- Standard 7-day/30-day trailing window pair (bounded, no future leakage).
-- {{partition_col}}/{{order_col}} are substituted by GoldSignalBuilders (see
-- signals.py) and GoldFeatureBuilders (see features.py), which each
-- interpolate this fragment into their own WINDOW clause — format_staples.sql
-- and price_features.sql respectively.
w7  AS (PARTITION BY {partition_col} ORDER BY {order_col}
         ROWS BETWEEN 6 PRECEDING AND CURRENT ROW),
w30 AS (PARTITION BY {partition_col} ORDER BY {order_col}
         ROWS BETWEEN 29 PRECEDING AND CURRENT ROW)
