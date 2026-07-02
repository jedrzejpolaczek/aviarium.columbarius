SELECT 1 FROM (
    SELECT id
    FROM silver_meta_history
    WHERE legalities IS NOT NULL
    GROUP BY id
    HAVING COUNT(DISTINCT legalities) > 1
) t
LIMIT 1
