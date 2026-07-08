-- Fast pre-check: does silver_meta_history contain any legality change at
-- all (more than one distinct legalities value per card)? Used as a cheap
-- skip gate before running the more expensive LAG()-based transition query
-- in transitions_cte.sql.
SELECT 1 FROM (
    SELECT id
    FROM silver_meta_history
    WHERE legalities IS NOT NULL
    GROUP BY id
    HAVING COUNT(DISTINCT legalities) > 1
) t
LIMIT 1
