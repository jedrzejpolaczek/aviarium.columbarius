# Price Feed Anomalies — Investigation Notes

Started: 2026-07-06. Context: notebook re-runs against 32 daily snapshots
(2026-05-26 to 2026-07-06) surfaced two anomalies in `gold_price_features`
that are independent of the frozen-price bug fixed separately.

## Card-count discontinuity

### Background (established in a prior session)

1. `bronze_scryfall_prices_history` has exactly 530,095 distinct `id`s on
   *every* snapshot date from 2026-05-26 to 2026-07-06 — the drop is not a
   bronze refresh/ingestion event.
2. The drop is already present one layer up, in `silver_prices_history`
   (append-only, one row per card per day, never rewritten historically):

   ```python
   import duckdb
   con = duckdb.connect("data/silver/cards.duckdb", read_only=True)
   print(con.execute("""
       SELECT snapshot_date, COUNT(DISTINCT uuid) AS n_cards, COUNT(*) AS n_rows
       FROM silver_prices_history
       GROUP BY snapshot_date ORDER BY snapshot_date
   """).df().to_string())
   ```

   Result: exactly 98,353 cards/day from 2026-05-26 through 2026-06-15, then
   exactly 96,261 cards/day from 2026-06-16 onward (a drop of 2,092 cards) —
   a clean step function, not noise.
3. `silver_prices_history` is written via `append()` (see
   `src/data/cards/storage/silver/storage.py`), so each day's row count
   reflects whatever the Silver price builder produced *on that actual
   calendar day* — historical rows are never recomputed. This means the drop
   reflects a change in eligible cards *at the time each snapshot was
   captured*, not a rebuild artifact.
4. `git log --oneline --since=2026-06-14 --until=2026-06-18` (no path filter)
   returns **zero commits** — ruling out a code deploy in this repo as the
   cause during that exact window.

### Step 1 findings

Query run against real `data/gold/cards.duckdb` (read-only) with
`data/silver/cards.duckdb` attached read-only:

- **Missing card count reproduced exactly: 2,092** — cards present in
  `silver_prices_history` on 2026-06-15 but absent on 2026-06-16.
- **`set_type`/`rarity`/`layout` profile**: no single clean signature across
  all three dimensions (93 distinct combinations by full breakdown), but a
  strong signature emerges on **`layout` alone**:

  | layout           | n   |
  |------------------|-----|
  | transform        | 517 |
  | adventure        | 180 |
  | modal_dfc        | 116 |
  | split            | 106 |
  | prepare          |  46 |
  | reversible_card  |  42 |
  | aftermath        |  29 |
  | flip             |  17 |

  This breakdown covers all 1,053 of the "still present" subset (see below)
  and sums exactly to 1,053. **Critically, zero of these cards have
  `layout = 'normal'`** — every single one is a multi-faced card layout
  (transform, adventure, modal_dfc, split, aftermath, flip,
  reversible_card, prepare). `rarity` and `set_type` breakdowns show no
  comparable concentration (spread across rare/uncommon/mythic/common and
  across expansion/promo/masters/draft_innovation/box/commander/etc.),
  consistent with these being ordinary multi-face cards from many sets
  rather than a special promo/memorabilia category.

- **`still_present` count: 1,053 of 2,092** (roughly half) — these uuids
  still exist in the *current* `silver_cards` table today. The remaining
  **1,039 are truly gone** from current `silver_cards` (no row at all).
  This is neither of the two clean outcomes anticipated going in
  (`still_present ≈ 0` or `still_present ≈ len(missing)`); it's a genuine
  50/50 split, which itself turned out to be the key clue.

### Root-cause mechanism identified

`src/data/cards/storage/silver/sql/silver_cards.sql` contains a `deduped`
CTE (lines 293–308) that explicitly collapses multi-face MTGJson rows down
to **one row per `scryfall_id`**:

```sql
deduped AS (
    SELECT * EXCLUDE rn
    FROM (
        SELECT *,
               ROW_NUMBER() OVER (
                   PARTITION BY scryfall_id
                   ORDER BY
                       CASE WHEN uuid IS NOT NULL THEN 0 ELSE 1 END,
                       uuid
               ) AS rn
        FROM with_canonical
        WHERE scryfall_id IS NOT NULL
    )
    WHERE rn = 1
)
```

MTGJson issues **one `uuid` per face** for transform/adventure/split/etc.
cards, but all faces of a physical card share the same Scryfall
`scryfall_id`. This CTE keeps only the front face (lowest `uuid`) per
`scryfall_id` — this is why exactly the multi-face layouts show up in the
"missing" set: they're the category where MTGJson-to-Scryfall is
one-to-many, so a dedup step exists at all.

`src/data/cards/storage/silver/prices.py::_build_scryfall_base` then builds
each day's price row set by joining `bronze_scryfall_prices_history`
(keyed by `scryfall_id`) against a `card_map` derived from **that day's**
`silver_cards` (`COALESCE(uuid, canonical_uuid)`, one row per
`scryfall_id`). Because `silver_cards` is rebuilt fresh every run
(`CREATE OR REPLACE TABLE`) from current Bronze, and `silver_prices_history`
permanently records whichever uuid won that day's front-face tiebreak, a
shift in Bronze MTGJson's per-face `uuid` values or in the `ROW_NUMBER`
ordering (e.g., a new face being assigned a lower `uuid`, or a face being
added/removed) is sufficient to produce a permanent one-time step change in
the append-only history — with zero effect on bronze row counts (which are
per-`scryfall_id`/`id`, not per-MTGJson-`uuid`), consistent with finding #1
above.

Confirmed today's `silver_cards` still dedupes cleanly (0 `scryfall_id`s
with more than 1 row; 515,728 total rows), so this dedup behavior is
long-standing pipeline behavior, not a new regression. `git log` shows no
commits to `silver_cards.sql` near 2026-06-16 (only unrelated
funny/memorabilia-exclusion and refactor commits, all either well before or
well after the discontinuity window), which is consistent with the earlier
"zero commits 06-14..06-18" finding — the dedup *logic* itself did not
change.

`bronze_mtgjson_cards` and `bronze_scryfall_cards` are current-state only
(not historized like the price tables), so it is not possible to directly
diff Bronze's card attributes as they stood on 2026-06-15 vs. 2026-06-16 —
this limits full certainty, but all available evidence (stable bronze price
row counts, no relevant code changes, and a clean 100%-multi-face-layout
signature) points to the same place.

### Conclusion

**Not a pipeline bug — an upstream MTGJson/Scryfall data change affecting
multi-faced card identifier mapping**, most likely a bulk correction to
MTGJson's per-face `uuid` assignments (or their linkage to Scryfall
`scryfall_id`s) for transform/adventure/modal_dfc/split/aftermath/flip/
reversible_card/prepare cards around 2026-06-16. The existing
`silver_cards.sql` dedup step (`PARTITION BY scryfall_id ... rn = 1`) is
working as designed — it deterministically keeps one row per physical card
— but it is inherently sensitive to which `uuid` MTGJson assigns as the
"first" face on any given day. When MTGJson's upstream data shifts (e.g. a
face's uuid changes, or which face's data is considered canonical shifts),
the tiebreak winner can change, silently altering which uuid gets priced
from that day forward.

This does not require a code fix: the current behavior (one row per
physical card, keyed by the front face) is intentional. It is recorded here
as an informational/monitoring item — if MTGJson corrections like this
recur, the price history for the affected cards will show similar
step-discontinuities in future card counts (not in individual card prices,
since `_fill_price_history` forward-fills for cards with no data, but the
*population* of tracked uuids can shift). No further action is planned
against this repo's code as a result of this investigation.
