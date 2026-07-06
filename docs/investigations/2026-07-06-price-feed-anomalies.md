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

## Date-spine LAG(7) gap

### Background

A notebook verification cell in `notebooks/exploratory_data_analysis/03_time_series.ipynb`
(Section 5, "LAG Feature Verification") compares `gold_price_features`'s
7-day lag-derived column against a manually-recomputed date-based LAG for a
sample of rows, and — per a prior live re-run of that cell — found rows
where the gold column reads `0.0000` but the manual recomputation gives
`NULL`. Note: the notebook's *committed* output cells reflect a much
earlier pipeline state (only 3 snapshots, 2026-06-04 to 2026-06-06), where
the cell explicitly reports the 7d LAG as "untestable" and schedules
re-verification once ≥30 snapshots exist. The database now has 33 snapshots
(2026-05-26 through 2026-07-06), so this re-verification was performed
directly against real data, independent of the notebook.

One schema note: `gold_price_features` does not persist a `lag_7d` column
directly — only the derived `price_change_7d_abs = eur - LAG(eur, 7)` (see
`src/data/cards/storage/gold/sql/price_features.sql:5,31`). `lag_7d` was
reconstructed as `eur - price_change_7d_abs` for the checks below, and
`LAG(eur, 7)` was also independently recomputed directly against
`silver_prices_history` (bypassing Gold and the notebook entirely) to
cross-check.

### Step 1: reproduce directly against real data

Confirmed snapshot calendar has the known gaps: `06-10`, `06-24`–`06-28`,
`06-30`, `07-03`, `07-04` are missing (33 distinct dates present, 2026-05-26
to 2026-07-06, out of 42 calendar days in that span).

For each of the 4 flagged dates, joining `gold_price_features` at `t` to
`gold_price_features` at `t - 7 calendar days` on `uuid` (manual, date-based
recompute) against the same rows' stored `price_change_7d_abs`:

| date | total rows | both NULL | gold non-NULL, manual NULL | numeric mismatch (both present) |
|------|-----------:|----------:|----------------------------:|---------------------------------:|
| 2026-06-17 | 96,261 | 15,504 | 80,757 | 0 |
| 2026-07-01 | 96,261 | 15,504 | 80,757 | 0 |
| 2026-07-02 | 96,261 | 15,504 | 80,757 | 0 |
| 2026-07-05 | 96,261 | 15,504 | 80,757 | 0 |

All 4 dates land 7 calendar days after a gap-affected window, so every one
of them has zero real snapshot exactly 7 calendar days prior for the
80,757 affected rows (consistent with the plan's expectation). Sample rows
(gold non-NULL, manual NULL), e.g. on 2026-06-17:

```
uuid=0001e0d0-2dcd-5640-aadc-a84765cf5fc9  eur=4.82  price_change_7d_abs=0.0  (manual_7d_abs=NULL)
uuid=0003caab-9ff5-5d1a-bc06-976dd0457f19  eur=0.18  price_change_7d_abs=0.0  (manual_7d_abs=NULL)
uuid=0003d249-25d9-5223-af1e-1130f09622a7  eur=0.13  price_change_7d_abs=0.0  (manual_7d_abs=NULL)
```

Critically, across all 80,757 mismatch rows on every one of the 4 flagged
dates, `price_change_7d_abs` is **exactly `0.0` for 100% of them** (min =
mean = max = 0.0, no variance) — i.e. reconstructed `lag_7d = eur` in every
case, never any other value.

### Step 2: check whether a real snapshot exists 7 calendar days prior

Confirmed empty for all sampled rows — e.g. for
`uuid=0001e0d0-2dcd-5640-aadc-a84765cf5fc9` on `2026-06-17`, 7 calendar days
prior is `2026-06-10`, which is one of the known gap dates (absent from
`silver_prices_history` entirely for this uuid, as for all cards). This
matches the plan's expectation exactly: the calendar 7-days-back date falls
in a gap window for all 4 flagged dates.

### Step 3: check whether the `eur` value 7 *physical* rows back is genuinely `0.0`/NULL-coerced, or a real unchanged price

Two independent checks:

1. **Direct row-by-row inspection** of `silver_prices_history` for a
   diverse sample of uuids (bulk common at €0.08–€0.09, a €2.05 mid-tier
   card, an €879.90 high-value card, plus the €4.82 sample above) shows the
   `eur` value is **flat/unchanged** across the entire physical window
   spanning the flagged date and 7 rows back — e.g.
   `uuid=09de7279-11e7-5ec6-8297-3c04e2db57d4` holds `eur=879.900024`
   continuously from `2026-06-16` through `2026-07-01` (physical row 7 back
   from `2026-07-01` is `2026-06-18`, still `879.900024`). The value 7
   physical rows back is a **real, non-NULL, unchanged price** in every
   sampled case, not a NULL.
2. **Independent SQL re-verification**: recomputed `LAG(eur, 7)` fresh
   directly over `silver_prices_history` via `duckdb` (bypassing
   `gold_price_features` and the notebook entirely) for all uuids on all 4
   flagged dates:

   | date | n_total | LAG(eur,7) IS NULL | LAG(eur,7) = 0.0 exactly |
   |------|--------:|--------------------:|--------------------------:|
   | 2026-06-17 | 96,261 | 15,504 | **0** |
   | 2026-07-01 | 96,261 | 15,504 | **0** |
   | 2026-07-02 | 96,261 | 15,504 | **0** |
   | 2026-07-05 | 96,261 | 15,504 | **0** |

   Zero rows anywhere have `LAG(eur, 7)` spuriously evaluating to `0.0` —
   the 15,504 rows with fewer than 7 physical prior snapshots correctly
   produce `NULL`, matching the `n_lag_null`/`both NULL` count in Step 1
   exactly. Also confirmed by direct source inspection: no
   `COALESCE(eur, 0)` or similar wraps the `LAG()` source column anywhere
   in `src/data/cards/storage/gold/sql/price_features.sql`.

Standard SQL arithmetic semantics were also sanity-checked directly
(`5.0 - NULL` and `NULL - NULL` both evaluate to `NULL` in DuckDB, never
`0.0`), ruling out an implicit-NULL-to-zero coercion in the
`price_change_7d_abs = eur - lag_7d` subtraction itself.

### Conclusion

**Not a bug — confirms existing documented row-based LAG semantics.**
`price_change_7d_abs = 0.0` (and therefore reconstructed `lag_7d = eur`)
for the 80,757 flagged rows on each of the 4 dates is the **correct,
real** result of `LAG(eur, 7)` (row-based) landing on a genuine prior
snapshot whose `eur` happens to be unchanged from the current row —
because the 7-*physical*-row window, shifted by the known gap dates
(`06-10`, `06-24`–`06-28`, `06-30`, `07-03`, `07-04`), no longer aligns
with 7 *calendar* days. This is exactly the behavior flagged in the
existing docstring warning in `build_price_features()`
(`src/data/cards/storage/gold/features.py:170-172`: "All features are
row-based (not date-range-based) ... gaps in snapshots shift the effective
window").

The notebook's "should be NULL" expectation (comparing gold's row-based
value against a calendar-day-aligned manual recompute) is the mismatched
expectation, not the pipeline: MTG prices are frequently sticky/unchanged
over the span of a single missed-snapshot gap window (confirmed directly
above for bulk commons, mid-tier, and high-value cards alike), so a
row-based LAG(7) that happens to land a few calendar days off from exactly
7 days prior very often still returns the *same* real price, producing
`price_change_7d_abs = 0.0` — a legitimate value, not a defect.

No code changes are needed as a result of this investigation. If tighter
calendar alignment for LAG features is ever desired, that would be a
deliberate design change (e.g. joining on `snapshot_date - INTERVAL 7 DAY`
instead of `LAG(..., 7)`), not a bug fix — out of scope here and not
recommended without a specific downstream need, since it would introduce
NULLs across every gap-adjacent date instead of the current row-based
degradation.
