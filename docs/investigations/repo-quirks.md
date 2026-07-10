# Repo Quirks — Curiosities Log

Running, undated log of small oddities found during audits/investigations that
turned out to be harmless. Recorded so nobody re-investigates them as a "bug"
a second time. Each entry is one sentence: what it looks like, and what it
actually is.

## Commit `0babd82` has a copy-pasted, misleading message

Commit `0babd82` (2026-06-21) carries the exact same subject line as the
preceding commit `5a81400` ("fix: COALESCE format_count terms to prevent NULL
propagation from sparse legalities JSON"), but its actual diff is unrelated —
it only removes an unused `pandas` import and applies `ruff format`; the real
(and only) fix for the NULL-propagation bug is `5a81400`, which is complete
and covered by `test_format_count_is_non_null_and_counts_legal_formats`.

## `lag_features.sql`'s `rolling_mean_7d`/`momentum_7d` look like they duplicate `price_features.sql`

They do, for those two columns specifically (`rolling_mean_7d` ≡ `price_7d_avg`,
`momentum_7d` ≡ `price_change_7d_pct` — same partition/order/formula), but it's
intentional: `lag_features.sql` is deliberately self-contained so
`tests/ml/features/test_lag.py` can exercise it against a minimal 5-column
synthetic `gold_price_features` fixture without needing to also fabricate
Gold's precomputed rolling-window columns. The *other* columns compared in the
same investigation (`rolling_std_14d`, `rolling_min_30d`, `rolling_max_30d`)
are not duplicates at all — different window widths and aggregate functions
than anything in `price_features.sql`. Considered and rejected merging the two
real duplicate columns during the round-3 maintainability remediation
(2026-07-08) — see that plan's Task 9 for the full trade-off.

## "7d"/"30d" lag and rolling columns can silently span more or fewer calendar days near a snapshot gap

`price_features.sql`, `lag_features.sql`, and `format_staples.sql` (via the
shared `_rolling_7_30_window.sql` fragment) all compute `lag_7d`/`lag_30d`/
`rolling_mean_7d`/`price_change_7d_pct`/etc. with row-offset window functions
(`LAG(eur, 7) OVER (... ORDER BY snapshot_date)`, `ROWS BETWEEN 6 PRECEDING`),
not date-range ones — already called out as intentional in `features.py`'s
`build_price_features` docstring ("row-based, not date-range-based; gaps in
snapshots shift the effective window"). Confirmed on the real snapshot history
(2026-05-26 → 2026-07-09, 36 dates with 4 real gaps: missing 06-10,
06-24→06-28, 06-30, 07-03→07-04) that this isn't just theoretical: the row
computed as `lag_7d` for snapshot 2026-06-29 actually comes from 2026-06-17 —
12 calendar days back, not 7 — and 2026-07-05's `lag_7d` comes from
2026-06-20, 15 days back. `lag_30d` for 2026-07-09 reaches back to 2026-05-31,
39 days, not 30. The ML **target** (`target.sql`'s `log_return_7d`) is exempt —
it does a proper exact-date join — so training *labels* aren't affected, only
the pre-computed *feature* and *signal* columns that feed `lag_features.sql`
(model input features) and `signals.py`/`underpriced.py` (recommendation
signals). Decided (2026-07-10) to keep the row-based behavior for now: fixing
it means rewriting 3 SQL files to exact-date self-joins / `RANGE BETWEEN
INTERVAL ... PRECEDING` frames, updating whatever tests assert the current
values, rebuilding the whole Gold layer, and re-running every notebook that
reads these columns — a cost that shrinks in relative importance as more
daily snapshots accumulate and gaps become a smaller fraction of each card's
history.
