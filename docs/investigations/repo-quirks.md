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
