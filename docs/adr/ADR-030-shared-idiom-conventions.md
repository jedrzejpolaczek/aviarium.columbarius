# ADR-030: Shared Idiom Conventions

## Context

Three successive maintainability audits (2026-07-07 rounds 1 and 2, 2026-07-08
round 3) each found the same *shapes* of duplication reappearing in files the
previous round hadn't touched: a guard clause for "skip this build if an
upstream table is missing" independently written in bronze, silver, and gold;
a "catch this I/O error, re-raise as a domain exception" try/except
independently written in the sources, storage, and pipeline layers; a script
bootstrap (logging setup, `GOLD_DB_PATH` default) independently reinvented
per script. Each round fixed the specific files the audit named. None of them
made the underlying idiom discoverable, so the next file written in that
style reinvented it rather than reusing what already existed.

## Decision

Document the canonical location for each recurring idiom here, so "does a
helper for this already exist?" has one answer instead of requiring a fresh
grep of the whole codebase every time.

| Idiom | Canonical helper | Do not |
|---|---|---|
| "Skip this build if a required upstream table is missing, log what's missing" | `warn_if_missing()` in `src/data/cards/storage/base/storage.py` | Write a new `if missing: logger.warning(...)` block in bronze/silver/gold storage classes |
| "Load a JSON file, raise a clear error if missing/malformed" | `load_json_file()` in `src/data/json_files.py` | Write a new `try: json.loads(Path(...).read_text()) except FileNotFoundError/JSONDecodeError` block anywhere under `src/data/` |
| Gold DuckDB file path default | `GOLD_DB_PATH` in `src/data/repository.py` | Define `os.getenv("GOLD_DB_PATH", ...)` again in a new script or app module |
| Script logging bootstrap | `from src.logger import get_logger, setup_logging`; call `setup_logging(log_dir=Path("logs"))` as the first line of `main()`, and `logger = get_logger(__name__)` at module level if the script logs directly | Use stdlib `logging.basicConfig`/`logging.getLogger` in any file under `scripts/` |
| Cleaning up temporary downloaded HTML files | `_cleanup_html_files()` in `src/data/cards/sources/scrapers.py` | Write a new bare `Path(p).unlink(missing_ok=True)` loop in a new scraper |
| Fast-training LightGBM params for tests | `FAST_LIGHTGBM_PARAMS` in `tests/ml/conftest.py` | Redefine an identical `LightGBMParams(...)` fixture in a new `tests/ml/**` test file |
| Euro-price null-safe formatting (frontend) | `formatEur()` in `frontend/src/format.ts` | Write a new `value !== null ? \`€${value.toFixed(2)}\` : ...` inline in a component |

Two apparent duplications were investigated and deliberately **not** merged —
adding them to this table as "do not fix" so a future audit doesn't re-flag
them as missed work:

- `src/ml/features/sql/lag_features.sql`'s `rolling_mean_7d`/`momentum_7d`
  vs. `src/data/cards/storage/gold/sql/price_features.sql`'s
  `price_7d_avg`/`price_change_7d_pct` — numerically identical, but merging
  would couple the ML feature layer's tests to Gold's precomputed schema for
  a two-column saving. See `docs/investigations/repo-quirks.md`.
- The `(a-b)/NULLIF(b,0)` percent-change formula, which recurs in
  `src/ml/features/sql/lag_features.sql` (`momentum_7d`) and
  `src/data/cards/storage/gold/sql/price_features.sql`
  (`price_change_1d_pct`/`7d_pct`/`30d_pct`, `is_price_spike`), and the
  separate `COALESCE(x::BOOLEAN, false)` pattern in `silver_cards.sql`
  (26 occurrences) — round 2 explicitly left both out of scope: each
  occurrence is a single self-contained line with negligible bug risk, and a
  DuckDB macro would add indirection for no safety gain.

## Consequences

### Positive
- A contributor (human or agent) writing a new storage tier, script, or SQL
  feature file has one table to check before writing a new guard
  clause/try-except/bootstrap from scratch.
- Future audits can check this table first and skip re-investigating
  already-adjudicated "duplication" (the two items explicitly marked
  "do not fix" above).

### Negative
- This table needs updating whenever a new idiom is extracted or a new
  "intentionally not merged" decision is made — it will rot if treated as
  write-once.

### Neutral
- This ADR does not introduce any new abstraction itself; it only indexes
  abstractions introduced by the round-3 remediation plan
  (`docs/superpowers/plans/2026-07-08-maintainability-round3.md`).
