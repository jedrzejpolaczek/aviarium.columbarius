# ADR-029: `DuckDBRepository` for Connection Creation Outside the Storage Tier

**Date:** 2026-07-08
**Status:** Accepted

## Context

`BaseStorage._open_connection` (storage tier, `src/data/cards/storage/base/storage.py`)
already centralized `duckdb.connect(...)` with error wrapping (`StorageConnectionError`)
and logging for storage-tier subclasses. But four locations outside that tier
independently called `duckdb.connect(...)` directly — six individual calls in total,
since `health.py` alone made three — with no shared error handling:

- `app/main.py` — opens the Gold DB connection stored in `app.state` at FastAPI startup
  (one connection).
- `src/data/cards/storage/health.py` — three separate connections (Bronze, Silver, Gold)
  for the standalone health-check pipeline.
- `scripts/check_and_retrain.py` — opens the Gold DB connection to decide whether
  retraining is needed (one connection).
- `scripts/train_model.py` — opens the Gold DB connection to run manual training
  (one connection).

None of these are `BaseStorage` subclasses (they don't own a storage lifecycle — they're
one-shot scripts or app-level entry points), so none could reach `_open_connection`
without subclassing `BaseStorage` for a single connection call, which would have been the
wrong tool: `BaseStorage` carries table-management responsibilities these callers don't
need.

This is not a re-litigation of ADR-024. ADR-024 decided *where computation happens*
(DuckDB, in-process, over `duckdb.DuckDBPyConnection`) for the Gold signal builders and
similar large-history queries. This ADR addresses a different problem: *how the
connection itself gets created* outside the storage tier. The raw
`duckdb.DuckDBPyConnection` type and the SQL that runs over it are unchanged by this ADR.

## Decision

Two new pieces, plus four migrated locations (six individual `duckdb.connect(...)`
calls):

1. **`src/data/db.py::open_connection(db_path, read_only)`** — extracted from
   `BaseStorage._open_connection`; `BaseStorage` now delegates to it. One implementation
   of `duckdb.connect(...)` + `StorageConnectionError` wrapping + logging, shared by both
   the storage tier and everything outside it.

2. **`src/data/repository.py::DuckDBRepository` / `open_repository(db_path, read_only)`**
   — a thin wrapper around one `duckdb.DuckDBPyConnection`, built on `open_connection`,
   for callers that need a named, injectable type instead of importing
   `duckdb.DuckDBPyConnection` directly. It exposes:
   - `.connection` — the raw `duckdb.DuckDBPyConnection`, public, for callers that need
     to run arbitrary SQL through it unchanged.
   - `.get_tables()` / `.query_df(sql, params)` — convenience methods, added alongside
     connection management for symmetry with the existing free `get_tables()` function
     in `src/data/cards/storage/base/storage.py`. **These do not correspond to an
     observed duplication problem.** An earlier draft of this ADR (and the class's
     original docstring) claimed callers "already reimplement ad hoc" this logic; a
     code-review pass on the introducing commit found that claim false — all three
     locations already migrated at that point (`app/main.py`, `health.py`,
     `check_and_retrain.py`), plus `train_model.py` (migrated by this task), only
     ever use `.connection` and `.close()`. The docstring was corrected accordingly
     (commit `0e73d7e`), and this ADR keeps that honesty: the two methods
     are a low-cost convenience addition, not a fix.
   - `.close()`, `__enter__`/`__exit__` — context-manager support, added for parity with
     `BaseStorage` in the storage tier, so callers can write
     `with open_repository(...) as repo:` instead of an explicit `try`/`finally`.

3. **Migration of the four locations**, across this task sequence:
   - `app/main.py` + `app/dependencies.py` — `app.state.repo` (a `DuckDBRepository`)
     replaces `app.state.db` (a raw connection); `get_db()` returns the repository.
   - `src/data/cards/storage/health.py` — all three connections (Bronze, Silver, Gold).
   - `scripts/check_and_retrain.py` — the single Gold connection.
   - `scripts/train_model.py` (this task) — the single Gold connection:
     `repo = open_repository(args.db_path, read_only=True)`, `conn = repo.connection`
     passed to `get_latest_gold_snapshot_date` and `retrain` exactly as the raw
     connection was before. `train_model.py` never called `conn.close()` explicitly
     (the process exits after printing the run ID), so there is no corresponding
     `repo.close()` to add.

**This does not amend ADR-024's core decision.** `DuckDBRepository` is deliberately
*not* threaded through `src.ml.features.pipeline.build_inference_features`,
`src.ml.training.*` (e.g. `walk_forward_cv`), or the monitoring functions in
`src.monitoring.*` — these keep taking a raw `duckdb.DuckDBPyConnection` exactly as
before. Two reasons:
- They run genuinely custom, per-function SQL (window functions, JSON extraction,
  aggregation) — the pattern ADR-024 established. Wrapping every such function's
  signature in `DuckDBRepository` would be a type change with no behavioral upside,
  since none of them need `.get_tables()`/`.query_df()` or connection lifecycle
  management — they receive an already-open connection and never own its lifecycle.
- The actual defect being fixed here is at the four *connection-creation* locations,
  not at every function that later borrows the connection. Threading a new type through
  every SQL-running function signature in the codebase would be disproportionate
  churn relative to that defect.

**The `PriceModel`/`lightgbm` type-alias idea (from an earlier planning draft) is
explicitly rejected.** Commit `97b575d` ("fix: correct require_model's type from
LightGBMPriceModel to lgb.Booster, matching what's actually injected") is a very
recent, deliberate correctness fix: `app/dependencies.py::require_model` previously
claimed to return `LightGBMPriceModel`, a wrapper class, when the value actually
populated into `app.state.model` (via `load_model_from_mlflow` in
`src.ml.training.tracking`) is a plain `lgb.Booster`. Introducing a new alias for this
now — even one intended only for `DuckDBRepository`-adjacent code — would reverse that
fix's intent (making the annotated type diverge from the runtime type again) for no
functional gain to this ADR's scope.

## Consequences

### Positive

- One factory (`open_connection`) instead of six independently-implemented
  `duckdb.connect(...)` calls across four locations, each of which previously had to
  remember read-only handling and had no shared error path.
- `StorageConnectionError` wrapping and structured logging (`open_connection`'s
  `logger.info("Connected to DuckDB (read_only=%s) at %s", ...)`) now apply uniformly
  at all four locations. Before this sequence, only the storage tier (via `BaseStorage`)
  got this — the four outside-tier locations raised whatever `duckdb.Error` DuckDB
  happened to throw, uncaught and unlogged.
- `DuckDBRepository` gives call sites outside the storage tier a named, injectable type
  (useful for `app.state.repo` and FastAPI's `Depends()` machinery in
  `app/dependencies.py::get_db`) instead of importing `duckdb.DuckDBPyConnection`
  directly everywhere.

### Negative

- Extra indirection at the app layer: `app.state.repo.connection` where
  `app.state.db` used to be a raw connection directly. Callers that only ever wanted
  the connection now go through one more attribute access.
- `get_tables()`/`query_df()` exist on `DuckDBRepository` as convenience methods without
  corresponding to an observed duplication problem at any of the four migrated
  locations — all of them use only `.connection` and (where applicable) `.close()`/the
  context-manager protocol. This is called out explicitly rather than oversold, per the
  corrected class docstring (commit `0e73d7e`).

## Alternatives Considered

| Approach | Reason rejected |
|---|---|
| Full `Protocol`/repository type threaded through every SQL-running function (`build_inference_features`, `walk_forward_cv`, `src.monitoring.*`) | Disproportionate churn relative to the actual defect (uncontrolled connection creation at four locations). Those functions run genuinely custom per-function SQL per ADR-024 and never own connection lifecycle — there is nothing for a repository type to centralize there. |
| `PriceModel`/`lightgbm` type alias for model-adjacent code | Rejected — reverses the intent of a very recent, deliberate correctness fix (commit `97b575d`) that corrected `require_model`'s return type from a wrapper class to `lgb.Booster` to match what MLflow actually returns at runtime. Introducing a new alias now would make the annotated type diverge from the runtime type again, for no gain within this ADR's scope. |
| Leave the four duplicated `duckdb.connect(...)` locations as-is | Rejected — each location had independently omitted the error-wrapping and logging the storage tier already had via `BaseStorage._open_connection`, meaning connection failures outside the storage tier failed differently (and less informatively) depending on which of the four locations hit them. |

## Affected ADRs

- **ADR-024** — Clarifies scope: this ADR governs connection *creation* outside the
  storage tier. It does not amend ADR-024's decision that DuckDB is the compute layer
  for large Silver-history queries, and does not change the type
  (`duckdb.DuckDBPyConnection`) those queries run over.
