# ADR-019: FastAPI Startup Pre-computation Strategy

## Context

The REST API exposes five endpoints: `/health`, `/cards`, `/predict/{card_name}`,
`/similar/{card_name}`, and `/underpriced`.  Each of the latter three requires:

1. A DuckDB query to retrieve price and card feature data.
2. An sklearn pipeline transformation to produce model-ready features.
3. An ML inference call (LightGBM booster or similarity index).

Two competing designs were considered:

**Option A — per-request computation**: Each request opens a DuckDB connection,
queries the relevant rows, runs the pipeline, and calls the model.

**Option B — startup pre-computation**: A single DuckDB query and pipeline
`fit_transform` runs at startup.  Requests read from pre-built DataFrames
stored in `app.state`.

## Decision

Use **startup pre-computation** (Option B).

At startup, `lifespan` in `app/main.py`:

1. Connects DuckDB (read-only) and stores the repository in `app.state.repo`.
2. Builds `X_all` — the full feature matrix for all cards at the latest
   price snapshot — by joining `build_lag_features` output with
   `gold_card_features`.
3. Fits the sklearn `Pipeline` once on `X_all`; stores the pre-transformed
   matrix as `X_all_t` (a float64 DataFrame with pipeline feature names as
   columns).
4. Loads the LightGBM booster from MLflow and stores it in `app.state.model`.
5. Builds a `CardSimilarityIndex` with `n_neighbors=50`.

## Consequences

### Positive

- **O(1) per-request lookup**: `/predict` slices one row from `X_all_t` by
  DataFrame index, then calls `model.predict(1-row df)`.  No DuckDB IO per request.
- **O(n) card listing**: `/cards` reads `X_all` (already in memory) and returns
  the full catalogue — uuid, name, set_code, rarity, eur — with no DuckDB IO.
- **Consistent snapshot**: All endpoints in a single server instance answer from
  the same snapshot date — no partial updates mid-request.
- **Pipeline fitted on full population**: SimpleImputer medians are derived from
  the full card catalogue rather than a single-card "training set" of one row.
- **Similarity index built once**: Cosine NearestNeighbors over ~80k cards takes
  ~2 seconds at startup; repeating it per request would cause 30-second timeouts.

### Negative

- **Stale data**: Prices from a new ETL run are not reflected until the server
  is restarted.  Acceptable for a daily-update pipeline; add a `/refresh` admin
  endpoint if intraday freshness is required.
- **Startup latency**: Connecting DuckDB, building features, and fitting the
  similarity index takes ~10–30 seconds.  Docker health-check `start_period`
  should be set to at least 60 seconds.
- **Memory**: Storing `X_all` (~80k × 35 columns, ~22 MB) and `X_all_t`
  (~80k × 22 columns, ~14 MB) in memory is acceptable on any modern server
  (≥ 512 MB RAM).

## Why app.state over Global Variables

FastAPI's `app.state` is the idiomatic way to share data across requests.
Module-level globals:

- Are invisible to tests (can't be injected or monkeypatched cleanly).
- Create implicit coupling between `main.py` and every router that imports
  the global.
- Break when running multiple worker processes (each process has its own globals).

`app.state` is accessible from every handler via `request.app.state` and can
be fully replaced in tests by creating a separate `FastAPI` instance with a
mock lifespan — no patching required.

## Why n_neighbors=50 at Startup

The `/similar` endpoint accepts `n` between 1 and 50.  `CardSimilarityIndex`
requires `n_neighbors` to be set at `fit()` time (NearestNeighbors is
pre-built for a fixed k).  Fitting with `n_neighbors=50` at startup allows any
`n ≤ 50` without re-fitting; handlers truncate with `similar_df.head(n)`.

An alternative — fitting with a dynamic k per request — would require either
re-fitting the entire index (expensive) or rebuilding NearestNeighbors for each
query, which defeats the purpose of the pre-built index.

## Why Degraded Mode Instead of Hard Failure

`MODEL_RUN_ID` may be unset during development or when the team is between
model runs.  Requiring a valid model at startup would prevent developers from
running the API locally to test `/health` or `/similar`.  The degraded mode
allows the server to start and serve model-independent endpoints; `/predict`
and `/underpriced` return 503 with a clear message.
