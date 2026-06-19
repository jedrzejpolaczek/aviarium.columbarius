# ADR-016: Separation of `app/` and `src/` Layers

## Context

The project has two top-level Python packages that could plausibly be merged:

- `app/` — FastAPI application: routers, schemas, dependency injection, `main.py`
- `src/` — business logic: ML models, data pipeline, monitoring, logging

An alternative layout would place `app/` inside `src/`, treating the HTTP layer
as just another module of the core package.

## Decision

Keep `app/` and `src/` as separate, sibling top-level packages.

- `app/` knows about HTTP, FastAPI, and request/response contracts. It may import
  from `src/`, but `src/` must never import from `app/`.
- `src/` contains all logic that is independent of the transport layer: ML
  inference, data access, monitoring.

## Consequences

- Business logic in `src/` can be tested without starting the HTTP server.
- The boundary makes it obvious when HTTP concerns (status codes, serialization)
  are leaking into business logic.
- Adding a CLI, background worker, or alternative transport in the future only
  requires a new sibling entry-point; `src/` is left untouched.
- The one cost is that the import root is not a single package — `app` and `src`
  must both be on `PYTHONPATH` (or installed as packages), which `pyproject.toml`
  already handles.
