# ADR-032: Hot Model-Reload Endpoint

**Date:** 2026-07-10
**Status:** Accepted

## Context

`scripts/rollback_model.py` re-aliases the MLflow Registry `production` alias to an older, known-good model version — an instant operation at the registry layer. But per ADR-019, `app/main.py`'s `lifespan` reads `MODEL_RUN_ID` exactly once, at process startup, and loads that model into `app.state.model`. A registry alias swap alone does not change what the *running* API process has in memory: an operator must also edit `docker/.env` and restart the container for a rollback (or a fresh retrain's new run_id) to actually take effect.

This means the real MTTR for a bad-model incident is not "one alias reassignment" (as ADR-020 describes for the registry layer) but "one alias reassignment **plus** a container restart" — and a restart re-runs the entire `lifespan` startup sequence (DuckDB connect, feature-matrix build, pipeline fit, similarity-index build), none of which needs to change just to swap the model.

## Decision

A new, independent router, `app/routers/admin.py`, exposing `POST /admin/reload-model`:

- Request body: `{"model_run_id": "<run_id>"}` (a `ReloadModelRequest` Pydantic model, one field).
- Auth: a shared-secret `X-Admin-Token` header, compared against the `ADMIN_TOKEN` environment variable using `hmac.compare_digest` (constant-time, not `!=`) — this is the first credential-gated surface anywhere in this API (see "Threat Model" below). Returns 503 if `ADMIN_TOKEN` isn't configured at all (hot reload is opt-in per deployment, not silently open when nobody set a token), 403 on a mismatch.
- On a valid request: calls `load_model_from_mlflow(model_run_id)` (the same function `lifespan` already uses at startup). On success, swaps `app.state.model` and `app.state.model_run_id` in place and returns `{"status": "reloaded", "model_run_id": ...}` (200). On an MLflow load failure, returns 502 and **leaves `app.state` untouched** — the state mutation happens only after the load has already succeeded, so a bad `model_run_id` never leaves the API half-updated.

This does not touch `X_all`, `X_all_t`, `pipeline`, `feature_names`, `similarity_index`, or `repo` — those remain startup-only, exactly as ADR-019 describes. Only `model`/`model_run_id` become mutable post-startup.

## Threat Model

Before this change, the API had no authenticated endpoint of any kind — `SECURITY.md` states the project "does not handle user authentication ... the primary security concern is dependency supply-chain risk." This endpoint is a deliberate, narrow exception: it can replace the production model, so it is gated behind a secret the operator must explicitly configure. A rejected token is logged (`logger.warning`) for basic auditability; there is no rate-limiting on repeated failed attempts, which is judged acceptable for this project's single-operator scale but is worth revisiting if the deployment model ever changes.

## Alternatives Considered

| Approach | Reason rejected |
|---|---|
| File-watcher on `docker/.env` (e.g. inotify) triggering an automatic reload | Adds a background thread/process to a single-worker sync API for a rare, manually-triggered event. An explicit endpoint is simpler to reason about, test, and audit (who called it, when) than an implicit filesystem watch. |
| OAuth/JWT-based admin auth | Disproportionate for a single-operator deployment with no existing identity provider. A shared-secret header matches the project's current "no external auth" posture exactly — it's the minimum viable gate, not a general-purpose auth system. |
| Container-level supervisor watching `MODEL_RUN_ID` and restarting automatically | Still pays the restart-latency cost this ADR exists to remove, and additionally throws away the in-memory feature matrices, which don't need to be rebuilt to swap a model. |
| Re-fitting the whole `lifespan` sequence on reload (not just the model) | Unnecessary: `X_all`/`X_all_t`/`pipeline`/`similarity_index` are derived from the Gold snapshot, not from the model — swapping the model doesn't invalidate them. |

## Consequences

### Positive

- Rollback MTTR drops from "edit `docker/.env` + restart container" to "one authenticated POST" — no restart, no loss of the pre-built feature matrices.
- Failure is safe to retry: `app.state` is provably unchanged on a 502, verified by a dedicated test (`test_reload_model_returns_502_when_mlflow_load_fails`).

### Negative

- The `model`/`model_run_id` swap is two separate attribute writes, not one atomic operation. A request racing an in-flight reload (FastAPI's sync route handlers run in a threadpool, so real interleaving is possible) could in principle read the new model paired with the old `model_run_id` label, or vice versa — a cosmetic metadata mismatch, not a crash. Accepted as low-severity given hot reload is an infrequent, manually-triggered admin action, not something expected to race meaningful concurrent traffic.
- No rate-limiting on failed-auth attempts, only a log line — acceptable for the current single-operator threat model, flagged here for reconsideration if that ever changes.

## Affected ADRs

- **ADR-019** — Clarifies scope: `app.state.model` and `app.state.model_run_id` are no longer guaranteed immutable for the lifetime of the process. Every other `app.state` field ADR-019 and the C4 `api-state.md` diagram describe remains startup-only.
