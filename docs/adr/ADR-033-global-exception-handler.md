# ADR-033: Global API Exception Handler

**Date:** 2026-07-10
**Status:** Accepted

## Context

Every *anticipated* error path in this API already returns a structured, consistent body: `require_model` raises 503, `get_request_features` raises 503, `require_match` raises 404 — all via `HTTPException` with a `{"detail": ...}` payload. But an unhandled exception inside a route body (a bug triggered by unexpected input, or any code path none of those guards cover) had no equivalent: it propagated as FastAPI/Starlette's bare default 500, with no consistent client-facing body, no alert, and no log correlation beyond whatever Uvicorn's own traceback dump happened to capture.

## Decision

`app/main.py::register_exception_handlers(app)` registers a single catch-all handler via `@app.exception_handler(Exception)`, called immediately after `FastAPI(...)` construction and before the CORS middleware. On any exception not already resolved by FastAPI's `HTTPException` handling, it:

1. Logs at `error` level with `exc_info=True` (method, path, exception).
2. Calls the existing `send_alert("Unhandled API exception", f"{request.method} {request.url.path}: {exc}")` — the same account-free alerting mechanism from ADR-031, not a new channel.
3. Returns `JSONResponse(status_code=500, content={"detail": "Internal server error."})` — a fixed, generic message. The raw exception is never included in the response body, only in the log and the alert — a client learns nothing more than "something went wrong," while an operator gets the full detail through the existing observability surface.

FastAPI/Starlette registers `Exception`-keyed handlers into the outermost `ServerErrorMiddleware`, while `HTTPException` is resolved by the inner `ExceptionMiddleware` first — so this handler cannot intercept or mask any of the existing 404/503/502 paths (verified directly against Starlette's `build_middleware_stack`, and pinned by `test_existing_http_exceptions_are_not_affected_by_the_handler`).

## Alternatives Considered

| Approach | Reason rejected |
|---|---|
| Custom ASGI middleware wrapping every request in `try`/`except` | `@app.exception_handler(Exception)` already gives the correct precedence semantics (never intercepts `HTTPException`) without hand-rolling exception routing that FastAPI/Starlette already implements correctly. |
| A third-party error-tracking SaaS (Sentry, Rollbar) | Same "no external account configured by default" posture as ADR-031 — this project's alerting stays self-hosted (JSONL log + desktop notification + optional webhook), not tied to a new paid/free-tier service. |
| Returning the actual exception message to the client | Rejected as an information-disclosure risk — an internal exception string (e.g. from a future bug) could describe file paths, query fragments, or other implementation detail a client has no need to see. The fixed "Internal server error." string is deliberate. |

## Consequences

### Positive

- Any future unguarded bug now surfaces the same way an existing degraded-startup failure already does — logged and alerted, not silently swallowed into stdout.
- Consistent API contract: every error response from this API, anticipated or not, is a JSON body with a `detail` field.

### Negative

- The *alert* message (not the HTTP response) includes the raw exception string via `send_alert`, which is routed to the JSONL log, a desktop notification, and (if `ALERT_WEBHOOK_URL` is configured per ADR-031) an external webhook. If a future bug's exception message ever embedded something sensitive (e.g. a connection string), it would reach that webhook. This is the same tradeoff `lifespan`'s existing degraded-startup alert already makes — not a new risk introduced by this ADR, but worth naming since the webhook channel from ADR-031 widens the alert's reach.

## Affected ADRs

None amended.
