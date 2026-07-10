# ADR-014: HTTP Retry with Exponential Backoff for Download Functions

## Context

The ingesting pipeline makes 200+ HTTP requests per run — one bulk JSON download
per source and one HTML request per tournament, event, and deck on mtgtop8.com.
External services (mtgtop8.com, mtggoldfish.com, Cardmarket) do not guarantee
availability and enforce rate limits in production.

`download_json_from_url()` and `download_html_page()` previously made a single
`requests.get()` call with no retry logic. A single transient error (429, 503)
would abort the entire pipeline, discarding all work done in that run.

## Decision

Add exponential backoff retry to both download functions using the `tenacity`
library, applied at the raw HTTP fetch level (`_fetch_with_retry`, shared by
both download functions) before errors are wrapped in `SourceDownloadError`.

**Parameters:**
- Maximum 5 attempts
- Exponential backoff: 1s → 2s → 4s → 8s → 16s (multiplier 2, cap 30s)
- Retry only on transient status codes: `429`, `500`, `502`, `503`, `504`
- Permanent errors (`404`, `401`, `403`) raise immediately — retrying them
  wastes time and cannot succeed
- Each retry attempt is logged at `WARNING` level via `before_sleep_log`

**Architecture:** The retry decorator is applied to the private
`_fetch_with_retry` helper — a single generic fetch-then-parse function shared
by both `download_json_from_url` and `download_html_page` — not to the public
`download_*` functions themselves. This keeps the separation clean: the inner
helper raises `requests.HTTPError` (which tenacity can intercept), while the
outer `download_*` functions wrap the final failure in `SourceDownloadError`.

## Consequences

### Positive
- Pipeline survives transient network hiccups and rate-limit responses without
  manual intervention.
- 200+ request runs against mtgtop8.com are resilient to occasional 429s.
- Retry behaviour is independently testable via `_is_retryable_http_error`.

### Negative
- In the worst case (5 attempts, full backoff) a single failing request adds
  up to 1 + 2 + 4 + 8 + 16 = 31 seconds of delay before the error propagates.
- Adds `tenacity` as a runtime dependency.

### Neutral
- Rate limiting is reactive (retries after 429) not proactive (no global
  request throttle). A proactive throttle is a separate concern and is not
  implemented here.

## Alternatives Considered

| Approach | Reason rejected |
|---|---|
| Manual retry loop with `time.sleep` | More code, harder to test, `tenacity` covers all edge cases (jitter, max attempts, logging) out of the box |
| `urllib3` built-in retry via `HTTPAdapter` | Tied to the transport layer; harder to control which status codes trigger retry and does not support `before_sleep` logging |
| No retry | Single transient error aborts a 20-minute pipeline run — unacceptable in production |
