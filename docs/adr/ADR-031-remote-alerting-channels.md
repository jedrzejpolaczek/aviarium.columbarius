# ADR-031: Remote Alerting Channels (Webhook + Heartbeat)

**Date:** 2026-07-10
**Status:** Accepted

## Context

`src/monitoring/alerts.py` (introduced 2026-07-09) gave every scheduled script and the API a shared, account-free way to raise an alert: a durable JSONL log plus a best-effort desktop notification via `plyer`. Its own module docstring and `docs/runbooks/model-incidents.md` both explicitly stated the project had "no Slack/email/PagerDuty credentials configured."

The 2026-07-10 production-readiness audit flagged this as the single biggest remaining gap in that design, for two distinct reasons:

1. **Reach**: the desktop notification only appears if the operator happens to be logged in and looking at the machine that ran the failing script. There is no way to be notified anywhere else.
2. **Silent non-execution**: `send_alert` only fires from *inside* a run that actually started. If the scheduled task itself never runs at all — the host machine is off, the cron entry or Task Scheduler task was deleted, credentials expired — nothing raises an alert, because nothing ran to raise one.

## Decision

Two new, purely additive channels, both optional and both following the existing "durable-first, best-effort-second, never raise" pattern already established by `send_alert`:

1. **`ALERT_WEBHOOK_URL`** (`src/monitoring/alerts.py::_notify_webhook`) — a third channel inside `send_alert` itself. Read via `os.getenv` at call time (not import time, so it can be toggled per-deployment without a code change). If set, POSTs `{"text": f"*{subject}*\n{message}"}` via `httpx.post(..., timeout=5.0)` — the flat `{"text": ...}` shape is accepted by Slack, Discord, and Mattermost incoming webhooks without a client SDK. If unset, the channel is skipped entirely (no network call). A failed request is caught (`httpx.HTTPError`), logged, and swallowed — exactly like the existing desktop-notification failure path.

2. **`HEARTBEAT_URL`** (`scripts/check_and_retrain.py::_ping_heartbeat`) — a dead-man's-switch ping, the healthchecks.io convention: `GET HEARTBEAT_URL` on success, `GET HEARTBEAT_URL/fail` on failure, fired on **every** exit path of `main()` (gold DB missing, no trigger, no trainable snapshot, retrain outcome). Unlike `send_alert`, this is about the absence of a signal being the alert: if the scheduled task stops running at all, the external monitoring service (not this codebase) notices the missing ping and pages someone.

## Alternatives Considered

| Approach | Reason rejected |
|---|---|
| Slack-specific SDK (`slack_sdk`) | New dependency for a single HTTP POST; the generic `{"text": ...}` webhook payload already works against Slack, Discord, and Mattermost incoming webhooks without a client library. |
| SMTP email alerting | Requires an SMTP relay and credentials this project has never had configured — the same "account-free by default" constraint `alerts.py` was built around in the first place. A webhook URL is zero-config by comparison: point it at any of a dozen free-tier services. |
| Full paging integration (PagerDuty, Opsgenie) | Disproportionate to this project's actual operational scale — a solo-operator deployment processing public MTG price data, not a service with an on-call rotation (see `SECURITY.md`'s stated scope). |
| A dedicated heartbeat *library* instead of a plain `httpx.get` | The healthchecks.io convention is a bare GET to a URL — no protocol, no auth, no payload. A library would add a dependency for something `httpx.get(url, timeout=5.0)` already does in one line. |

## Consequences

### Positive

- Failures reach the operator wherever they are, not just whoever is looking at the machine's screen when a script happens to fail.
- The heartbeat closes a blind spot plain error-alerting structurally cannot cover: it is the first mechanism in this project that can detect its own *absence* of execution.
- Both channels are purely additive — `send_alert`'s existing durable JSONL log remains the reliable source of truth regardless of whether either new channel is configured or reachable.

### Negative

- Neither channel has delivery confirmation or retry. If the webhook endpoint is down, the alert silently doesn't arrive there — the JSONL log is the only channel with a durability guarantee.
- Two more environment variables (`ALERT_WEBHOOK_URL`, `HEARTBEAT_URL`) an operator must discover to get full observability. Documented in `docs/runbooks/model-incidents.md`'s "Alerting" section as the single place to look.

## Affected ADRs

None amended — this is a purely additive extension of the existing (pre-ADR) `alerts.py` design from the 2026-07-09 production-readiness blockers plan.
