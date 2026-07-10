"""Local, account-free-by-default alerting for scheduled jobs and API
degraded-mode events.

Three channels, all best-effort except the first:
    1. A durable JSON-lines log at ``alerts_log_path`` (default
       ``logs/alerts.jsonl``) — every alert is appended here regardless of
       whether the other two channels succeed, so any future
       dashboard/tool can replay history. This one always succeeds or logs
       why it didn't; it never depends on external services.
    2. A best-effort desktop notification via ``plyer`` — visible
       immediately if the machine is logged in and unlocked, but never
       required: a failure to notify (headless environment, missing OS
       backend, no display) is caught and logged, never raised.
    3. An optional HTTP POST to ``ALERT_WEBHOOK_URL`` (read from the
       environment at call time) — a Slack/Discord/Mattermost-compatible
       incoming webhook. Skipped entirely if the env var is unset; a
       failed request is caught and logged, never raised.

This project has no Slack/email/PagerDuty credentials configured by
default (see docs/runbooks/model-incidents.md, "Alerting") — set
``ALERT_WEBHOOK_URL`` in the deployment environment to enable channel 3.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import httpx as httpx  # explicit re-export: tests patch alerts.httpx.post,
# which requires this module to explicitly re-export the name under mypy
# --strict (no_implicit_reexport) — see the same pattern in
# scripts/check_and_retrain.py (duckdb) and scripts/backup_data.py (shutil).

from src.logger import get_logger


logger = get_logger(__name__)

ALERTS_LOG_PATH = Path("logs/alerts.jsonl")


def send_alert(
    subject: str,
    message: str,
    *,
    severity: str = "error",
    alerts_log_path: Path = ALERTS_LOG_PATH,
) -> None:
    """Record *subject*/*message* as an alert, best-effort notify the desktop,
    and best-effort POST to a configured webhook.

    Never raises: a failure in any of the best-effort channels is logged and
    swallowed so a broken alert path can never crash the caller's actual job.

    Args:
        subject:         Short alert title (e.g. "Backup failed").
        message:         Full alert body (e.g. the exception string).
        severity:        Free-text severity label, stored in the log line.
                          Defaults to "error".
        alerts_log_path: Where to append the JSON-lines record. Overridable
                          for tests; defaults to ``logs/alerts.jsonl``.
    """
    _append_to_log(subject, message, severity, alerts_log_path)
    _notify_desktop(subject, message)
    _notify_webhook(subject, message)


def _append_to_log(
    subject: str, message: str, severity: str, alerts_log_path: Path
) -> None:
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "severity": severity,
        "subject": subject,
        "message": message,
    }
    try:
        alerts_log_path.parent.mkdir(parents=True, exist_ok=True)
        with alerts_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except OSError as exc:
        logger.error("Could not write alert to %s: %s", alerts_log_path, exc)


def _notify_desktop(subject: str, message: str) -> None:
    try:
        from plyer import notification

        notification.notify(title=subject, message=message, timeout=10)
    except Exception as exc:
        logger.warning("Desktop notification failed (non-fatal): %s", exc)


def _notify_webhook(subject: str, message: str) -> None:
    webhook_url = os.getenv("ALERT_WEBHOOK_URL", "")
    if not webhook_url:
        return
    try:
        httpx.post(webhook_url, json={"text": f"*{subject}*\n{message}"}, timeout=5.0)
    except httpx.HTTPError as exc:
        logger.warning("Webhook alert failed (non-fatal): %s", exc)
