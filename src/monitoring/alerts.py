"""Local, account-free alerting for scheduled jobs and API degraded-mode events.

Two channels, both always attempted:
    1. A durable JSON-lines log at ``alerts_log_path`` (default
       ``logs/alerts.jsonl``) — every alert is appended here regardless of
       whether the desktop notification succeeds, so any future
       dashboard/tool can replay history.
    2. A best-effort desktop notification via ``plyer`` — visible
       immediately if the machine is logged in and unlocked, but never
       required: a failure to notify (headless environment, missing OS
       backend, no display) is caught and logged, never raised.

This project has no Slack/email/PagerDuty credentials configured (see
docs/runbooks/model-incidents.md, "Known limitation"). This module is the
minimal upgrade from "nobody is notified" to "an alert is durably recorded
and, best-effort, shown on the operator's screen" — without requiring any
external account.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

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
    """Record *subject*/*message* as an alert and best-effort notify the desktop.

    Never raises: a failure in either channel is logged and swallowed so a
    broken alert path can never crash the caller's actual job.

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
