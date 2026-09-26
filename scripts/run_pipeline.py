"""ETL pipeline entry point — downloads, validates, and loads all three
medallion tiers (Bronze/Silver/Gold) via daily_pipeline().

Writes a JSON status file to ``logs/last_pipeline_status.json`` and sends a
best-effort alert (see src.monitoring.alerts) on failure. Mirrors the
pattern already used by scripts/check_and_retrain.py so a failed overnight
ETL run is as observable as a failed retrain — previously the only way to
notice was reading the day's log file by hand.

Three outcomes, not two:
    "success"  — every tier ran and every health check passed.
    "degraded" — the run finished, but a Bronze source delivered nothing or a
                 health check FAILed. The pipeline does not raise in either
                 case, so this state used to be recorded as "success"; a
                 36-day Scryfall outage (2026-07-29 → 2026-09-02) was reported
                 as a successful run every single day and was found two months
                 late, by accident.
    "error"    — daily_pipeline raised.

Usage:
    python -m scripts.run_pipeline
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx as httpx  # explicit re-export: tests patch run_pipeline.httpx.get,
# which requires this module to explicitly re-export the name under mypy
# --strict (no_implicit_reexport) — same pattern as scripts/check_and_retrain.py.

from src.data.cards.pipelines import daily_pipeline
from src.logger import get_logger, setup_logging
from src.monitoring.alerts import send_alert

STATUS_PATH = Path("logs/last_pipeline_status.json")

logger = get_logger(__name__)


def _write_status(status: dict[str, object]) -> None:
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(json.dumps(status, indent=2))


def _ping_heartbeat(ok: bool) -> None:
    """Best-effort GET to HEARTBEAT_URL (success) or HEARTBEAT_URL/fail
    (failure) — a dead-man's-switch for the ETL schedule itself.

    Without it, a scheduled task that stops running altogether leaves the last
    status file frozen on its final value and nothing ever contradicts it.
    Skipped entirely if HEARTBEAT_URL is unset. Never raises.
    """
    heartbeat_url = os.getenv("HEARTBEAT_URL", "")
    if not heartbeat_url:
        return
    url = heartbeat_url if ok else f"{heartbeat_url}/fail"
    try:
        httpx.get(url, timeout=5.0)
    except httpx.HTTPError as exc:
        logger.warning("Heartbeat ping failed (non-fatal): %s", exc)


def main() -> int:
    log_file = setup_logging(log_dir=Path("logs"))
    if log_file:
        print(f"Logging to {log_file}")
    config_path = "configs/data_sources.yaml"

    try:
        outcome = daily_pipeline(config_path)
    except Exception as exc:
        logger.error("ETL pipeline failed: %s", exc, exc_info=True)
        send_alert("ETL pipeline failed", str(exc))
        _write_status(
            {
                "checked_at": datetime.now(timezone.utc).isoformat(),
                "result": "error",
                "error": str(exc),
            }
        )
        _ping_heartbeat(ok=False)
        return 1

    if outcome.is_degraded:
        detail = outcome.summary()
        logger.error("ETL pipeline degraded — %s", detail)
        send_alert("ETL pipeline degraded", detail)
        _write_status(
            {
                "checked_at": datetime.now(timezone.utc).isoformat(),
                "result": "degraded",
                "detail": detail,
                "failed_sources": outcome.failed_sources,
                "failed_checks": [
                    f"{r.layer}/{r.name}: {r.detail}" for r in outcome.failed_checks
                ],
            }
        )
        _ping_heartbeat(ok=False)
        return 1

    _write_status(
        {
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "result": "success",
        }
    )
    _ping_heartbeat(ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
