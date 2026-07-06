"""Scheduled drift/MAPE check with conditional retraining.

Run on a daily schedule (cron / Windows Task Scheduler) after the ETL
pipeline (`make pipeline`). Wraps :func:`should_retrain` /
:func:`retrain` from ``src.monitoring.retraining`` so retraining only
happens when a real trigger fires (ban/unban event or a 3-day MAPE
alert), instead of retraining unconditionally like
``scripts/train_model.py`` does.

Writes a JSON status file to ``logs/last_check_status.json`` on every run
so an operator (or a future alerting tool) can check the outcome without
reading log files. See docs/runbooks/model-incidents.md for what to do
with each result.

Usage:
    python -m scripts.check_and_retrain
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import duckdb

from src.logger import get_logger, setup_logging
from src.monitoring.retraining import retrain, should_retrain

GOLD_DB_PATH = os.getenv("GOLD_DB_PATH", "data/gold/cards.duckdb")
STATUS_PATH = Path("logs/last_check_status.json")

logger = get_logger(__name__)


def _write_status(status: dict[str, object]) -> None:
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(json.dumps(status, indent=2))


def main() -> int:
    setup_logging(log_dir=Path("logs"))

    if not os.path.exists(GOLD_DB_PATH):
        logger.error(
            "Gold DB not found: %s — run the ETL pipeline first.", GOLD_DB_PATH
        )
        _write_status(
            {
                "checked_at": datetime.now(timezone.utc).isoformat(),
                "result": "error",
                "reason": "gold_db_missing",
            }
        )
        return 1

    conn = duckdb.connect(GOLD_DB_PATH, read_only=True)
    try:
        triggered, reason = should_retrain(conn)

        if not triggered:
            logger.info("No retrain trigger fired (reason=%s).", reason)
            _write_status(
                {
                    "checked_at": datetime.now(timezone.utc).isoformat(),
                    "result": "no_retrain",
                    "reason": reason,
                }
            )
            return 0

        row = conn.execute(
            "SELECT MAX(snapshot_date) FROM gold_price_features"
        ).fetchone()
        if row is None or row[0] is None:
            logger.error("gold_price_features is empty — cannot retrain.")
            _write_status(
                {
                    "checked_at": datetime.now(timezone.utc).isoformat(),
                    "result": "error",
                    "reason": "no_snapshot",
                }
            )
            return 1
        snapshot_date = str(row[0])

        logger.warning(
            "Retrain triggered (reason=%s) — retraining on snapshot %s.",
            reason,
            snapshot_date,
        )
        try:
            run_id = retrain(conn, snapshot_date)
        except Exception as exc:
            logger.error("Retrain failed: %s", exc)
            _write_status(
                {
                    "checked_at": datetime.now(timezone.utc).isoformat(),
                    "result": "error",
                    "reason": "retrain_failed",
                    "error": str(exc),
                }
            )
            return 1

        logger.info("Retrain complete. New run_id: %s", run_id)
        _write_status(
            {
                "checked_at": datetime.now(timezone.utc).isoformat(),
                "result": "retrained",
                "reason": reason,
                "run_id": run_id,
            }
        )
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
