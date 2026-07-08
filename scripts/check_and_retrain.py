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
import sys
from datetime import datetime, timezone
from pathlib import Path

import duckdb as duckdb  # explicit re-export: tests patch check_and_retrain.duckdb.connect,
# which requires this module to explicitly re-export the name under mypy --strict
# (no_implicit_reexport) — a plain `import duckdb` makes the attribute invisible
# to importers even though it works fine at runtime.

from scripts._common import gold_db_exists
from src.data.cards.storage.gold.storage import get_latest_gold_snapshot_date
from src.data.repository import GOLD_DB_PATH, open_repository
from src.logger import get_logger, setup_logging
from src.monitoring.retraining import retrain, should_retrain

STATUS_PATH = Path("logs/last_check_status.json")

logger = get_logger(__name__)


def _write_status(status: dict[str, object]) -> None:
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(json.dumps(status, indent=2))


def _check_preconditions(
    conn: duckdb.DuckDBPyConnection,
) -> tuple[bool, str, str | None]:
    """Check whether a retrain should run.

    Returns (triggered, reason, snapshot_date). ``snapshot_date`` is None
    when not applicable (no trigger fired, or no gold snapshot exists).
    """
    triggered, reason = should_retrain(conn)
    if not triggered:
        return False, reason, None
    snapshot_date = get_latest_gold_snapshot_date(conn)
    return True, reason, snapshot_date


def _do_retrain(
    conn: duckdb.DuckDBPyConnection, snapshot_date: str, reason: str
) -> tuple[bool, dict[str, object]]:
    logger.warning(
        "Retrain triggered (reason=%s) — retraining on snapshot %s.",
        reason,
        snapshot_date,
    )
    try:
        run_id = retrain(conn, snapshot_date)
    except Exception as exc:
        logger.error("Retrain failed: %s", exc)
        return False, {
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "result": "error",
            "reason": "retrain_failed",
            "error": str(exc),
        }

    logger.info("Retrain complete. New run_id: %s", run_id)
    return True, {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "result": "retrained",
        "reason": reason,
        "run_id": run_id,
    }


def main() -> int:
    setup_logging(log_dir=Path("logs"))

    if not gold_db_exists(GOLD_DB_PATH):
        _write_status(
            {
                "checked_at": datetime.now(timezone.utc).isoformat(),
                "result": "error",
                "reason": "gold_db_missing",
            }
        )
        return 1

    repo = open_repository(GOLD_DB_PATH, read_only=True)
    conn = repo.connection
    try:
        triggered, reason, snapshot_date = _check_preconditions(conn)

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

        if snapshot_date is None:
            logger.error("gold_price_features is empty — cannot retrain.")
            _write_status(
                {
                    "checked_at": datetime.now(timezone.utc).isoformat(),
                    "result": "error",
                    "reason": "no_snapshot",
                }
            )
            return 1

        ok, status = _do_retrain(conn, snapshot_date, reason)
        _write_status(status)
        return 0 if ok else 1
    finally:
        repo.close()


if __name__ == "__main__":
    sys.exit(main())
