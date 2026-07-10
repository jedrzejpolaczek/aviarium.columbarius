"""ETL pipeline entry point — downloads, validates, and loads all three
medallion tiers (Bronze/Silver/Gold) via daily_pipeline().

Writes a JSON status file to ``logs/last_pipeline_status.json`` and sends a
best-effort alert (see src.monitoring.alerts) on failure. Mirrors the
pattern already used by scripts/check_and_retrain.py so a failed overnight
ETL run is as observable as a failed retrain — previously the only way to
notice was reading the day's log file by hand.

Usage:
    python -m scripts.run_pipeline
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.data.cards.pipelines import daily_pipeline
from src.logger import get_logger, setup_logging
from src.monitoring.alerts import send_alert

STATUS_PATH = Path("logs/last_pipeline_status.json")

logger = get_logger(__name__)


def _write_status(status: dict[str, object]) -> None:
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(json.dumps(status, indent=2))


def main() -> int:
    log_file = setup_logging(log_dir=Path("logs"))
    if log_file:
        print(f"Logging to {log_file}")
    config_path = "configs/data_sources.yaml"

    try:
        daily_pipeline(config_path)
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
        return 1

    _write_status(
        {
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "result": "success",
        }
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
