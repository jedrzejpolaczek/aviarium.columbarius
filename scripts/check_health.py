"""Standalone entry point for data health checks.

Run after daily_pipeline to validate that all Bronze/Silver/Gold tables are
populated, fresh, and internally consistent. Exits with code 1 on any failure.

Usage:
    python -m scripts.check_health
"""

import datetime
from pathlib import Path

from src.data.cards.pipelines import load_config
from src.data.cards.storage.health import run_health_checks
from src.logger import setup_logging


def main() -> None:
    setup_logging(log_dir=Path("logs"))
    config = load_config("configs/data_sources.yaml")
    run_health_checks(
        bronze_path=config["storage"]["bronze_duckdb_path"],
        silver_path=config["storage"]["silver_duckdb_path"],
        gold_path=config["storage"]["gold_duckdb_path"],
        today=datetime.date.today(),
    )


if __name__ == "__main__":
    main()
