import argparse
import logging
import os
import sys

from scripts._common import gold_db_exists
from src.data.cards.storage.gold.storage import get_latest_gold_snapshot_date
from src.data.repository import open_repository
from src.ml.training.tracking import setup_experiment

GOLD_DB_PATH = os.getenv("GOLD_DB_PATH", "data/gold/cards.duckdb")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the MTG price prediction model."
    )
    parser.add_argument(
        "--db-path",
        default=GOLD_DB_PATH,
        help="Path to the Gold DuckDB file (default: data/gold/cards.duckdb or GOLD_DB_PATH env var)",
    )
    args = parser.parse_args()

    if not gold_db_exists(args.db_path):
        sys.exit(1)

    repo = open_repository(args.db_path, read_only=True)
    conn = repo.connection
    snapshot_date = get_latest_gold_snapshot_date(conn)

    if snapshot_date is None:
        logger.error("gold_price_features is empty — run the ETL pipeline first.")
        sys.exit(1)

    logger.info("Training on snapshot: %s", snapshot_date)

    setup_experiment()

    from src.monitoring.retraining import retrain

    run_id = retrain(conn, snapshot_date)

    print(f"\n{'=' * 60}")
    print(f"MODEL_RUN_ID = {run_id}")
    print(f"{'=' * 60}")
    print("\nUstaw w PowerShell:")
    print(f'  $env:MODEL_RUN_ID = "{run_id}"')
    print("\nUruchom API:")
    print("  uv run uvicorn app.main:app --reload")


if __name__ == "__main__":
    main()
