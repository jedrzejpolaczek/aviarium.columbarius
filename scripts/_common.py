"""Shared helpers for the CLI scripts in this directory."""

import os

from src.logger import get_logger

logger = get_logger(__name__)


def gold_db_exists(db_path: str) -> bool:
    """Return True if the Gold DuckDB file exists, logging an error if not.

    Both train_model.py and check_and_retrain.py need this check before
    connecting, but differ in what they do on failure (sys.exit vs. writing
    a JSON status file) — this helper owns only the check + log, callers
    own the failure action.
    """
    if not os.path.exists(db_path):
        logger.error("Gold DB not found: %s — run the ETL pipeline first.", db_path)
        return False
    return True
