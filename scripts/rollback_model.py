"""Manual rollback of the production model to a previous MLflow Registry version.

Use when a newly promoted model (via src.monitoring.retraining.retrain) performs
worse in production than the previous version — this re-aliases "production" to
point at an older, known-good registered version instead of retraining.

List available versions first:
    uv run python -c "
    import mlflow
    mlflow.set_tracking_uri('sqlite:///mlflow.db')
    client = mlflow.tracking.MlflowClient()
    for v in client.search_model_versions(\"name='mtg_price_model'\"):
        print(v.version, v.run_id, v.aliases)
    "

Usage:
    python -m scripts.rollback_model --version 3
"""

import argparse
import sys as sys
from pathlib import Path

import mlflow

from src.logger import get_logger, setup_logging
from src.ml.training.tracking import setup_experiment
from src.monitoring.retraining import MODEL_REGISTRY_NAME

logger = get_logger(__name__)


def rollback(version: str, model_name: str = MODEL_REGISTRY_NAME) -> None:
    """Re-alias ``production`` to point at an existing registered model version.

    Args:
        version:    Registered model version number to roll back to (as a string).
        model_name: Registered model name. Defaults to MODEL_REGISTRY_NAME.

    Raises:
        mlflow.exceptions.MlflowException: If ``version`` does not exist.
    """
    client = mlflow.tracking.MlflowClient()
    client.set_registered_model_alias(model_name, "production", version)
    logger.info("Rolled back '%s' production alias to version %s.", model_name, version)


def main() -> int:
    setup_logging(log_dir=Path("logs"))
    setup_experiment()
    parser = argparse.ArgumentParser(
        description="Roll back the production model to a previous MLflow Registry version."
    )
    parser.add_argument(
        "--version", required=True, help="Registered model version to roll back to."
    )
    parser.add_argument("--model-name", default=MODEL_REGISTRY_NAME)
    args = parser.parse_args()

    try:
        rollback(args.version, args.model_name)
    except mlflow.exceptions.MlflowException as exc:
        logger.error("Rollback failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
