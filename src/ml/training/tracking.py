"""
MLflow experiment tracking — wraps the MLflow API so model code stays clean.

WHY MLFLOW:
MLflow records every experiment run: hyperparameters, metrics, and the model
artefact. Any past run can be reproduced or rolled back. The run_id returned
by log_model() is the identifier used by the FastAPI service at startup.

MOST IMPORTANT PARAMETER:
Always log 'gold_snapshot_date' — it provides full data lineage:
  run_id → gold_snapshot_date → Silver join → Bronze download.

QUICK START:
After the first run, launch `mlflow ui` and open http://localhost:5000
to browse all experiments and compare runs visually.
"""

import os
import tempfile
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import pandas as pd


EXPERIMENT_NAME = "mtg_price_prediction"

# Project root: src/ml/training/ → src/ml/ → src/ → project root
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
# Default tracking DB at project root so all training scripts share the same store.
# Override with MLFLOW_TRACKING_URI env var (e.g. in Docker: sqlite:////app/mlflow.db).
_DEFAULT_TRACKING_URI = f"sqlite:///{_PROJECT_ROOT / 'mlflow.db'}"


def setup_experiment(name: str = EXPERIMENT_NAME) -> None:
    """Create the MLflow experiment if it does not exist yet.

    Sets the tracking URI to the project-root SQLite database unless
    ``MLFLOW_TRACKING_URI`` is already set in the environment.
    Call once at the start of a training script before opening any run.

    Args:
        name: Experiment name (defaults to EXPERIMENT_NAME).
    """
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri(_DEFAULT_TRACKING_URI)
    mlflow.set_experiment(name)


@contextmanager
def start_run(
    run_name: str, snapshot_date: str = ""
) -> Generator[mlflow.ActiveRun, None, None]:
    """Context manager that opens an MLflow run and logs core lineage params.

    Always logs 'gold_snapshot_date' so every run is traceable back to the
    exact data slice it was trained on.

    Args:
        run_name:      Human-readable label shown in the MLflow UI.
        snapshot_date: ISO date string of the gold snapshot used for training.

    Yields:
        mlflow.ActiveRun — call run.info.run_id to get the run identifier.

    Usage:
        with start_run("tiered_lgbm_v1", "2026-06-09") as run:
            log_metrics({"mae_tier1": 0.15})
            log_model(model, "model_tier1")
            print(run.info.run_id)
    """
    with mlflow.start_run(run_name=run_name) as run:
        if snapshot_date:
            mlflow.log_param("gold_snapshot_date", snapshot_date)
        mlflow.log_param("run_name", run_name)
        yield run


def log_params(params: Any) -> None:
    """Log model hyperparameters to the active MLflow run.

    Accepts a dataclass (e.g. LightGBMParams), a plain dict, or any object
    that supports vars(). Converts to a flat dict before logging.

    Args:
        params: Hyperparameters in any of the above forms.
    """
    if isinstance(params, dict):
        mlflow.log_params(params)
    else:
        try:
            mlflow.log_params(vars(params))
        except TypeError:
            mlflow.log_param("params", str(params))


def log_metrics(metrics: dict[str, float], step: int | None = None) -> None:
    """Log evaluation metrics to the active MLflow run.

    Args:
        metrics: Dict of metric name → value,
                 e.g. {"mae_tier1": 0.15, "mape_tier1": 12.3}.
        step:    Optional training step / fold index for time-series charts.
    """
    mlflow.log_metrics(metrics, step=step)


def log_model(model: Any, artifact_path: str = "model") -> str:
    """Save a LightGBM model as an MLflow artefact and return the run_id.

    The FastAPI service uses this run_id to load the model at startup:
        MODEL_RUN_ID=<run_id> in docker-compose.yml.

    Args:
        model:         Fitted LightGBMPriceModel (model.model is a lgb.Booster).
        artifact_path: Sub-path within the MLflow run artefact store.

    Returns:
        run_id string of the active run.
    """
    mlflow.lightgbm.log_model(model.model, artifact_path)
    active = mlflow.active_run()
    if active is None:
        raise RuntimeError("No active MLflow run.")
    return str(active.info.run_id)


def load_model_from_mlflow(run_id: str) -> lgb.Booster:
    """Load a LightGBM booster from MLflow by run_id.

    Used by app/main.py in the lifespan event at server startup.

    Args:
        run_id: The run_id returned by log_model().

    Returns:
        lgb.Booster loaded from the MLflow artefact store.
    """
    model_uri = f"runs:/{run_id}/model"
    return cast(lgb.Booster, mlflow.lightgbm.load_model(model_uri))


def log_cv_results(cv_df: pd.DataFrame) -> None:
    """Log walk-forward CV results as a CSV artefact and averaged metrics.

    Saves the full per-fold DataFrame so it can be downloaded from the
    MLflow UI, then computes mean MAE and MAPE per tier and logs them
    as scalar metrics for easy comparison across runs.

    Args:
        cv_df: DataFrame returned by trainer.walk_forward_cv(),
               must contain columns: fold_idx, tier, mae, mape.
    """
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        cv_df.to_csv(f, index=False)
        tmp_path = f.name

    mlflow.log_artifact(tmp_path, "cv_results")

    for tier in cv_df["tier"].unique():
        tier_df = cv_df[cv_df["tier"] == tier]
        log_metrics(
            {
                f"cv_mae_tier{tier}": float(tier_df["mae"].mean()),
                f"cv_mape_tier{tier}": float(tier_df["mape"].mean()),
            }
        )
