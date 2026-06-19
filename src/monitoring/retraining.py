"""Automated retraining pipeline — from trigger detection to MLflow promotion.

Retraining workflow:
    1. :func:`should_retrain` checks two independent signals in priority order:
       - Ban/unban event today → immediate retrain (don't wait for MAPE).
       - MAPE > 30% for 3 consecutive days → drift-induced retrain.
    2. :func:`retrain` builds a fresh LightGBM model via walk-forward CV,
       then trains a final model on the full latest snapshot, and logs
       everything to MLflow with the snapshot date as a lineage parameter.
    3. :func:`_compare_and_promote` compares the new model's CV MAPE against
       the current Production model.  Only promotes if the new model is better
       (or if no Production model exists yet).
    4. :func:`promote_to_production` registers the model in MLflow Registry
       and sets the ``production`` alias.

Rollback strategy:
    MLflow Registry retains all registered versions.  Promoting a new model
    does not delete the previous Production version — it moves to an
    "Archived" stage.  To rollback, re-alias the previous version.

Versioning:
    Each retrain run logs ``gold_snapshot_date`` as a parameter so any model
    version can be traced back to the exact data slice it was trained on.
"""

import duckdb
import pandas as pd

from src.logger import get_logger
from src.monitoring.event_trigger import get_todays_events, has_ban_event_today
from src.monitoring.mape_tracker import compute_rolling_mape, is_mape_alert


logger = get_logger(__name__)

MODEL_REGISTRY_NAME = "mtg_price_model"


def should_retrain(conn: duckdb.DuckDBPyConnection) -> tuple[bool, str]:
    """Decide whether retraining is needed and return the triggering reason.

    Checks two independent signals in priority order:
    1. Ban/unban event today → immediate retrain (format changes cause large
       price drops within 24 hours; waiting for MAPE wastes two days).
    2. MAPE > 30% for 3 consecutive days → accumulated prediction error.

    Args:
        conn: Open DuckDB connection with ``gold_events`` and
              ``gold_predictions`` / ``gold_price_features`` in scope.

    Returns:
        Tuple of ``(should_retrain: bool, reason: str)`` where ``reason`` is
        one of ``"ban_event"``, ``"mape_threshold"``, or ``"no_trigger"``.
    """
    if has_ban_event_today(conn):
        events = get_todays_events(conn)
        logger.info(
            "Ban/unban event detected — triggering immediate retrain: %s", events
        )
        return True, "ban_event"

    mape_df = compute_rolling_mape(conn)
    if is_mape_alert(mape_df):
        logger.info(
            "MAPE alert triggered. Last 3 days:\n%s", mape_df.tail(3).to_string()
        )
        return True, "mape_threshold"

    return False, "no_trigger"


def retrain(
    conn: duckdb.DuckDBPyConnection, snapshot_date: str
) -> str:  # pragma: no cover
    """Run the full retraining pipeline and return the new model's MLflow run_id.

    Steps:
    1. Run walk-forward CV with :class:`LightGBMPriceModel` to measure
       out-of-sample performance across historical folds.
    2. Build the full feature matrix at ``snapshot_date`` (same logic as
       ``app/main.py`` startup) and train a final model on all available data.
    3. Log CV results and the final model artefact to a new MLflow run.
    4. Call :func:`_compare_and_promote` to conditionally promote to Production.

    Args:
        conn:          Open DuckDB connection with both gold tables in scope.
        snapshot_date: ISO date string used to build the final production model.
                       Should be the latest snapshot with a t+7 counterpart.

    Returns:
        MLflow run_id of the new run (use as ``MODEL_RUN_ID`` env variable).

    Raises:
        RuntimeError: ``snapshot_date`` produces an empty training dataset
                      (either no lag features or no t+7 targets available).
    """
    from src.ml.features.lag import build_target
    from src.ml.features.pipeline import (
        build_feature_pipeline,
        build_inference_features,
        get_feature_names,
        LEAKAGE_COLS,
    )
    from src.ml.models.lightgbm_model import LightGBMPriceModel
    from src.ml.training.tracking import (
        log_cv_results,
        log_model,
        setup_experiment,
        start_run,
    )
    from src.ml.training.trainer import InsufficientDataError, walk_forward_cv

    setup_experiment()

    # 1. Walk-forward CV to measure out-of-sample performance
    try:
        cv_results = walk_forward_cv(conn, LightGBMPriceModel())
        logger.info(
            "Walk-forward CV complete: %d fold(s).",
            len(cv_results["fold_idx"].unique()) if not cv_results.empty else 0,
        )
    except InsufficientDataError as exc:
        logger.warning(
            "Insufficient data for CV (%s) — training final model only.", exc
        )
        cv_results = pd.DataFrame(
            columns=[
                "fold_idx",
                "val_snapshot",
                "model",
                "tier",
                "n_cards",
                "mae",
                "mape",
            ]
        )

    # 2. Build feature matrix at snapshot_date — same logic as API startup
    X_raw = build_inference_features(conn, snapshot_date)

    target_df = build_target(conn, snapshot_date)
    X_with_target = X_raw.merge(
        target_df[["uuid", "log_return_7d"]], on="uuid", how="inner"
    )
    y_full = X_with_target["log_return_7d"]
    drop_cols = ["log_return_7d"] + [
        c for c in LEAKAGE_COLS if c in X_with_target.columns
    ]
    X_full = X_with_target.drop(columns=drop_cols)

    valid = y_full.notna()
    X_full = X_full[valid].reset_index(drop=True)
    y_full = y_full[valid].reset_index(drop=True)

    if X_full.empty:
        raise RuntimeError(
            f"snapshot_date={snapshot_date!r} produced an empty training dataset. "
            "Ensure a t+7 counterpart exists in gold_price_features."
        )

    pipeline = build_feature_pipeline()
    X_t = pipeline.fit_transform(X_full)
    feature_names = get_feature_names(pipeline)
    X_df = pd.DataFrame(X_t, columns=feature_names)

    # Train final model — X_val=None triggers internal 80/20 temporal split
    final_model = LightGBMPriceModel()
    final_model.fit(X_df, y_full)

    # 3. Log to MLflow
    with start_run("auto_retrain", snapshot_date) as run:
        if not cv_results.empty:
            log_cv_results(cv_results)
        log_model(final_model, "model")
        run_id = str(run.info.run_id)

    logger.info("Retrain complete. New run_id: %s", run_id)

    # 4. Promote if new model is better than current Production
    _compare_and_promote(cv_results, run_id)
    return run_id


def _compare_and_promote(cv_results: pd.DataFrame, new_run_id: str) -> None:
    """Compare the new model against Production and promote if better.

    Uses CV MAPE on Tier 1 as the primary comparison metric (Tier 1 covers
    99.15% of the catalogue).  If no Production model exists yet, promotes
    unconditionally.  On any MLflow error, promotes conservatively to ensure
    the service always has a model.

    Args:
        cv_results:  DataFrame from :func:`walk_forward_cv` (may be empty when
                     insufficient data was available for CV).
        new_run_id:  MLflow run_id of the newly trained model.
    """
    import mlflow

    client = mlflow.tracking.MlflowClient()
    model_name = MODEL_REGISTRY_NAME

    # Compute new model's Tier 1 MAPE from CV results
    tier1_cv = (
        cv_results[cv_results["tier"] == 1] if not cv_results.empty else pd.DataFrame()
    )
    new_mape = float(tier1_cv["mape"].mean()) if not tier1_cv.empty else float("inf")

    try:
        # Look up current Production model by alias (MLflow 2.x+)
        prod_version = client.get_model_version_by_alias(model_name, "production")
        prod_run_id = prod_version.run_id
        if prod_run_id is None:
            logger.warning(
                "Production model version has no run_id — promoting new model."
            )
            promote_to_production(new_run_id, model_name)
            return
        prod_metrics = client.get_run(prod_run_id).data.metrics
        prod_mape = prod_metrics.get("cv_mape_tier1", float("inf"))

        if new_mape <= prod_mape:
            logger.info(
                "New model CV MAPE %.4f ≤ Production MAPE %.4f → promoting.",
                new_mape,
                prod_mape,
            )
            promote_to_production(new_run_id, model_name)
        else:
            logger.info(
                "New model CV MAPE %.4f > Production MAPE %.4f → keeping Production.",
                new_mape,
                prod_mape,
            )

    except mlflow.exceptions.MlflowException as exc:
        if getattr(exc, "error_code", None) == "RESOURCE_DOES_NOT_EXIST":
            logger.info(
                "No Production model found yet — promoting new model unconditionally."
            )
            promote_to_production(new_run_id, model_name)
        else:
            logger.error("Unexpected MLflow error during promotion comparison: %s", exc)
            raise


def promote_to_production(run_id: str, model_name: str = MODEL_REGISTRY_NAME) -> None:
    """Register a model run in MLflow Registry and set the ``production`` alias.

    Archives the previous Production version automatically.  Uses the MLflow
    2.x alias API (``set_registered_model_alias``) with a fallback to the
    deprecated stage-transition API for older MLflow versions.

    Args:
        run_id:     MLflow run_id whose ``model`` artefact to register.
        model_name: Registered model name in MLflow Registry.
                    Defaults to :data:`MODEL_REGISTRY_NAME`.
    """
    import mlflow

    client = mlflow.tracking.MlflowClient()

    result = mlflow.register_model(f"runs:/{run_id}/model", model_name)
    version = result.version

    try:
        # MLflow 2.x+ alias API
        client.set_registered_model_alias(model_name, "production", version)
        logger.info(
            "Model '%s' version %s registered as 'production' alias.",
            model_name,
            version,
        )
    except AttributeError:
        # Fallback: deprecated stage transition (MLflow < 2.0)
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Production",
            archive_existing_versions=True,
        )
        logger.info(
            "Model '%s' version %s transitioned to Production stage.",
            model_name,
            version,
        )
