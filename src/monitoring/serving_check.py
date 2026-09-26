"""Consistency check between the model being served and the registry alias.

Two independent pointers decide which model matters, and nothing compared them:

- ``MODEL_RUN_ID`` in ``docker/.env`` is what the API actually loads at startup
  (see ``app/main.py``).
- the ``production`` alias in the MLflow Registry is what
  ``_compare_and_promote`` measures new candidates against, and what
  ``scripts/rollback_model.py`` moves.

On 2026-09-26 they disagreed: the API was serving run
``9c1ec7de65c7476aa75a199b7189d6f2`` (registry version 3) while the
``production`` alias pointed at version 2, run ``c46d4787…``. Every automatic
promotion decision was therefore being made against a model nobody was
serving, and a rollback would have "rolled back" to something already live.

This module only reports. Moving the alias is an operator action —
``uv run python -m scripts.rollback_model --version <n>`` — because which of
the two pointers is right is a judgement call about what should be live, not
something a monitoring check should decide on its own.
"""

import os

from src.logger import get_logger
from src.monitoring.retraining import MODEL_REGISTRY_NAME


logger = get_logger(__name__)


def check_serving_matches_production_alias(
    serving_run_id: str,
    model_name: str = MODEL_REGISTRY_NAME,
) -> tuple[bool, str]:
    """Compare the served run against the registry's ``production`` alias.

    Args:
        serving_run_id: The run_id the API is configured to load, i.e. the
                        value of ``MODEL_RUN_ID``. An empty string means the
                        API is running in degraded mode with no model.
        model_name:     Registered model name. Defaults to
                        :data:`MODEL_REGISTRY_NAME`.

    Returns:
        ``(ok, detail)``. ``ok`` is False only for a genuine disagreement
        between two known pointers. Anything unknowable — no MODEL_RUN_ID
        configured, no alias set yet, MLflow unreachable — returns True with a
        detail explaining why the check was skipped, so a monitoring job never
        pages anyone over a check it could not perform.
    """
    import mlflow  # deferred: keeps importing this module cheap for callers
    # that only need the other monitoring helpers (same pattern as retraining).

    if not serving_run_id:
        return True, "MODEL_RUN_ID is unset — nothing is being served, skipped"

    try:
        prod_version = mlflow.tracking.MlflowClient().get_model_version_by_alias(
            model_name, "production"
        )
    except mlflow.exceptions.MlflowException as exc:
        if getattr(exc, "error_code", None) == "RESOURCE_DOES_NOT_EXIST":
            return True, f"no 'production' alias on {model_name!r} yet — skipped"
        return True, f"could not read the registry ({exc}) — skipped"

    if prod_version.run_id == serving_run_id:
        return True, (
            f"served run matches 'production' alias "
            f"(version {prod_version.version}, run {serving_run_id})"
        )

    return False, (
        f"MODEL_RUN_ID={serving_run_id} is being served, but the 'production' "
        f"alias on {model_name!r} points at version {prod_version.version} "
        f"(run {prod_version.run_id}). Promotion comparisons and rollbacks use "
        f"the alias, not the served model. Align them with "
        f"`uv run python -m scripts.rollback_model --version <n>`, or update "
        f"MODEL_RUN_ID in docker/.env and POST /admin/reload-model."
    )


def log_serving_alias_check(model_name: str = MODEL_REGISTRY_NAME) -> tuple[bool, str]:
    """Run the check against ``MODEL_RUN_ID`` from the environment and log it."""
    ok, detail = check_serving_matches_production_alias(
        os.getenv("MODEL_RUN_ID", ""), model_name
    )
    if ok:
        logger.info("Serving/registry check: %s", detail)
    else:
        logger.error("Serving/registry mismatch: %s", detail)
    return ok, detail
