"""Administrative endpoints for operational control.

Currently one endpoint: hot model reload. Protected by a shared-secret
header (``X-Admin-Token``) checked against the ``ADMIN_TOKEN`` environment
variable — disabled entirely (503) if that variable isn't set, so hot
reload is opt-in per deployment rather than silently accepting any token
when nobody configured one.
"""

import hmac
import os

from fastapi import APIRouter, Header, HTTPException, Request
from pydantic import BaseModel

from src.logger import get_logger
from src.ml.training.tracking import load_model_from_mlflow


logger = get_logger(__name__)

router = APIRouter(prefix="/admin", tags=["admin"])


class ReloadModelRequest(BaseModel):
    model_run_id: str


def _require_admin_token(x_admin_token: str | None) -> None:
    """Raise 503 if ADMIN_TOKEN isn't configured, 403 if it doesn't match.

    Uses ``hmac.compare_digest`` (constant-time) rather than ``!=`` since
    this compares a caller-supplied secret against the real token.
    """
    expected = os.getenv("ADMIN_TOKEN", "")
    if not expected:
        raise HTTPException(
            503, detail="ADMIN_TOKEN not configured — hot reload is disabled."
        )
    if not hmac.compare_digest(x_admin_token or "", expected):
        logger.warning("Rejected admin request: invalid X-Admin-Token.")
        raise HTTPException(403, detail="Invalid admin token.")


@router.post("/reload-model")
def reload_model(
    request: Request,
    body: ReloadModelRequest,
    x_admin_token: str | None = Header(default=None),
) -> dict[str, str]:
    """Re-load the LightGBM model from MLflow by run_id, in-place.

    Lets an operator make ``scripts/rollback_model.py``'s registry-alias
    swap (or a fresh retrain's new run_id) take effect immediately, without
    editing ``docker/.env`` and restarting the container.

    Args:
        request: Incoming FastAPI request carrying ``app.state``.
        body: JSON body with the target ``model_run_id``.
        x_admin_token: Shared-secret header, checked against ADMIN_TOKEN.

    Returns:
        ``{"status": "reloaded", "model_run_id": ...}`` on success.

    Raises:
        HTTPException: 503 if ADMIN_TOKEN isn't configured, 403 if the
            token doesn't match, 502 if the MLflow load itself fails
            (invalid run_id, registry unreachable, etc.) — app.state is
            left unchanged in that case.
    """
    _require_admin_token(x_admin_token)
    try:
        model = load_model_from_mlflow(body.model_run_id)
    except Exception as exc:
        logger.error("Hot reload failed for run_id=%s: %s", body.model_run_id, exc)
        raise HTTPException(
            502, detail=f"Failed to load model {body.model_run_id}: {exc}"
        ) from exc

    request.app.state.model = model
    request.app.state.model_run_id = body.model_run_id
    logger.info("Model hot-reloaded to run_id=%s", body.model_run_id)
    return {"status": "reloaded", "model_run_id": body.model_run_id}
