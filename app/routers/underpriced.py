"""Underpriced-cards discovery endpoint.

Scans the entire card catalogue at the latest price snapshot, runs the
LightGBM booster on the pre-built feature matrix, converts predicted
log_return_7d values to absolute EUR prices, and delegates flagging logic to
``src.ml.recommendation.underpriced.flag_underpriced``.

Thresholds (from ADR-018 and underpriced.py):
    - Tier 1 (< €100): flagged when predicted/actual > 1.3
    - Tier 2 (€100–€1,000): same threshold; Bayesian guardrail (BA-02) pending
    - Tier 3 (> €1,000): never flagged — too few data points

Performance note:
    ``model.predict(X_all_t)`` runs once per request against the full
    pre-transformed matrix (~80k rows).  LightGBM batch prediction is fast
    enough for this without per-request caching.  Add DuckDB result caching
    (e.g. materialise to ``gold_price_predictions``) when latency becomes a
    concern.
"""

from datetime import date

import lightgbm as lgb
import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, Query

from app.dependencies import RequestFeatures, get_request_features, require_model
from app.routers.predict import inverse_log_return
from app.schemas.responses import UnderpricedCard, UnderpricedResponse
from src.ml.recommendation.underpriced import flag_underpriced


router = APIRouter(prefix="/underpriced", tags=["recommendation"])


def _run_underpriced_inference(
    X_all: pd.DataFrame,
    X_all_t: pd.DataFrame,
    model: lgb.Booster,
    tier: int | None,
    min_confidence: float,
) -> pd.DataFrame:
    log_returns: np.ndarray = np.asarray(model.predict(X_all_t))
    eur = X_all["eur"].to_numpy()
    predicted_eur = inverse_log_return(eur, log_returns)

    result_df = pd.DataFrame(
        {
            "uuid": X_all["uuid"].values,
            "name": X_all["name"].values,
            "eur": eur,
            "predicted_eur": predicted_eur,
        }
    )
    result_df = result_df[result_df["eur"].notna()].copy()

    flagged = flag_underpriced(result_df)
    flagged = flagged[flagged["is_underpriced"]]
    if tier is not None:
        flagged = flagged[flagged["tier"] == tier]
    return flagged[flagged["confidence"] >= min_confidence]


def _to_underpriced_cards(flagged: pd.DataFrame) -> list[UnderpricedCard]:
    return [
        UnderpricedCard(
            name=str(row["name"]),
            uuid=str(row["uuid"]),
            actual_price=float(row["eur"]),
            predicted_price=float(row["predicted_eur"]),
            confidence=float(row["confidence"]),
            tier=int(row["tier"]),
            reason=str(row["reason"]),
        )
        for _, row in flagged.iterrows()
    ]


@router.get("/", response_model=UnderpricedResponse)
def get_underpriced_cards(
    tier: int | None = Query(
        default=None, ge=1, le=3, description="Filter by tier (1, 2, or 3)"
    ),
    min_confidence: float = Query(
        default=1.3, ge=1.0, description="Minimum predicted/actual ratio"
    ),
    features: RequestFeatures = Depends(get_request_features),
    model: lgb.Booster = Depends(require_model),
) -> UnderpricedResponse:
    """Return all cards the model considers underpriced at the latest snapshot.

    Runs batch inference over the pre-built feature matrix, computes
    ``predicted_eur = expm1(log1p(eur) + log_return_7d)`` for each card,
    then calls ``flag_underpriced`` to apply tier-based thresholds.

    Results are sorted by confidence (predicted/actual ratio) descending so
    the most underpriced cards appear first.

    Args:
        tier:           Optional tier filter (1, 2, or 3). Returns all tiers
                        when not specified.
        min_confidence: Minimum predicted/actual ratio to include in results.
                        Default 1.3 matches the ``TIER1_FLAG_THRESHOLD``
                        constant in ``flag_underpriced``.
        features:       Pre-computed feature matrices and model_run_id,
                        injected via ``get_request_features`` dependency.
        model:          LightGBM booster injected via ``require_model`` dependency.

    Returns:
        UnderpricedResponse containing a list of UnderpricedCard objects, the
        generation date, and the MLflow run_id of the model used.

    Raises:
        HTTPException 503: Model not loaded (MODEL_RUN_ID not set or load failed).
    """
    X_all = features.X_all
    X_all_t = features.X_all_t
    model_run_id = features.model_run_id

    flagged = _run_underpriced_inference(X_all, X_all_t, model, tier, min_confidence)
    cards = _to_underpriced_cards(flagged)

    return UnderpricedResponse(
        cards=cards,
        generated_at=date.today(),
        model_run_id=model_run_id,
    )
