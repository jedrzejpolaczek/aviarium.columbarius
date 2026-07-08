"""Single-card price prediction endpoint.

Predicts the log_return_7d for a named card and converts the result to an
absolute EUR price.  The response follows the three-tier pricing strategy:

- Tier 1 / 2 (< €1,000): returns ``predicted_price`` as a float.
- Tier 3 (> €1,000): returns ``predicted_price = null`` — too few training
  examples; callers should fall back to a direct Cardmarket lookup.

Feature matrix lookup:
    X_all and X_all_t are pre-built at startup (see ``app.main.lifespan``).
    The handler looks up the card's index in X_all by name and slices the
    corresponding pre-transformed row from X_all_t for O(1) inference.

URL encoding:
    Card names with spaces must be percent-encoded in the URL, e.g.
    ``GET /predict/Lightning%20Bolt``.
"""

import lightgbm as lgb
import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends

from app.dependencies import (
    RequestFeatures,
    get_request_features,
    require_match,
    require_model,
)
from app.pricing import inverse_log_return
from app.schemas.responses import PredictionResponse
from src.ml.models.tiered import assign_tier


def _predict_from_index(
    idx: int,
    card_name: str,
    X_all: pd.DataFrame,
    X_all_t: pd.DataFrame,
    model: object,
    model_run_id: str,
) -> PredictionResponse:
    """Run inference for one card identified by its row index in X_all/X_all_t."""
    eur_raw = X_all.at[idx, "eur"]
    current_eur = float(eur_raw) if pd.notna(eur_raw) else None  # type: ignore[arg-type]
    tier = assign_tier(current_eur if current_eur is not None else 0.0)

    if tier == 3:
        return PredictionResponse(
            card_name=card_name,
            current_price=current_eur,
            predicted_price=None,
            log_return_7d=None,
            tier=3,
            model_run_id=model_run_id,
        )

    log_return = float(model.predict(X_all_t.iloc[[idx]])[0])  # type: ignore[attr-defined]
    predicted_price = (
        float(inverse_log_return(np.array([current_eur]), np.array([log_return]))[0])
        if current_eur is not None
        else None
    )
    return PredictionResponse(
        card_name=card_name,
        current_price=current_eur,
        predicted_price=predicted_price,
        log_return_7d=log_return,
        tier=tier,
        model_run_id=model_run_id,
    )


router = APIRouter(prefix="/predict", tags=["prediction"])


@router.get("/uuid/{uuid}", response_model=PredictionResponse)
def predict_price_by_uuid(
    uuid: str,
    features: RequestFeatures = Depends(get_request_features),
    model: lgb.Booster = Depends(require_model),
) -> PredictionResponse:
    """Predict the EUR price of a card identified by its MTGJson UUID.

    Preferred over the name-based endpoint when a specific printing matters —
    UUID is unique per printing, so there is no ambiguity across editions.

    Args:
        uuid: MTGJson UUID of the specific card printing.
        features: Pre-computed feature matrices and model_run_id, injected
            via ``get_request_features`` dependency.
        model: LightGBM booster injected via ``require_model`` dependency.

    Returns:
        PredictionResponse identical in shape to the name-based endpoint.

    Raises:
        HTTPException 404: UUID not found in the feature matrix.
        HTTPException 503: Model not loaded.
    """
    X_all = features.X_all
    X_all_t = features.X_all_t
    model_run_id = features.model_run_id

    matches = require_match(X_all, "uuid", uuid, "UUID")

    idx = int(matches.index[0])
    card_name = str(X_all.at[idx, "name"])
    return _predict_from_index(idx, card_name, X_all, X_all_t, model, model_run_id)


@router.get("/{card_name}", response_model=PredictionResponse)
def predict_price(
    card_name: str,
    features: RequestFeatures = Depends(get_request_features),
    model: lgb.Booster = Depends(require_model),
) -> PredictionResponse:
    """Predict the EUR price of a single card seven days from now.

    Looks up the card by exact name in the pre-built feature matrix, runs the
    LightGBM booster, and converts the predicted log_return_7d back to an
    absolute EUR price using the inverse of ``log1p``.

    For cards with multiple printings (same name, different UUID), the first
    entry in the feature matrix is used — typically the most recent printing.

    Args:
        card_name: Exact card name, URL-decoded by FastAPI (e.g. "Force of Will").
        features:  Pre-computed feature matrices and model_run_id, injected
                   via ``get_request_features`` dependency.
        model:     LightGBM booster injected via ``require_model`` dependency.
                   Sourced from ``app.state.model``.

    Returns:
        PredictionResponse containing card_name, current_price, predicted_price,
        log_return_7d, tier, and model_run_id.

        ``predicted_price`` is ``None`` for Tier 3 cards (> €1,000).

    Raises:
        HTTPException 404: Card not found in the feature matrix.
        HTTPException 503: Model not loaded (MODEL_RUN_ID not set or load failed).
    """
    X_all = features.X_all
    X_all_t = features.X_all_t
    model_run_id = features.model_run_id

    matches = require_match(X_all, "name", card_name, "Card")

    idx = int(matches.index[0])
    return _predict_from_index(idx, card_name, X_all, X_all_t, model, model_run_id)
