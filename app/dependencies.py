"""FastAPI dependency callables for injecting shared application state.

Each function is used with ``Depends()`` in route handlers to inject the
appropriate resource from ``request.app.state``.

    get_db                — DuckDBRepository wrapping the shared connection
    get_similarity_index  — built CardSimilarityIndex (or None before first build)
    require_model         — loaded lgb.Booster, or raises 503
    get_request_features  — RequestFeatures bundling X_all/X_all_t/model_run_id

``require_match`` is also defined here but is not a ``Depends()`` target — see
its own docstring for why.
"""

from dataclasses import dataclass
from typing import cast

import lightgbm as lgb
import pandas as pd
from fastapi import HTTPException, Request

from src.data.repository import DuckDBRepository
from src.ml.recommendation.similarity import CardSimilarityIndex


def get_db(request: Request) -> DuckDBRepository:
    """Return the shared DuckDB repository from application state.

    Args:
        request: Incoming FastAPI request carrying ``app.state``.

    Returns:
        DuckDBRepository set during application lifespan startup
        (app/main.py's ``_connect_db``).
    """
    return request.app.state.repo  # type: ignore[no-any-return]


def get_similarity_index(request: Request) -> CardSimilarityIndex | None:
    """Return the card similarity index from application state, or None.

    Args:
        request: Incoming FastAPI request carrying ``app.state``.

    Returns:
        Built CardSimilarityIndex, or None if the index has not been built yet.
    """
    return cast(CardSimilarityIndex | None, request.app.state.similarity_index)


def require_model(request: Request) -> lgb.Booster:
    """Return the loaded model from application state, or raise 503.

    Use this (instead of reading ``request.app.state.model`` directly) in
    handlers that cannot proceed without a model — it removes the
    `if model is None: raise HTTPException(503, ...)` boilerplate repeated
    across predict.py and underpriced.py.

    Args:
        request: Incoming FastAPI request carrying ``app.state``.

    Returns:
        Loaded lgb.Booster (see ``load_model_from_mlflow`` in
        ``src.ml.training.tracking``, which is what actually populates
        ``app.state.model`` at startup).

    Raises:
        HTTPException: 503 if no model has been trained/loaded yet.
    """
    model = request.app.state.model
    if model is None:
        raise HTTPException(
            503, detail="Model not loaded. Set MODEL_RUN_ID env variable."
        )
    return cast(lgb.Booster, model)


@dataclass
class RequestFeatures:
    """Bundles the three pieces of app.state every predict/underpriced/cards
    handler reads — replaces the 3-line
    `X_all = request.app.state.X_all; X_all_t = ...; model_run_id = ...`
    block repeated across cards.py, predict.py (x2), and underpriced.py.
    """

    X_all: pd.DataFrame
    X_all_t: pd.DataFrame
    model_run_id: str


def get_request_features(request: Request) -> RequestFeatures:
    """Return the pre-computed feature matrices and active model_run_id.

    app.main's lifespan always sets X_all/X_all_t/model_run_id together at
    startup, so any handler using this dependency implicitly requires all
    three to be present on app.state — including cards.py, which only reads
    X_all itself. A hand-built test app that sets X_all without the other
    two will fail here with AttributeError rather than at the call site.

    model_run_id defaults to "" via getattr because it's legitimately empty
    in degraded mode (no MODEL_RUN_ID set / model load failed, see
    app.main._load_model_or_degrade); X_all/X_all_t have no such degraded
    state, so a missing attribute there is a real bug, not a normal mode.
    """
    return RequestFeatures(
        X_all=request.app.state.X_all,
        X_all_t=request.app.state.X_all_t,
        model_run_id=getattr(request.app.state, "model_run_id", ""),
    )


def require_match(
    df: pd.DataFrame, column: str, value: str, entity_name: str
) -> pd.DataFrame:
    """Return rows of df where df[column] == value, or raise 404 if none match.

    Unlike ``require_model``, this is a plain function rather than a
    ``Depends()`` target: FastAPI dependencies are resolved once per request
    from request-scoped state (e.g. ``app.state.model``), but the DataFrame,
    column, and lookup value here differ per call site within the same
    handler (uuid vs. name lookups, different entity names for the error
    message), which doesn't fit the no-argument ``Depends()`` shape. Call it
    directly from inside a handler body instead. If a future guard needs
    per-request singleton state, prefer the ``require_model``/``Depends()``
    pattern; if it needs per-call arguments, prefer this pattern.

    Args:
        df: DataFrame to filter.
        column: Name of the column to match against.
        value: Value to look up in ``column``.
        entity_name: Human-readable entity name used in the 404 detail
            message (e.g. "UUID", "Card").

    Returns:
        The subset of df where df[column] == value. Guaranteed non-empty.

    Raises:
        HTTPException: 404 if no rows match.
    """
    matches = df[df[column] == value]
    if matches.empty:
        raise HTTPException(404, detail=f"{entity_name} '{value}' not found.")
    return matches
