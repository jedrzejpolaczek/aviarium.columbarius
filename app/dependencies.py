"""FastAPI dependency callables for injecting shared application state.

Each function is used with ``Depends()`` in route handlers to inject the
appropriate resource from ``request.app.state``.

    get_db                — open DuckDB connection
    get_model             — loaded LightGBMPriceModel (or None before first train)
    get_similarity_index  — built CardSimilarityIndex (or None before first build)
    require_model         — loaded LightGBMPriceModel, or raises 503
    require_match         — rows of a DataFrame matching a column value, or raises 404
"""

from typing import cast

import duckdb
import pandas as pd
from fastapi import HTTPException, Request

from src.ml.models.lightgbm_model import LightGBMPriceModel
from src.ml.recommendation.similarity import CardSimilarityIndex


def get_db(request: Request) -> duckdb.DuckDBPyConnection:
    """Return the shared DuckDB connection from application state.

    Args:
        request: Incoming FastAPI request carrying ``app.state``.

    Returns:
        Open DuckDB connection set during application lifespan startup.
    """
    return request.app.state.db  # type: ignore[no-any-return]


def get_model(request: Request) -> LightGBMPriceModel | None:
    """Return the loaded price model from application state, or None.

    Args:
        request: Incoming FastAPI request carrying ``app.state``.

    Returns:
        Loaded LightGBMPriceModel, or None if no model has been trained yet.
    """
    return cast(LightGBMPriceModel | None, request.app.state.model)


def get_similarity_index(request: Request) -> CardSimilarityIndex | None:
    """Return the card similarity index from application state, or None.

    Args:
        request: Incoming FastAPI request carrying ``app.state``.

    Returns:
        Built CardSimilarityIndex, or None if the index has not been built yet.
    """
    return cast(CardSimilarityIndex | None, request.app.state.similarity_index)


def require_model(request: Request) -> LightGBMPriceModel:
    """Return the loaded model, or raise 503 if it isn't loaded.

    Use this instead of get_model in handlers that cannot proceed without a
    model — it removes the `if model is None: raise HTTPException(503, ...)`
    boilerplate repeated across predict.py and underpriced.py.
    """
    model = request.app.state.model
    if model is None:
        raise HTTPException(
            503, detail="Model not loaded. Set MODEL_RUN_ID env variable."
        )
    return cast(LightGBMPriceModel, model)


def require_match(
    df: pd.DataFrame, column: str, value: str, entity_name: str
) -> pd.DataFrame:
    """Return rows of df where df[column] == value, or raise 404 if none match."""
    matches = df[df[column] == value]
    if matches.empty:
        raise HTTPException(404, detail=f"{entity_name} '{value}' not found.")
    return matches
