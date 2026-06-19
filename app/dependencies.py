"""FastAPI dependency callables for injecting shared application state.

Each function is used with ``Depends()`` in route handlers to inject the
appropriate resource from ``request.app.state``.

    get_db                — open DuckDB connection
    get_model             — loaded LightGBMPriceModel (or None before first train)
    get_similarity_index  — built CardSimilarityIndex (or None before first build)
"""

from typing import cast

import duckdb
from fastapi import Request

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
