"""FastAPI application entry point — wires startup, shutdown, and HTTP routing.

Lifespan pattern (FastAPI 0.95+):
    All expensive one-time setup (DuckDB connection, pipeline fitting, model
    load, similarity index construction) happens inside ``lifespan`` before
    ``yield``. Every subsequent request reads pre-built state from
    ``app.state`` — zero latency per request for IO or heavy computation.

Pre-computation strategy:
    Feature matrix X_all (raw) and X_all_t (pipeline-transformed) are built
    once at startup for the latest price snapshot. This amortises the cost of
    DuckDB queries and sklearn transformations across all requests.

Degraded mode:
    The server starts and responds to /health and /similar even if MODEL_RUN_ID
    is unset or the model fails to load. Only /predict and /underpriced return
    503 in that case.

Environment variables:
    GOLD_DB_PATH  -- Path to the DuckDB database file.
                     Default: ``data/gold/cards.duckdb``
    MODEL_RUN_ID  -- MLflow run ID of the LightGBM model to load.
                     Leave empty to start without prediction capability.

Quick start:
    uv run uvicorn app.main:app --reload --port 8000
    http://localhost:8000/docs   (Swagger UI)
    http://localhost:8000/health (health check)
"""

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import NamedTuple

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sklearn.pipeline import Pipeline

from app.routers import cards, health, predict, similar, underpriced
from src.data.cards.storage.gold.storage import get_latest_gold_snapshot_date
from src.data.repository import GOLD_DB_PATH, DuckDBRepository, open_repository
from src.ml.features.pipeline import (
    build_feature_pipeline,
    build_inference_features,
    get_feature_names,
)
from src.ml.recommendation.similarity import SIMILARITY_FEATURES, CardSimilarityIndex
from src.logger import get_logger, setup_logging
from src.ml.training.tracking import load_model_from_mlflow


MODEL_RUN_ID = os.getenv("MODEL_RUN_ID", "")

logger = get_logger(__name__)


class FeatureMatrices(NamedTuple):
    """Result of :func:`_build_feature_matrices` — one field per pipeline artifact."""

    X_all: pd.DataFrame
    X_all_t: pd.DataFrame
    pipeline: Pipeline
    feature_names: list[str]


def _connect_db() -> DuckDBRepository:
    """Open a read-only DuckDB repository (API never writes)."""
    return open_repository(GOLD_DB_PATH, read_only=True)


def _build_feature_matrices(
    db: duckdb.DuckDBPyConnection, snapshot_date: str
) -> FeatureMatrices:
    """Build the full feature matrix and fit the sklearn pipeline once.

    Returns the raw feature matrix ``X_all``, the pipeline-transformed matrix
    ``X_all_t`` (pre-computed for O(1) per-request inference), the fitted
    pipeline, and the resulting feature names.
    """
    X_all = build_inference_features(db, snapshot_date)
    pipeline = build_feature_pipeline()
    X_t = pipeline.fit_transform(X_all)
    feature_names = get_feature_names(pipeline)
    X_all_t = pd.DataFrame(np.array(X_t, dtype=np.float64), columns=feature_names)
    return FeatureMatrices(X_all, X_all_t, pipeline, feature_names)


def _load_model_or_degrade(model_run_id: str) -> tuple[lgb.Booster | None, str]:
    """Load the LightGBM booster from MLflow, or start in degraded mode.

    Degraded mode (model=None) is entered if ``model_run_id`` is empty or the
    MLflow load raises — /predict and /underpriced return 503 in that case.
    """
    if not model_run_id:
        logger.info("No MODEL_RUN_ID set — /predict and /underpriced will return 503.")
        return None, ""
    try:
        model = load_model_from_mlflow(model_run_id)
        logger.info("Model loaded: %s", model_run_id)
        return model, model_run_id
    except Exception as exc:
        logger.warning("Model load failed (%s): %s", model_run_id, exc)
        return None, ""


def _build_similarity_index(X_all: pd.DataFrame) -> CardSimilarityIndex:
    """Build a CardSimilarityIndex with n_neighbors=50 (max returned by /similar)."""
    sim_df = X_all.copy()
    for feat in SIMILARITY_FEATURES:
        if feat not in sim_df.columns:
            sim_df[feat] = 0

    sim_index = CardSimilarityIndex(n_neighbors=50)
    sim_index.fit(sim_df)
    return sim_index


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup and shutdown context manager for the FastAPI application.

    Startup (before ``yield``):
        1. Open a read-only DuckDB repository and store it in ``app.state.repo``.
        2. Determine the latest available price snapshot date.
        3. Build the full feature matrix ``X_all`` (raw) for all cards at that
           snapshot by joining lag features with static card attributes.
        4. Fit the sklearn pipeline once; store the pre-transformed matrix
           ``X_all_t`` for O(1) per-request inference.
        5. Load the LightGBM booster from MLflow if ``MODEL_RUN_ID`` is set.
           On failure the server starts in degraded mode (model=None).
        6. Build a ``CardSimilarityIndex`` with n_neighbors=50, enabling the
           /similar endpoint to return up to 50 results without re-fitting.

    Shutdown (after ``yield``):
        Closes the DuckDB connection.

    Raises:
        RuntimeError: If ``gold_price_features`` is empty (ETL not yet run).
    """
    setup_logging(logging.INFO)

    # 1. Connect DuckDB (read-only — API never writes)
    app.state.repo = _connect_db()

    # 2. Latest snapshot available in the database
    snapshot_date = get_latest_gold_snapshot_date(app.state.repo.connection)
    if snapshot_date is None:
        raise RuntimeError("gold_price_features is empty — run the ETL pipeline first.")
    app.state.snapshot_date = snapshot_date

    # 3-4. Build full feature matrix and fit sklearn pipeline once
    features = _build_feature_matrices(app.state.repo.connection, snapshot_date)
    app.state.X_all = features.X_all
    app.state.X_all_t = features.X_all_t
    app.state.pipeline = features.pipeline
    app.state.feature_names = features.feature_names

    # 5. Load LightGBM booster from MLflow (optional — degraded mode on failure)
    app.state.model, app.state.model_run_id = _load_model_or_degrade(MODEL_RUN_ID)

    # 6. Build CardSimilarityIndex with n_neighbors=50 (max returned by /similar)
    app.state.similarity_index = _build_similarity_index(features.X_all)

    yield

    app.state.repo.close()


app = FastAPI(
    title="MTG Price Prediction API",
    description="Predykcja cen kart Magic: The Gathering",
    version="0.1.0",
    lifespan=lifespan,
)

_cors_origins = [
    o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["GET"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(cards.router)
app.include_router(predict.router)
app.include_router(similar.router)
app.include_router(underpriced.router)
