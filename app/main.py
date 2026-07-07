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

import duckdb
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import cards, health, predict, similar, underpriced
from src.ml.features.pipeline import (
    build_feature_pipeline,
    build_inference_features,
    get_feature_names,
)
from src.ml.recommendation.similarity import SIMILARITY_FEATURES, CardSimilarityIndex
from src.logger import get_logger, setup_logging
from src.ml.training.tracking import load_model_from_mlflow


GOLD_DB_PATH = os.getenv("GOLD_DB_PATH", "data/gold/cards.duckdb")
MODEL_RUN_ID = os.getenv("MODEL_RUN_ID", "")

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup and shutdown context manager for the FastAPI application.

    Startup (before ``yield``):
        1. Open a read-only DuckDB connection and store it in ``app.state.db``.
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
    app.state.db = duckdb.connect(GOLD_DB_PATH, read_only=True)

    # 2. Latest snapshot available in the database
    _row = app.state.db.execute(
        "SELECT MAX(snapshot_date) FROM gold_price_features"
    ).fetchone()
    if _row is None or _row[0] is None:
        raise RuntimeError("gold_price_features is empty — run the ETL pipeline first.")
    snapshot_date = str(_row[0])
    app.state.snapshot_date = snapshot_date

    # 3. Build full feature matrix (lag + card features, log transforms, stub cols)
    X_all = build_inference_features(app.state.db, snapshot_date)
    app.state.X_all = X_all

    # 4. Fit sklearn pipeline once; pre-transform for O(1) inference per request
    pipeline = build_feature_pipeline()
    X_t = pipeline.fit_transform(X_all)
    feature_names = get_feature_names(pipeline)
    app.state.pipeline = pipeline
    app.state.feature_names = feature_names
    app.state.X_all_t = pd.DataFrame(
        np.array(X_t, dtype=np.float64), columns=feature_names
    )

    # 5. Load LightGBM booster from MLflow (optional — degraded mode on failure)
    if MODEL_RUN_ID:
        try:
            app.state.model = load_model_from_mlflow(MODEL_RUN_ID)
            app.state.model_run_id = MODEL_RUN_ID
            logger.info("Model loaded: %s", MODEL_RUN_ID)
        except Exception as exc:
            logger.warning("Model load failed (%s): %s", MODEL_RUN_ID, exc)
            app.state.model = None
            app.state.model_run_id = ""
    else:
        app.state.model = None
        app.state.model_run_id = ""
        logger.info("No MODEL_RUN_ID set — /predict and /underpriced will return 503.")

    # 6. Build CardSimilarityIndex with n_neighbors=50 (max returned by /similar)
    sim_df = X_all.copy()
    for feat in SIMILARITY_FEATURES:
        if feat not in sim_df.columns:
            sim_df[feat] = 0

    sim_index = CardSimilarityIndex(n_neighbors=50)
    sim_index.fit(sim_df)
    app.state.similarity_index = sim_index

    yield

    app.state.db.close()


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
