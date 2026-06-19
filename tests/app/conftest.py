"""Shared fixtures for FastAPI endpoint tests.

Uses a test-specific FastAPI app with a mock lifespan instead of the
production lifespan from ``app.main``.  This avoids the need for a real
DuckDB database file or an MLflow server during unit tests.

Test data layout (3 cards):
    - "Lightning Bolt"  eur=1.5   (Tier 1)
    - "Dark Ritual"     eur=0.3   (Tier 1)
    - "Black Lotus"     eur=1500.0 (Tier 3)

Mock model always returns log_returns [0.5, -0.1, 0.0] regardless of input:
    - Lightning Bolt:  predicted_eur ≈ 3.12  confidence ≈ 2.08 → underpriced
    - Dark Ritual:     predicted_eur ≈ 0.27  confidence ≈ 0.90 → not underpriced
    - Black Lotus:     Tier 3 → never flagged
"""

from collections.abc import AsyncIterator, Generator
from contextlib import asynccontextmanager
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routers import health, predict, similar, underpriced


_LOG_RETURNS = np.array([0.5, -0.1, 0.0])


def _make_X_all() -> pd.DataFrame:
    """Minimal feature matrix covering all three pricing tiers."""
    return pd.DataFrame(
        {
            "name": ["Lightning Bolt", "Dark Ritual", "Black Lotus"],
            "uuid": ["uuid_lb", "uuid_dr", "uuid_bl"],
            "eur": [1.5, 0.3, 1500.0],
        }
    )


def _make_X_all_t() -> pd.DataFrame:
    """Minimal pre-transformed matrix — contents ignored by the mock model."""
    return pd.DataFrame({"f1": [1.0, 2.0, 3.0], "f2": [0.1, 0.2, 0.3]})


def _make_similarity_index() -> MagicMock:
    """Mock CardSimilarityIndex that returns two cards for any query."""
    idx = MagicMock()
    idx.cards_df = pd.DataFrame(
        {
            "name": ["Lightning Bolt", "Dark Ritual", "Black Lotus"],
            "uuid": ["uuid_lb", "uuid_dr", "uuid_bl"],
            "eur": [1.5, 0.3, 1500.0],
        }
    )
    idx.find_similar.return_value = pd.DataFrame(
        {
            "name": ["Dark Ritual", "Black Lotus"],
            "uuid": ["uuid_dr", "uuid_bl"],
            "eur": [0.3, 1500.0],
            "similarity_score": [0.95, 0.80],
        }
    )
    return idx


def _make_mock_model() -> MagicMock:
    """Mock lgb.Booster that returns a fixed log_returns array."""
    model = MagicMock()
    model.predict.return_value = _LOG_RETURNS
    return model


@pytest.fixture(scope="module")
def test_client() -> Generator[TestClient, None, None]:
    """TestClient backed by a test app with mock lifespan.

    All ``app.state`` attributes are populated with deterministic test data so
    tests are fully isolated from the file system and ML infrastructure.

    Yields:
        Configured ``TestClient`` instance.
    """
    mock_model = _make_mock_model()
    mock_sim_index = _make_similarity_index()

    @asynccontextmanager
    async def mock_lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.model = mock_model
        app.state.db = MagicMock()
        app.state.X_all = _make_X_all()
        app.state.X_all_t = _make_X_all_t()
        app.state.model_run_id = "test-run-123"
        app.state.snapshot_date = "2026-01-01"
        app.state.similarity_index = mock_sim_index
        yield

    _app = FastAPI(lifespan=mock_lifespan)
    _app.include_router(health.router)
    _app.include_router(predict.router)
    _app.include_router(similar.router)
    _app.include_router(underpriced.router)

    with TestClient(_app) as client:
        yield client


@pytest.fixture(scope="module")
def test_client_no_model() -> Generator[TestClient, None, None]:
    """TestClient where model=None to test 503 responses."""

    @asynccontextmanager
    async def mock_lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.model = None
        app.state.db = MagicMock()
        app.state.X_all = _make_X_all()
        app.state.X_all_t = _make_X_all_t()
        app.state.model_run_id = ""
        app.state.snapshot_date = "2026-01-01"
        app.state.similarity_index = _make_similarity_index()
        yield

    _app = FastAPI(lifespan=mock_lifespan)
    _app.include_router(health.router)
    _app.include_router(predict.router)
    _app.include_router(similar.router)
    _app.include_router(underpriced.router)

    with TestClient(_app) as client:
        yield client


@pytest.fixture(scope="module")
def test_client_no_similarity() -> Generator[TestClient, None, None]:
    """TestClient where similarity_index=None to test 503 response from /similar."""

    @asynccontextmanager
    async def mock_lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.model = _make_mock_model()
        app.state.db = MagicMock()
        app.state.X_all = _make_X_all()
        app.state.X_all_t = _make_X_all_t()
        app.state.model_run_id = "test-run-123"
        app.state.snapshot_date = "2026-01-01"
        app.state.similarity_index = None
        yield

    _app_no_sim = FastAPI(lifespan=mock_lifespan)
    _app_no_sim.include_router(health.router)
    _app_no_sim.include_router(predict.router)
    _app_no_sim.include_router(similar.router)
    _app_no_sim.include_router(underpriced.router)

    with TestClient(_app_no_sim) as client:
        yield client


@pytest.fixture(scope="module")
def test_client_similarity_error() -> Generator[TestClient, None, None]:
    """TestClient where find_similar raises ValueError to test 404 from /similar."""

    @asynccontextmanager
    async def mock_lifespan(app: FastAPI) -> AsyncIterator[None]:
        idx = _make_similarity_index()
        idx.find_similar.side_effect = ValueError("UUID not in index")
        app.state.model = _make_mock_model()
        app.state.db = MagicMock()
        app.state.X_all = _make_X_all()
        app.state.X_all_t = _make_X_all_t()
        app.state.model_run_id = "test-run-123"
        app.state.snapshot_date = "2026-01-01"
        app.state.similarity_index = idx
        yield

    _app_sim_err = FastAPI(lifespan=mock_lifespan)
    _app_sim_err.include_router(health.router)
    _app_sim_err.include_router(predict.router)
    _app_sim_err.include_router(similar.router)
    _app_sim_err.include_router(underpriced.router)

    with TestClient(_app_sim_err) as client:
        yield client
