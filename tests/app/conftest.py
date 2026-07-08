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
from typing import TypedDict
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routers import health, predict, similar, underpriced


_LOG_RETURNS = np.array([0.5, -0.1, 0.0])


class _StateOverrides(TypedDict, total=False):
    """Overridable subset of ``app.state`` attributes for ``_build_test_app``.

    ``total=False`` since each fixture only overrides the 1-2 keys it cares
    about; the field list below is the authoritative set of overridable keys
    and their real types, so mypy flags typos and wrong-type values at each
    call site.
    """

    model: MagicMock | None
    repo: MagicMock
    X_all: pd.DataFrame
    X_all_t: pd.DataFrame
    model_run_id: str
    snapshot_date: str
    similarity_index: MagicMock | None


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


def _build_test_app(state_overrides: _StateOverrides) -> FastAPI:
    """Build a test FastAPI app with a mock lifespan populating ``app.state``.

    All ``app.state`` attributes are populated with deterministic test data so
    tests are fully isolated from the file system and ML infrastructure.
    ``state_overrides`` replaces individual base values for the app-specific
    test client variants (e.g. ``model=None`` for a 503 test).

    Args:
        state_overrides: Mapping of ``app.state`` attribute names to values
            that override the defaults for this particular test app. Keys
            must be a subset of ``_StateOverrides``'s fields.

    Returns:
        A configured but not-yet-started ``FastAPI`` instance.
    """
    base_state: _StateOverrides = {
        "model": _make_mock_model(),
        "repo": MagicMock(connection=MagicMock()),
        "X_all": _make_X_all(),
        "X_all_t": _make_X_all_t(),
        "model_run_id": "test-run-123",
        "snapshot_date": "2026-01-01",
        "similarity_index": _make_similarity_index(),
    }
    state: _StateOverrides = {**base_state, **state_overrides}

    @asynccontextmanager
    async def mock_lifespan(app: FastAPI) -> AsyncIterator[None]:
        for key, value in state.items():
            setattr(app.state, key, value)
        yield

    app = FastAPI(lifespan=mock_lifespan)
    for router in (health.router, predict.router, similar.router, underpriced.router):
        app.include_router(router)
    return app


@pytest.fixture(scope="module")
def test_client() -> Generator[TestClient, None, None]:
    """TestClient backed by a test app with mock lifespan.

    All ``app.state`` attributes are populated with deterministic test data so
    tests are fully isolated from the file system and ML infrastructure.

    Yields:
        Configured ``TestClient`` instance.
    """
    with TestClient(_build_test_app({})) as client:
        yield client


@pytest.fixture(scope="module")
def test_client_no_model() -> Generator[TestClient, None, None]:
    """TestClient where model=None to test 503 responses."""
    with TestClient(_build_test_app({"model": None, "model_run_id": ""})) as client:
        yield client


@pytest.fixture(scope="module")
def test_client_no_similarity() -> Generator[TestClient, None, None]:
    """TestClient where similarity_index=None to test 503 response from /similar."""
    with TestClient(_build_test_app({"similarity_index": None})) as client:
        yield client


@pytest.fixture(scope="module")
def test_client_similarity_error() -> Generator[TestClient, None, None]:
    """TestClient where find_similar raises ValueError to test 404 from /similar."""
    idx = _make_similarity_index()
    idx.find_similar.side_effect = ValueError("UUID not in index")
    with TestClient(_build_test_app({"similarity_index": idx})) as client:
        yield client
