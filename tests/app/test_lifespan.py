"""Integration tests for the real app.main.lifespan startup path.

Unlike tests/app/conftest.py's mock_lifespan fixtures (used by the other
app tests), these tests run the REAL lifespan function against a tiny,
real DuckDB file on disk, exercising:
    - the DuckDB connect + latest-snapshot query,
    - build_inference_features() (lag features + card features join),
    - the sklearn pipeline fit_transform(),
    - the MLflow model-load success/failure branches (degraded mode),
    - CardSimilarityIndex.fit(),
    - the shutdown DB-close step.

Fixture schema notes:
    gold_price_features only needs uuid, snapshot_date, eur, edhrec_rank,
    foil_premium — that is all src/ml/features/lag.py's build_lag_features()
    selects from it (see src/ml/features/sql/lag_features.sql).

    gold_card_features must NOT also define edhrec_rank/foil_premium: in the
    real Gold schema (src/data/cards/storage/gold/features.py) those two
    columns live only on gold_price_features. Duplicating them on
    gold_card_features (as an earlier fixture draft did) makes
    lag_df.merge(card_df, on="uuid") suffix them to edhrec_rank_x/_y and
    foil_premium_x/_y, which then makes build_feature_pipeline().fit_transform
    raise ValueError (IMPUTE_MEDIAN_COLS expects bare "edhrec_rank" /
    "foil_premium" column names). Matching the real Gold schema avoids the
    collision.

    SIMILARITY_FEATURES (src/ml/recommendation/similarity.py) requires
    rarity_ord (derived by _enrich_card_df from "rarity"), mana_value,
    color_count, color_identity_count, format_count, is_legendary,
    is_commander_legal, is_modern_legal — all included below.

GOLD_DB_PATH / MODEL_RUN_ID timing:
    app/main.py reads both as module-level constants at import time
    (``os.getenv(...)`` executed once, at module load). ``lifespan`` closes
    over those module globals, not over ``os.environ`` — so
    ``monkeypatch.setenv`` after import has no effect on it. The fix is to
    monkeypatch the module attributes directly:
    ``monkeypatch.setattr("app.main.GOLD_DB_PATH", ...)``.
"""

from collections.abc import Iterator
from pathlib import Path

import duckdb
import mlflow
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.main import lifespan


@pytest.fixture(autouse=True)
def isolated_mlflow_tracking(tmp_path: Path) -> Iterator[None]:
    """Point MLflow at a private, empty SQLite store for every test in this file.

    Prevents load_model_from_mlflow() from touching the real project-root
    mlflow.db (or trying to reach a remote tracking server) when a test
    exercises the "MODEL_RUN_ID set but load fails" branch.
    """
    db_path = tmp_path / "mlflow_isolated.db"
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    yield
    if mlflow.active_run():
        mlflow.end_run()


def _build_gold_db(db_path: Path, *, break_column: str | None = None) -> None:
    """Create a tiny, real Gold DuckDB file with the minimal valid schema.

    Args:
        db_path: Destination file path (must not already exist).
        break_column: If given, rename this column in gold_card_features to
            "_broken" to prove the test exercises real feature-building code
            (used only by the self-review / negative-control test).
    """
    con = duckdb.connect(str(db_path))
    con.execute("""
        CREATE TABLE gold_price_features (
            uuid VARCHAR, snapshot_date DATE, eur DOUBLE,
            edhrec_rank DOUBLE, foil_premium DOUBLE
        )
    """)
    con.execute("""
        INSERT INTO gold_price_features VALUES
        ('uuid-1', '2026-06-01', 1.5, 100.0, 1.1),
        ('uuid-2', '2026-06-01', 0.3, 200.0, 1.0)
    """)

    card_col = "mana_value" if break_column is None else break_column
    con.execute(f"""
        CREATE TABLE gold_card_features (
            uuid VARCHAR, name VARCHAR, rarity VARCHAR,
            print_count INTEGER, {card_col} DOUBLE, format_count INTEGER,
            is_reserved BOOLEAN, is_legendary BOOLEAN, is_commander_legal BOOLEAN,
            is_modern_legal BOOLEAN, color_count INTEGER, color_identity_count INTEGER,
            edhrec_saltiness DOUBLE
        )
    """)
    con.execute("""
        INSERT INTO gold_card_features VALUES
        ('uuid-1', 'Lightning Bolt', 'common', 4, 1.0, 3, false, false, true, true, 1, 1, 0.5),
        ('uuid-2', 'Dark Ritual',    'common', 2, 1.0, 1, false, false, true, true, 1, 1, 0.2)
    """)
    con.close()


@pytest.fixture
def tiny_gold_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "gold.duckdb"
    _build_gold_db(db_path)
    return db_path


def test_lifespan_populates_app_state_without_model(tiny_gold_db, monkeypatch):
    """Degraded mode (MODEL_RUN_ID=""): full real startup, no MLflow needed."""
    monkeypatch.setattr("app.main.GOLD_DB_PATH", str(tiny_gold_db))
    monkeypatch.setattr("app.main.MODEL_RUN_ID", "")

    app = FastAPI(lifespan=lifespan)
    with TestClient(app):
        state = app.state
        assert state.model is None
        assert state.model_run_id == ""
        assert state.snapshot_date == "2026-06-01"
        assert not state.X_all.empty
        assert {"uuid", "name", "eur", "log_eur", "rarity_ord"} <= set(
            state.X_all.columns
        )
        assert state.pipeline is not None
        assert not state.X_all_t.empty
        assert state.similarity_index is not None
        assert state.similarity_index.cards_df is not None
        assert set(state.similarity_index.cards_df["uuid"]) == {"uuid-1", "uuid-2"}


def test_lifespan_degrades_when_model_load_fails(tiny_gold_db, monkeypatch):
    """MODEL_RUN_ID set to a run that doesn't exist -> caught, degraded mode."""
    monkeypatch.setattr("app.main.GOLD_DB_PATH", str(tiny_gold_db))
    monkeypatch.setattr("app.main.MODEL_RUN_ID", "nonexistent-run-id")

    app = FastAPI(lifespan=lifespan)
    with TestClient(app):
        state = app.state
        assert state.model is None
        assert state.model_run_id == ""


def test_lifespan_closes_db_connection_on_shutdown(tiny_gold_db, monkeypatch):
    monkeypatch.setattr("app.main.GOLD_DB_PATH", str(tiny_gold_db))
    monkeypatch.setattr("app.main.MODEL_RUN_ID", "")

    app = FastAPI(lifespan=lifespan)
    with TestClient(app):
        db = app.state.db

    with pytest.raises(duckdb.ConnectionException):
        db.execute("SELECT 1")


def test_lifespan_raises_when_gold_price_features_empty(tmp_path, monkeypatch):
    """RuntimeError per the lifespan docstring when the ETL hasn't run yet."""
    db_path = tmp_path / "empty_gold.duckdb"
    con = duckdb.connect(str(db_path))
    con.execute(
        "CREATE TABLE gold_price_features "
        "(uuid VARCHAR, snapshot_date DATE, eur DOUBLE, "
        "edhrec_rank DOUBLE, foil_premium DOUBLE)"
    )
    con.close()

    monkeypatch.setattr("app.main.GOLD_DB_PATH", str(db_path))
    monkeypatch.setattr("app.main.MODEL_RUN_ID", "")

    app = FastAPI(lifespan=lifespan)
    with pytest.raises(RuntimeError, match="gold_price_features is empty"):
        with TestClient(app):
            pass
