"""Real-MLflow integration test for retrain() — the function has
`# pragma: no cover` today because every existing test mocks MLflow
entirely. This test uses a real sqlite-backed tracking store (same
pattern as tests/ml/training/test_tracking.py) and a deliberately tiny
DuckDB dataset that trips retrain()'s InsufficientDataError fallback,
so the full MLflow logging + promotion path runs for real without
needing 50+ days of synthetic price history for walk-forward CV.
"""

import duckdb
import mlflow
import pytest

from src.monitoring.retraining import retrain


@pytest.fixture(autouse=True)
def mlflow_tmp(tmp_path):
    """Redirect all MLflow I/O to an isolated SQLite database for this test."""
    db_path = tmp_path / "mlflow.db"
    uri = f"sqlite:///{db_path}"
    mlflow.set_tracking_uri(uri)
    yield uri
    if mlflow.active_run():
        mlflow.end_run()


@pytest.fixture
def tiny_conn():
    con = duckdb.connect(":memory:")
    con.execute("""
        CREATE TABLE gold_price_features AS
        SELECT * FROM (VALUES
            ('uuid-1', '2026-06-01', 1.5, 100.0, NULL),
            ('uuid-1', '2026-06-08', 1.8, 100.0, NULL),
            ('uuid-2', '2026-06-01', 0.3, 200.0, NULL),
            ('uuid-2', '2026-06-08', 0.4, 200.0, NULL)
        ) AS t(uuid, snapshot_date, eur, edhrec_rank, foil_premium)
    """)
    # edhrec_saltiness is required here (not in gold_price_features) because
    # IMPUTE_MEDIAN_COLS in src/ml/features/pipeline.py expects it, and in
    # production it is sourced from gold_card_features (see
    # GoldFeatureBuilders.build_card_features in
    # src/data/cards/storage/gold/features.py) — build_inference_features()
    # merges lag_df and card_df on uuid, so it must be present post-merge.
    con.execute("""
        CREATE TABLE gold_card_features AS
        SELECT * FROM (VALUES
            ('uuid-1', 'common', 3, 2.0, 1, false, false, true, NULL),
            ('uuid-2', 'rare',   1, 1.0, 1, false, false, true, NULL)
        ) AS t(uuid, rarity, print_count, mana_value, format_count,
                is_reserved, is_legendary, is_commander_legal, edhrec_saltiness)
    """)
    yield con
    con.close()


def test_retrain_logs_to_real_mlflow_and_promotes(tiny_conn):
    run_id = retrain(tiny_conn, "2026-06-01")

    run = mlflow.get_run(run_id)
    assert run.data.params.get("gold_snapshot_date") == "2026-06-01"

    client = mlflow.tracking.MlflowClient()
    prod_version = client.get_model_version_by_alias("mtg_price_model", "production")
    assert prod_version.run_id == run_id


def test_retrain_second_call_reregisters_and_repromotes(tiny_conn):
    first_run_id = retrain(tiny_conn, "2026-06-01")

    second_run_id = retrain(tiny_conn, "2026-06-01")

    client = mlflow.tracking.MlflowClient()
    prod_version = client.get_model_version_by_alias("mtg_price_model", "production")
    # tiny_conn's dataset is too small for walk-forward CV, so both calls hit
    # retrain()'s InsufficientDataError fallback and cv_results is empty both
    # times, making new_mape == float("inf") on every call. That means
    # `new_mape <= prod_mape` is `inf <= inf`, which is deterministically
    # True — the second call promotes every time, it is not a coin-flip tie.
    # This test therefore does NOT exercise the MAPE-comparison branch (that
    # is unit-tested with mocks in tests/monitoring/test_retraining.py); it
    # verifies that a repeat retrain() call re-registers a new model version
    # and correctly re-points the "production" alias at it.
    assert prod_version.run_id == second_run_id
