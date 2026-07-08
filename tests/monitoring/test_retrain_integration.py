"""Real-MLflow integration test for retrain() — the function has
`# pragma: no cover` today because every existing test mocks MLflow
entirely. This test uses a real sqlite-backed tracking store (same
pattern as tests/ml/training/test_tracking.py) and a deliberately tiny
DuckDB dataset that trips retrain()'s InsufficientDataError fallback,
so the full MLflow logging + promotion path runs for real without
needing 50+ days of synthetic price history for walk-forward CV.
"""

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


def test_retrain_logs_to_real_mlflow_and_promotes(tiny_gold_conn):
    run_id = retrain(tiny_gold_conn, "2026-06-01")

    run = mlflow.get_run(run_id)
    assert run.data.params.get("gold_snapshot_date") == "2026-06-01"

    client = mlflow.tracking.MlflowClient()
    prod_version = client.get_model_version_by_alias("mtg_price_model", "production")
    assert prod_version.run_id == run_id


def test_retrain_second_call_reregisters_and_repromotes(tiny_gold_conn):
    first_run_id = retrain(tiny_gold_conn, "2026-06-01")

    second_run_id = retrain(tiny_gold_conn, "2026-06-01")

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
