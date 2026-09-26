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
def mlflow_tmp(tmp_path, monkeypatch):
    """Redirect all MLflow I/O — tracking *and* artifacts — into tmp_path.

    See the same fixture in tests/ml/training/test_tracking.py: the tracking
    URI alone leaves artifact_location relative, so logged models landed in the
    real ./mlruns until cwd was moved too.
    """
    from src.ml.training import tracking

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(tracking, "_PROJECT_ROOT", tmp_path)
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


def test_retrain_second_call_registers_but_does_not_repromote(tiny_gold_conn):
    """A repeat retrain with no CV metrics must not displace the incumbent.

    tiny_gold_conn is too small for walk-forward CV, so both calls hit
    retrain()'s InsufficientDataError fallback and cv_results is empty both
    times. The first call seeds an empty registry and is promoted on purpose
    (a model beats no model). The second one has an incumbent to regress
    against and no measurement to justify replacing it, so it stays a
    registered version without the alias.

    This used to promote on every call: an empty cv_results gave
    new_mape = inf, the incumbent's missing cv_mape_tier1 also defaulted to
    inf, and `inf <= inf` is True — so an untested model replaced production
    unconditionally, every time.
    """
    first_run_id = retrain(tiny_gold_conn, "2026-06-01")

    second_run_id = retrain(tiny_gold_conn, "2026-06-01")

    assert second_run_id != first_run_id
    client = mlflow.tracking.MlflowClient()
    prod_version = client.get_model_version_by_alias("mtg_price_model", "production")
    assert prod_version.run_id == first_run_id

    # The second run is fully logged and its model artefact is on disk, so an
    # operator can register and promote it by hand. It is not auto-registered:
    # promote_to_production() is what calls register_model(), so refusing to
    # promote also declines to create a registry version.
    assert mlflow.get_run(second_run_id).info.status == "FINISHED"
    registered_runs = {
        v.run_id for v in client.search_model_versions("name='mtg_price_model'")
    }
    assert registered_runs == {first_run_id}
