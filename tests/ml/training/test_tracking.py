import numpy as np
import pandas as pd
import pytest

import mlflow


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def mlflow_tmp(tmp_path):
    """Redirect all MLflow I/O to an isolated SQLite database for each test.

    mlflow >= 3.13 dropped the file-based tracking store. sqlite:// is the
    lightweight local alternative that still works without a running server.

    autouse=True means every test in this file gets its own isolated store
    without having to request the fixture explicitly.
    """
    db_path = tmp_path / "mlflow.db"
    uri = f"sqlite:///{db_path}"
    mlflow.set_tracking_uri(uri)
    yield uri
    # End any accidentally open run so later tests start clean.
    if mlflow.active_run():
        mlflow.end_run()


@pytest.fixture
def fast_model():
    """Fitted LightGBMPriceModel for log_model / load_model tests."""
    from src.ml.models.lightgbm_model import LightGBMParams, LightGBMPriceModel

    rng = np.random.default_rng(0)
    n = 40
    X = pd.DataFrame({"f1": rng.normal(0, 1, n), "f2": rng.normal(0, 1, n)})
    y = pd.Series(rng.normal(0, 0.05, n))
    params = LightGBMParams(
        n_estimators=5,
        num_leaves=4,
        min_child_samples=5,
        learning_rate=0.3,
        subsample=1.0,
        colsample_bytree=1.0,
        random_state=0,
    )
    model = LightGBMPriceModel(params)
    model.fit(X.iloc[:30], y.iloc[:30], X.iloc[30:], y.iloc[30:])
    return model


# ---------------------------------------------------------------------------
# setup_experiment()
# ---------------------------------------------------------------------------


def test_setup_experiment_creates_experiment():
    from src.ml.training.tracking import setup_experiment

    setup_experiment("test_exp")
    exp = mlflow.get_experiment_by_name("test_exp")
    assert exp is not None


def test_setup_experiment_uses_default_name():
    from src.ml.training.tracking import EXPERIMENT_NAME, setup_experiment

    setup_experiment()
    exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    assert exp is not None


def test_setup_experiment_does_not_raise_on_repeat_call():
    from src.ml.training.tracking import setup_experiment

    setup_experiment("repeated")
    setup_experiment("repeated")  # second call must not raise


# ---------------------------------------------------------------------------
# start_run()
# ---------------------------------------------------------------------------


def test_start_run_yields_active_run():
    from src.ml.training.tracking import setup_experiment, start_run

    setup_experiment("test_start_run")
    with start_run("test_run") as run:
        assert run is not None
        assert mlflow.active_run() is not None


def test_start_run_ends_run_on_exit():
    from src.ml.training.tracking import setup_experiment, start_run

    setup_experiment("test_start_run")
    with start_run("test_run"):
        pass
    assert mlflow.active_run() is None


def test_start_run_logs_snapshot_date():
    from src.ml.training.tracking import setup_experiment, start_run

    setup_experiment("test_snapshot")
    with start_run("run_with_date", snapshot_date="2026-06-09") as run:
        run_id = run.info.run_id
    params = mlflow.get_run(run_id).data.params
    assert params.get("gold_snapshot_date") == "2026-06-09"


def test_start_run_no_snapshot_date_does_not_log_empty_string():
    from src.ml.training.tracking import setup_experiment, start_run

    setup_experiment("test_no_snapshot")
    with start_run("run_no_date") as run:
        run_id = run.info.run_id
    params = mlflow.get_run(run_id).data.params
    # Empty snapshot_date should not be logged
    assert "gold_snapshot_date" not in params


def test_start_run_run_name_appears_in_params():
    from src.ml.training.tracking import setup_experiment, start_run

    setup_experiment("test_run_name")
    with start_run("my_run") as run:
        run_id = run.info.run_id
    params = mlflow.get_run(run_id).data.params
    assert params.get("run_name") == "my_run"


# ---------------------------------------------------------------------------
# log_params()
# ---------------------------------------------------------------------------


def test_log_params_with_dict():
    from src.ml.training.tracking import log_params, setup_experiment, start_run

    setup_experiment("test_params")
    with start_run("params_test") as run:
        log_params({"learning_rate": 0.05, "num_leaves": 63})
        run_id = run.info.run_id
    data = mlflow.get_run(run_id).data.params
    assert data["learning_rate"] == "0.05"
    assert data["num_leaves"] == "63"


def test_log_params_with_dataclass():
    from src.ml.models.lightgbm_model import LightGBMParams
    from src.ml.training.tracking import log_params, setup_experiment, start_run

    setup_experiment("test_params_dc")
    params = LightGBMParams(learning_rate=0.1, num_leaves=32)
    with start_run("dc_test") as run:
        log_params(params)
        run_id = run.info.run_id
    data = mlflow.get_run(run_id).data.params
    assert data["learning_rate"] == "0.1"
    assert data["num_leaves"] == "32"


# ---------------------------------------------------------------------------
# log_metrics()
# ---------------------------------------------------------------------------


def test_log_metrics_basic():
    from src.ml.training.tracking import log_metrics, setup_experiment, start_run

    setup_experiment("test_metrics_log")
    with start_run("metrics_test") as run:
        log_metrics({"mae_tier1": 0.15, "mape_tier1": 12.3})
        run_id = run.info.run_id
    data = mlflow.get_run(run_id).data.metrics
    assert abs(data["mae_tier1"] - 0.15) < 1e-6
    assert abs(data["mape_tier1"] - 12.3) < 1e-4


def test_log_metrics_with_step():
    from src.ml.training.tracking import log_metrics, setup_experiment, start_run

    setup_experiment("test_metrics_step")
    with start_run("step_test") as run:
        log_metrics({"val_mae": 0.10}, step=1)
        log_metrics({"val_mae": 0.08}, step=2)
        run_id = run.info.run_id
    # The last logged value is what mlflow.get_run returns
    data = mlflow.get_run(run_id).data.metrics
    assert abs(data["val_mae"] - 0.08) < 1e-6


# ---------------------------------------------------------------------------
# log_model() / load_model_from_mlflow()
# ---------------------------------------------------------------------------


def test_log_model_returns_string_run_id(fast_model):
    from src.ml.training.tracking import log_model, setup_experiment, start_run

    setup_experiment("test_log_model")
    with start_run("model_test"):
        run_id = log_model(fast_model)
    assert isinstance(run_id, str)
    assert len(run_id) > 0


def test_load_model_from_mlflow_returns_booster(fast_model):
    import lightgbm as lgb
    from src.ml.training.tracking import (
        load_model_from_mlflow,
        log_model,
        setup_experiment,
        start_run,
    )

    setup_experiment("test_load_model")
    with start_run("save_and_load"):
        run_id = log_model(fast_model)

    loaded = load_model_from_mlflow(run_id)
    assert isinstance(loaded, lgb.Booster)


def test_loaded_model_can_predict(fast_model):
    from src.ml.training.tracking import (
        load_model_from_mlflow,
        log_model,
        setup_experiment,
        start_run,
    )

    setup_experiment("test_predict_loaded")
    rng = np.random.default_rng(99)
    X_test = pd.DataFrame({"f1": rng.normal(0, 1, 5), "f2": rng.normal(0, 1, 5)})
    with start_run("save_predict"):
        run_id = log_model(fast_model)

    loaded = load_model_from_mlflow(run_id)
    preds = loaded.predict(X_test)
    assert len(preds) == 5
    assert np.all(np.isfinite(preds))


# ---------------------------------------------------------------------------
# log_cv_results()
# ---------------------------------------------------------------------------


@pytest.fixture
def cv_df():
    return pd.DataFrame(
        {
            "fold_idx": [0, 0, 1, 1],
            "tier": [1, 2, 1, 2],
            "mae": [0.10, 0.20, 0.12, 0.18],
            "mape": [15.0, 25.0, 14.0, 22.0],
        }
    )


def test_log_cv_results_logs_average_metrics(cv_df):
    from src.ml.training.tracking import log_cv_results, setup_experiment, start_run

    setup_experiment("test_cv_results")
    with start_run("cv_test") as run:
        log_cv_results(cv_df)
        run_id = run.info.run_id
    data = mlflow.get_run(run_id).data.metrics
    assert "cv_mae_tier1" in data
    assert "cv_mae_tier2" in data


def test_log_cv_results_tier1_mean_correct(cv_df):
    from src.ml.training.tracking import log_cv_results, setup_experiment, start_run

    setup_experiment("test_cv_tier1")
    with start_run("cv_tier1") as run:
        log_cv_results(cv_df)
        run_id = run.info.run_id
    # Mean of fold 0 (0.10) and fold 1 (0.12) for tier 1 = 0.11
    data = mlflow.get_run(run_id).data.metrics
    assert abs(data["cv_mae_tier1"] - 0.11) < 1e-6


def test_log_cv_results_saves_csv_artifact(cv_df, tmp_path):
    from src.ml.training.tracking import log_cv_results, setup_experiment, start_run

    setup_experiment("test_cv_artifact")
    with start_run("cv_artifact") as run:
        log_cv_results(cv_df)
        run_id = run.info.run_id
    artifacts = mlflow.MlflowClient().list_artifacts(run_id, "cv_results")
    assert len(artifacts) >= 1
