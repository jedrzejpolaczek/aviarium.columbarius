"""Unit tests for scripts/rollback_model.py."""

from unittest.mock import MagicMock, patch

import mlflow
import mlflow.exceptions

from scripts import rollback_model
from scripts.rollback_model import rollback


def test_rollback_sets_production_alias():
    fake_client = MagicMock()
    with patch("mlflow.tracking.MlflowClient", return_value=fake_client):
        rollback_model.rollback("3", model_name="mtg_price_model")

    fake_client.set_registered_model_alias.assert_called_once_with(
        "mtg_price_model", "production", "3"
    )


def test_main_returns_0_on_success(monkeypatch):
    monkeypatch.setattr(
        rollback_model.sys, "argv", ["rollback_model.py", "--version", "3"]
    )
    monkeypatch.setattr(rollback_model, "setup_experiment", lambda: None)
    fake_client = MagicMock()
    with patch("mlflow.tracking.MlflowClient", return_value=fake_client):
        exit_code = rollback_model.main()

    assert exit_code == 0
    fake_client.set_registered_model_alias.assert_called_once_with(
        "mtg_price_model", "production", "3"
    )


def test_main_returns_1_on_missing_version(monkeypatch):
    monkeypatch.setattr(
        rollback_model.sys, "argv", ["rollback_model.py", "--version", "999"]
    )
    monkeypatch.setattr(rollback_model, "setup_experiment", lambda: None)
    fake_client = MagicMock()
    fake_client.set_registered_model_alias.side_effect = (
        mlflow.exceptions.MlflowException("not found")
    )
    with patch("mlflow.tracking.MlflowClient", return_value=fake_client):
        exit_code = rollback_model.main()

    assert exit_code == 1


def test_main_passes_model_name_override(monkeypatch):
    monkeypatch.setattr(
        rollback_model.sys,
        "argv",
        ["rollback_model.py", "--version", "3", "--model-name", "other_model"],
    )
    monkeypatch.setattr(rollback_model, "setup_experiment", lambda: None)
    fake_client = MagicMock()
    with patch("mlflow.tracking.MlflowClient", return_value=fake_client):
        exit_code = rollback_model.main()

    assert exit_code == 0
    fake_client.set_registered_model_alias.assert_called_once_with(
        "other_model", "production", "3"
    )


class _EchoModel(mlflow.pyfunc.PythonModel):  # type: ignore[name-defined]
    # mlflow.pyfunc doesn't explicitly re-export PythonModel in its stubs,
    # so mypy can't resolve the name even though it exists at runtime.
    def predict(self, context, model_input):
        return model_input


def test_rollback_sets_real_alias_to_specified_version(tmp_path):
    """Exercise rollback() against a real (sqlite-backed) MLflow model registry.

    Registers two throwaway versions of the same model name via two runs, then
    rolls back to version "1" and confirms the "production" alias really points
    at it — catching regressions (e.g. wrong alias name, wrong version format)
    that a fully-mocked MlflowClient would not.
    """
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    # mlflow's fluent API caches the active experiment id at module scope, so
    # without re-setting it here, start_run() below could reuse a stale
    # experiment id left behind by an earlier test's (different) tracking
    # store and fail with "No Experiment with id=<n> exists".
    mlflow.set_experiment("rollback_model_test")
    client = mlflow.tracking.MlflowClient()

    with mlflow.start_run():
        mlflow.pyfunc.log_model(
            "model",
            python_model=_EchoModel(),
            registered_model_name="mtg_price_model",
        )
    with mlflow.start_run():
        mlflow.pyfunc.log_model(
            "model",
            python_model=_EchoModel(),
            registered_model_name="mtg_price_model",
        )

    rollback(version="1", model_name="mtg_price_model")

    prod = client.get_model_version_by_alias("mtg_price_model", "production")
    # ModelVersion.version is typed str but comes back as int at runtime
    # against the sqlite-backed store used here.
    assert str(prod.version) == "1"
