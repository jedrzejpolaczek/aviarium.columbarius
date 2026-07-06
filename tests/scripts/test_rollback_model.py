"""Unit tests for scripts/rollback_model.py."""

from unittest.mock import MagicMock, patch

import mlflow.exceptions

from scripts import rollback_model


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
    fake_client = MagicMock()
    fake_client.set_registered_model_alias.side_effect = (
        mlflow.exceptions.MlflowException("not found")
    )
    with patch("mlflow.tracking.MlflowClient", return_value=fake_client):
        exit_code = rollback_model.main()

    assert exit_code == 1
