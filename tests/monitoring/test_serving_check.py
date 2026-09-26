"""Unit tests for src/monitoring/serving_check.py."""

from unittest.mock import MagicMock, patch

import mlflow
import mlflow.exceptions
import pytest
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR, RESOURCE_DOES_NOT_EXIST

from src.monitoring.serving_check import (
    check_serving_matches_production_alias,
    log_serving_alias_check,
)


def _client_returning(version: str, run_id: str) -> MagicMock:
    prod_version = MagicMock()
    prod_version.version = version
    prod_version.run_id = run_id
    client = MagicMock()
    client.get_model_version_by_alias.return_value = prod_version
    return client


def test_ok_when_served_run_matches_alias():
    with patch(
        "mlflow.tracking.MlflowClient", return_value=_client_returning("3", "run-abc")
    ):
        ok, detail = check_serving_matches_production_alias("run-abc")

    assert ok
    assert "matches" in detail


def test_mismatch_when_alias_points_elsewhere():
    """The real 2026-09-26 state: serving version 3, alias on version 2."""
    with patch(
        "mlflow.tracking.MlflowClient",
        return_value=_client_returning("2", "c46d4787f52a43faa7f95c071f02b5d8"),
    ):
        ok, detail = check_serving_matches_production_alias(
            "9c1ec7de65c7476aa75a199b7189d6f2"
        )

    assert not ok
    assert "9c1ec7de65c7476aa75a199b7189d6f2" in detail
    assert "c46d4787f52a43faa7f95c071f02b5d8" in detail
    assert "rollback_model" in detail


def test_skipped_when_model_run_id_unset():
    ok, detail = check_serving_matches_production_alias("")

    assert ok
    assert "unset" in detail


def test_skipped_when_no_production_alias_yet():
    client = MagicMock()
    client.get_model_version_by_alias.side_effect = mlflow.exceptions.MlflowException(
        "alias not found", error_code=RESOURCE_DOES_NOT_EXIST
    )
    with patch("mlflow.tracking.MlflowClient", return_value=client):
        ok, detail = check_serving_matches_production_alias("run-abc")

    assert ok
    assert "no 'production' alias" in detail


def test_skipped_when_registry_unreachable():
    """An unreadable registry must not page anyone — the check simply did not run."""
    client = MagicMock()
    client.get_model_version_by_alias.side_effect = mlflow.exceptions.MlflowException(
        "server unreachable", error_code=INTERNAL_ERROR
    )
    with patch("mlflow.tracking.MlflowClient", return_value=client):
        ok, detail = check_serving_matches_production_alias("run-abc")

    assert ok
    assert "skipped" in detail


@pytest.mark.parametrize("matching", [True, False])
def test_log_serving_alias_check_reads_env(monkeypatch, matching):
    monkeypatch.setenv("MODEL_RUN_ID", "run-abc")
    served = "run-abc" if matching else "run-other"
    with patch(
        "mlflow.tracking.MlflowClient", return_value=_client_returning("3", served)
    ):
        ok, _ = log_serving_alias_check()

    assert ok is matching
