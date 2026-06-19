"""Unit tests for src/monitoring/retraining.py.

MLflow-dependent functions (retrain, promote_to_production, _compare_and_promote)
are tested with mocked MLflow clients.  should_retrain is tested by mocking
the underlying monitoring functions it calls.
"""

from datetime import date
from unittest.mock import MagicMock, patch

import duckdb
import pandas as pd
import pytest

import mlflow.exceptions
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR, RESOURCE_DOES_NOT_EXIST

from src.monitoring.retraining import (
    MODEL_REGISTRY_NAME,
    _compare_and_promote,
    promote_to_production,
    should_retrain,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def conn_no_events():
    """DuckDB with empty gold_events and gold_predictions/gold_price_features."""
    con = duckdb.connect()
    con.execute("""
        CREATE TABLE gold_events (
            event_date DATE, format VARCHAR, event_type VARCHAR, card_name VARCHAR
        )
    """)
    con.execute("""
        CREATE TABLE gold_predictions (
            uuid VARCHAR, snapshot_date DATE,
            predicted_eur DOUBLE, model_run_id VARCHAR,
            created_at TIMESTAMP
        )
    """)
    con.execute("""
        CREATE TABLE gold_price_features (
            uuid VARCHAR, snapshot_date DATE, eur DOUBLE,
            edhrec_rank DOUBLE, foil_premium DOUBLE
        )
    """)
    yield con
    con.close()


@pytest.fixture
def conn_with_ban_event(conn_no_events):
    """DuckDB with a ban event on today's date."""
    conn_no_events.execute(
        "INSERT INTO gold_events VALUES (?, ?, ?, ?)",
        [date.today(), "modern", "ban", "Test Card"],
    )
    return conn_no_events


# ---------------------------------------------------------------------------
# should_retrain — ban event path
# ---------------------------------------------------------------------------


def test_should_retrain_returns_tuple(conn_with_ban_event):
    result = should_retrain(conn_with_ban_event)
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_should_retrain_ban_event_triggers_retrain(conn_with_ban_event):
    retrain_flag, reason = should_retrain(conn_with_ban_event)
    assert retrain_flag is True


def test_should_retrain_ban_event_reason_is_ban_event(conn_with_ban_event):
    _, reason = should_retrain(conn_with_ban_event)
    assert reason == "ban_event"


# ---------------------------------------------------------------------------
# should_retrain — MAPE threshold path
# ---------------------------------------------------------------------------


def test_should_retrain_mape_alert_triggers_retrain(conn_no_events):
    mape_df = pd.DataFrame({"mape": [35.0, 38.0, 40.0]})
    with (
        patch("src.monitoring.retraining.compute_rolling_mape", return_value=mape_df),
        patch("src.monitoring.retraining.is_mape_alert", return_value=True),
    ):
        retrain_flag, reason = should_retrain(conn_no_events)
    assert retrain_flag is True
    assert reason == "mape_threshold"


# ---------------------------------------------------------------------------
# should_retrain — no trigger path
# ---------------------------------------------------------------------------


def test_should_retrain_no_trigger_when_both_false(conn_no_events):
    mape_df = pd.DataFrame({"mape": [5.0, 6.0, 7.0]})
    with (
        patch("src.monitoring.retraining.compute_rolling_mape", return_value=mape_df),
        patch("src.monitoring.retraining.is_mape_alert", return_value=False),
    ):
        retrain_flag, reason = should_retrain(conn_no_events)
    assert retrain_flag is False
    assert reason == "no_trigger"


def test_should_retrain_no_trigger_reason_is_string(conn_no_events):
    mape_df = pd.DataFrame({"mape": []})
    with (
        patch("src.monitoring.retraining.compute_rolling_mape", return_value=mape_df),
        patch("src.monitoring.retraining.is_mape_alert", return_value=False),
    ):
        _, reason = should_retrain(conn_no_events)
    assert isinstance(reason, str)


def test_should_retrain_ban_takes_priority_over_mape(conn_with_ban_event):
    # Even if MAPE would NOT trigger, ban event should take priority
    mape_df = pd.DataFrame({"mape": [5.0, 5.0, 5.0]})
    with (
        patch("src.monitoring.retraining.compute_rolling_mape", return_value=mape_df),
        patch("src.monitoring.retraining.is_mape_alert", return_value=False),
    ):
        _, reason = should_retrain(conn_with_ban_event)
    assert reason == "ban_event"


def test_should_retrain_skips_mape_check_when_ban_detected(conn_with_ban_event):
    # is_mape_alert should never be called when ban event is found
    with patch("src.monitoring.retraining.is_mape_alert") as mock_mape:
        should_retrain(conn_with_ban_event)
    mock_mape.assert_not_called()


# ---------------------------------------------------------------------------
# promote_to_production — alias API path
# mlflow is imported locally inside the function, so we patch mlflow.* directly.
# ---------------------------------------------------------------------------


def test_promote_to_production_registers_model():
    mock_client = MagicMock()
    mock_result = MagicMock()
    mock_result.version = "3"

    with (
        patch("mlflow.register_model", return_value=mock_result),
        patch("mlflow.tracking.MlflowClient", return_value=mock_client),
    ):
        promote_to_production("run_xyz", "test_model")

    mock_result  # used for version


def test_promote_to_production_register_called_with_correct_uri():
    mock_client = MagicMock()
    mock_result = MagicMock()
    mock_result.version = "3"

    with (
        patch("mlflow.register_model", return_value=mock_result) as mock_reg,
        patch("mlflow.tracking.MlflowClient", return_value=mock_client),
    ):
        promote_to_production("run_xyz", "test_model")

    mock_reg.assert_called_once_with("runs:/run_xyz/model", "test_model")


def test_promote_to_production_sets_production_alias():
    mock_client = MagicMock()
    mock_result = MagicMock()
    mock_result.version = "3"

    with (
        patch("mlflow.register_model", return_value=mock_result),
        patch("mlflow.tracking.MlflowClient", return_value=mock_client),
    ):
        promote_to_production("run_xyz", "test_model")

    mock_client.set_registered_model_alias.assert_called_once_with(
        "test_model", "production", "3"
    )


def test_promote_to_production_uses_default_model_name():
    mock_client = MagicMock()
    mock_result = MagicMock()
    mock_result.version = "1"

    with (
        patch("mlflow.register_model", return_value=mock_result) as mock_reg,
        patch("mlflow.tracking.MlflowClient", return_value=mock_client),
    ):
        promote_to_production("run_abc")

    mock_reg.assert_called_once_with("runs:/run_abc/model", MODEL_REGISTRY_NAME)


def test_promote_to_production_falls_back_to_stage_on_attribute_error():
    """set_registered_model_alias not available → falls back to transition_model_version_stage."""
    mock_client = MagicMock()
    mock_client.set_registered_model_alias.side_effect = AttributeError("old API")
    mock_result = MagicMock()
    mock_result.version = "2"

    with (
        patch("mlflow.register_model", return_value=mock_result),
        patch("mlflow.tracking.MlflowClient", return_value=mock_client),
    ):
        promote_to_production("run_fallback", "test_model")

    mock_client.transition_model_version_stage.assert_called_once_with(
        name="test_model",
        version="2",
        stage="Production",
        archive_existing_versions=True,
    )


# ---------------------------------------------------------------------------
# _compare_and_promote — MLflow exception handling
# ---------------------------------------------------------------------------


def test_compare_and_promote_reraises_unexpected_mlflow_errors():
    unexpected_exc = mlflow.exceptions.MlflowException(
        "Server unreachable", error_code=INTERNAL_ERROR
    )
    with (
        patch("mlflow.tracking.MlflowClient") as mock_client_cls,
        patch("src.monitoring.retraining.promote_to_production") as mock_promote,
    ):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.get_model_version_by_alias.side_effect = unexpected_exc
        with pytest.raises(mlflow.exceptions.MlflowException):
            _compare_and_promote(pd.DataFrame(), "test-run-id")
        mock_promote.assert_not_called()


def test_compare_and_promote_promotes_when_no_production_alias():
    not_found_exc = mlflow.exceptions.MlflowException(
        "Alias not found", error_code=RESOURCE_DOES_NOT_EXIST
    )
    with (
        patch("mlflow.tracking.MlflowClient") as mock_client_cls,
        patch("src.monitoring.retraining.promote_to_production") as mock_promote,
    ):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.get_model_version_by_alias.side_effect = not_found_exc
        _compare_and_promote(pd.DataFrame(), "new-run-id")
        mock_promote.assert_called_once_with("new-run-id", MODEL_REGISTRY_NAME)


# ---------------------------------------------------------------------------
# _compare_and_promote — production model exists paths (lines 208-226)
# ---------------------------------------------------------------------------


def _make_cv(tier1_mape: float) -> pd.DataFrame:
    return pd.DataFrame({"tier": [1], "mape": [tier1_mape]})


def test_compare_and_promote_promotes_when_new_model_is_better():
    """New CV MAPE < prod MAPE → promote."""
    with (
        patch("mlflow.tracking.MlflowClient") as mock_client_cls,
        patch("src.monitoring.retraining.promote_to_production") as mock_promote,
    ):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        prod_version = MagicMock()
        prod_version.run_id = "old-run-id"
        mock_client.get_model_version_by_alias.return_value = prod_version
        mock_client.get_run.return_value.data.metrics = {"cv_mape_tier1": 0.30}

        _compare_and_promote(_make_cv(0.20), "new-run-id")

        mock_promote.assert_called_once_with("new-run-id", MODEL_REGISTRY_NAME)


def test_compare_and_promote_does_not_promote_when_new_model_is_worse():
    """New CV MAPE > prod MAPE → keep production, no promotion."""
    with (
        patch("mlflow.tracking.MlflowClient") as mock_client_cls,
        patch("src.monitoring.retraining.promote_to_production") as mock_promote,
    ):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        prod_version = MagicMock()
        prod_version.run_id = "old-run-id"
        mock_client.get_model_version_by_alias.return_value = prod_version
        mock_client.get_run.return_value.data.metrics = {"cv_mape_tier1": 0.10}

        _compare_and_promote(_make_cv(0.25), "new-run-id")

        mock_promote.assert_not_called()


def test_compare_and_promote_promotes_when_prod_run_id_is_none():
    """prod_version.run_id is None → promote unconditionally."""
    with (
        patch("mlflow.tracking.MlflowClient") as mock_client_cls,
        patch("src.monitoring.retraining.promote_to_production") as mock_promote,
    ):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        prod_version = MagicMock()
        prod_version.run_id = None
        mock_client.get_model_version_by_alias.return_value = prod_version

        _compare_and_promote(pd.DataFrame(), "new-run-id")

        mock_promote.assert_called_once_with("new-run-id", MODEL_REGISTRY_NAME)


def test_compare_and_promote_uses_inf_mape_when_cv_empty():
    """Empty cv_results → new_mape = inf → never promotes over a prod with finite MAPE."""
    with (
        patch("mlflow.tracking.MlflowClient") as mock_client_cls,
        patch("src.monitoring.retraining.promote_to_production") as mock_promote,
    ):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        prod_version = MagicMock()
        prod_version.run_id = "old-run-id"
        mock_client.get_model_version_by_alias.return_value = prod_version
        mock_client.get_run.return_value.data.metrics = {"cv_mape_tier1": 0.15}

        _compare_and_promote(pd.DataFrame(), "new-run-id")

        mock_promote.assert_not_called()
