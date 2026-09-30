"""Unit tests for scripts/check_and_retrain.py."""

import json
from unittest.mock import MagicMock

import mlflow
import pandas as pd
import pytest

from scripts import check_and_retrain
from src.monitoring.drift import DriftResult
from src.monitoring.prediction_tracker import SkillStatus

REAL_RUN_MONITORING = check_and_retrain.run_monitoring


@pytest.fixture(autouse=True)
def _no_real_monitoring(monkeypatch):
    """main() tests exercise the retrain decision; run_monitoring has its own tests."""
    monkeypatch.setattr(
        check_and_retrain,
        "run_monitoring",
        lambda conn: check_and_retrain.MonitoringResult(),
    )


def _make_fake_conn_with_snapshot(snapshot_date: str) -> MagicMock:
    """Build a fake DuckDB connection whose "latest trainable snapshot" query
    resolves to `snapshot_date`.

    get_latest_trainable_snapshot_date() first checks table presence via a
    ``SHOW TABLES`` query (get_tables) before running its own query, so the
    fake execute() must respond to both statements rather than always
    returning the same canned fetchone() result.
    """

    def _execute(sql, *args, **kwargs):
        result = MagicMock()
        if "SHOW TABLES" in sql:
            result.fetchall.return_value = [("gold_price_features",)]
        else:
            result.fetchone.return_value = (snapshot_date,)
        return result

    fake_conn = MagicMock()
    fake_conn.execute.side_effect = _execute
    return fake_conn


def test_main_returns_1_when_gold_db_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(
        check_and_retrain, "GOLD_DB_PATH", str(tmp_path / "missing.duckdb")
    )
    monkeypatch.setattr(check_and_retrain, "STATUS_PATH", tmp_path / "status.json")

    exit_code = check_and_retrain.main()

    assert exit_code == 1
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["result"] == "error"
    assert status["reason"] == "gold_db_missing"


def test_main_skips_retrain_when_no_trigger(tmp_path, monkeypatch):
    db_path = tmp_path / "cards.duckdb"
    db_path.touch()
    monkeypatch.setattr(check_and_retrain, "GOLD_DB_PATH", str(db_path))
    monkeypatch.setattr(check_and_retrain, "STATUS_PATH", tmp_path / "status.json")
    monkeypatch.setattr(
        check_and_retrain.duckdb, "connect", lambda *a, **k: MagicMock()
    )
    monkeypatch.setattr(
        check_and_retrain,
        "should_retrain",
        lambda conn, skill_df=None: (False, "no_trigger"),
    )
    mock_retrain = MagicMock()
    monkeypatch.setattr(check_and_retrain, "retrain", mock_retrain)

    exit_code = check_and_retrain.main()

    assert exit_code == 0
    mock_retrain.assert_not_called()
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["result"] == "no_retrain"
    assert status["reason"] == "no_trigger"


def test_main_retrains_when_triggered(tmp_path, monkeypatch):
    db_path = tmp_path / "cards.duckdb"
    db_path.touch()
    monkeypatch.setattr(check_and_retrain, "GOLD_DB_PATH", str(db_path))
    monkeypatch.setattr(check_and_retrain, "STATUS_PATH", tmp_path / "status.json")

    fake_conn = _make_fake_conn_with_snapshot("2026-07-01")
    monkeypatch.setattr(check_and_retrain.duckdb, "connect", lambda *a, **k: fake_conn)
    monkeypatch.setattr(
        check_and_retrain,
        "should_retrain",
        lambda conn, skill_df=None: (True, "model_worse_than_naive"),
    )
    monkeypatch.setattr(
        check_and_retrain, "retrain", lambda conn, snapshot_date: "abc123"
    )
    mock_send_alert = MagicMock()
    monkeypatch.setattr(check_and_retrain, "send_alert", mock_send_alert)

    exit_code = check_and_retrain.main()

    assert exit_code == 0
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["result"] == "retrained"
    assert status["reason"] == "model_worse_than_naive"
    assert status["run_id"] == "abc123"
    mock_send_alert.assert_not_called()


def test_main_writes_error_status_when_retrain_raises(tmp_path, monkeypatch):
    db_path = tmp_path / "cards.duckdb"
    db_path.touch()
    monkeypatch.setattr(check_and_retrain, "GOLD_DB_PATH", str(db_path))
    monkeypatch.setattr(check_and_retrain, "STATUS_PATH", tmp_path / "status.json")

    fake_conn = _make_fake_conn_with_snapshot("2026-07-01")
    monkeypatch.setattr(check_and_retrain.duckdb, "connect", lambda *a, **k: fake_conn)
    monkeypatch.setattr(
        check_and_retrain,
        "should_retrain",
        lambda conn, skill_df=None: (True, "model_worse_than_naive"),
    )

    def _raise(conn, snapshot_date):
        raise RuntimeError("mlflow boom")

    monkeypatch.setattr(check_and_retrain, "retrain", _raise)

    exit_code = check_and_retrain.main()

    assert exit_code == 1
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["result"] == "error"
    assert status["reason"] == "retrain_failed"
    assert "mlflow boom" in status["error"]


def test_main_alerts_when_gold_db_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(
        check_and_retrain, "GOLD_DB_PATH", str(tmp_path / "missing.duckdb")
    )
    monkeypatch.setattr(check_and_retrain, "STATUS_PATH", tmp_path / "status.json")
    mock_send_alert = MagicMock()
    monkeypatch.setattr(check_and_retrain, "send_alert", mock_send_alert)

    check_and_retrain.main()

    mock_send_alert.assert_called_once()


def test_main_alerts_when_no_snapshot(tmp_path, monkeypatch):
    db_path = tmp_path / "cards.duckdb"
    db_path.touch()
    monkeypatch.setattr(check_and_retrain, "GOLD_DB_PATH", str(db_path))
    monkeypatch.setattr(check_and_retrain, "STATUS_PATH", tmp_path / "status.json")
    monkeypatch.setattr(
        check_and_retrain.duckdb, "connect", lambda *a, **k: MagicMock()
    )
    monkeypatch.setattr(
        check_and_retrain,
        "should_retrain",
        lambda conn, skill_df=None: (True, "model_worse_than_naive"),
    )
    monkeypatch.setattr(
        check_and_retrain,
        "get_latest_trainable_snapshot_date",
        lambda conn: None,
    )
    mock_send_alert = MagicMock()
    monkeypatch.setattr(check_and_retrain, "send_alert", mock_send_alert)

    exit_code = check_and_retrain.main()

    assert exit_code == 1
    mock_send_alert.assert_called_once()


def test_main_alerts_when_retrain_raises(tmp_path, monkeypatch):
    db_path = tmp_path / "cards.duckdb"
    db_path.touch()
    monkeypatch.setattr(check_and_retrain, "GOLD_DB_PATH", str(db_path))
    monkeypatch.setattr(check_and_retrain, "STATUS_PATH", tmp_path / "status.json")

    fake_conn = _make_fake_conn_with_snapshot("2026-07-01")
    monkeypatch.setattr(check_and_retrain.duckdb, "connect", lambda *a, **k: fake_conn)
    monkeypatch.setattr(
        check_and_retrain,
        "should_retrain",
        lambda conn, skill_df=None: (True, "model_worse_than_naive"),
    )

    def _raise(conn, snapshot_date):
        raise RuntimeError("mlflow boom")

    monkeypatch.setattr(check_and_retrain, "retrain", _raise)
    mock_send_alert = MagicMock()
    monkeypatch.setattr(check_and_retrain, "send_alert", mock_send_alert)

    check_and_retrain.main()

    mock_send_alert.assert_called_once()
    assert "mlflow boom" in mock_send_alert.call_args.args[1]


def test_main_does_not_alert_when_no_trigger(tmp_path, monkeypatch):
    db_path = tmp_path / "cards.duckdb"
    db_path.touch()
    monkeypatch.setattr(check_and_retrain, "GOLD_DB_PATH", str(db_path))
    monkeypatch.setattr(check_and_retrain, "STATUS_PATH", tmp_path / "status.json")
    monkeypatch.setattr(
        check_and_retrain.duckdb, "connect", lambda *a, **k: MagicMock()
    )
    monkeypatch.setattr(
        check_and_retrain,
        "should_retrain",
        lambda conn, skill_df=None: (False, "no_trigger"),
    )
    mock_send_alert = MagicMock()
    monkeypatch.setattr(check_and_retrain, "send_alert", mock_send_alert)

    check_and_retrain.main()

    mock_send_alert.assert_not_called()


@pytest.fixture(autouse=True)
def mlflow_tmp_for_real_retrain(tmp_path, monkeypatch):
    """Isolate the tracking DB *and* the artifact tree.

    A fresh sqlite store gives new experiments the relative artifact_location
    'mlruns/<id>', resolved against the cwd — so redirecting only the tracking
    URI left logged models accumulating in the project's real ./mlruns.
    _PROJECT_ROOT is patched to match the new cwd so setup_experiment's
    project-root guard still passes.
    """
    from src.ml.training import tracking

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(tracking, "_PROJECT_ROOT", tmp_path)
    db_path = tmp_path / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    yield
    if mlflow.active_run():
        mlflow.end_run()


def test_do_retrain_calls_real_retrain_and_writes_run_id(tiny_gold_conn):
    ok, status = check_and_retrain._do_retrain(
        tiny_gold_conn, "2026-06-01", "model_worse_than_naive"
    )

    assert ok is True
    assert status["result"] == "retrained"
    assert status["reason"] == "model_worse_than_naive"
    assert "run_id" in status and status["run_id"]


def test_main_pings_heartbeat_success_url_when_configured(tmp_path, monkeypatch):
    monkeypatch.setenv("HEARTBEAT_URL", "https://hc-ping.com/abc-123")
    db_path = tmp_path / "cards.duckdb"
    db_path.touch()
    monkeypatch.setattr(check_and_retrain, "GOLD_DB_PATH", str(db_path))
    monkeypatch.setattr(check_and_retrain, "STATUS_PATH", tmp_path / "status.json")
    monkeypatch.setattr(
        check_and_retrain.duckdb, "connect", lambda *a, **k: MagicMock()
    )
    monkeypatch.setattr(
        check_and_retrain,
        "should_retrain",
        lambda conn, skill_df=None: (False, "no_trigger"),
    )
    mock_get = MagicMock()
    monkeypatch.setattr(check_and_retrain.httpx, "get", mock_get)

    check_and_retrain.main()

    mock_get.assert_called_once_with("https://hc-ping.com/abc-123", timeout=5.0)


def test_main_pings_heartbeat_fail_url_when_gold_db_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("HEARTBEAT_URL", "https://hc-ping.com/abc-123")
    monkeypatch.setattr(
        check_and_retrain, "GOLD_DB_PATH", str(tmp_path / "missing.duckdb")
    )
    monkeypatch.setattr(check_and_retrain, "STATUS_PATH", tmp_path / "status.json")
    mock_get = MagicMock()
    monkeypatch.setattr(check_and_retrain.httpx, "get", mock_get)

    check_and_retrain.main()

    mock_get.assert_called_once_with("https://hc-ping.com/abc-123/fail", timeout=5.0)


def test_main_skips_heartbeat_when_url_not_configured(tmp_path, monkeypatch):
    monkeypatch.delenv("HEARTBEAT_URL", raising=False)
    monkeypatch.setattr(
        check_and_retrain, "GOLD_DB_PATH", str(tmp_path / "missing.duckdb")
    )
    monkeypatch.setattr(check_and_retrain, "STATUS_PATH", tmp_path / "status.json")
    mock_get = MagicMock()
    monkeypatch.setattr(check_and_retrain.httpx, "get", mock_get)

    check_and_retrain.main()

    mock_get.assert_not_called()


def test_main_does_not_raise_when_heartbeat_ping_fails(tmp_path, monkeypatch):
    monkeypatch.setenv("HEARTBEAT_URL", "https://hc-ping.com/abc-123")
    monkeypatch.setattr(
        check_and_retrain, "GOLD_DB_PATH", str(tmp_path / "missing.duckdb")
    )
    monkeypatch.setattr(check_and_retrain, "STATUS_PATH", tmp_path / "status.json")
    monkeypatch.setattr(
        check_and_retrain.httpx,
        "get",
        MagicMock(side_effect=check_and_retrain.httpx.ConnectError("down")),
    )

    exit_code = check_and_retrain.main()  # must not raise

    assert exit_code == 1  # gold_db_missing branch still ran to completion


# ---------------------------------------------------------------------------
# run_monitoring (ADR-034)
# ---------------------------------------------------------------------------


@pytest.fixture
def monitoring_env(tmp_path, monkeypatch):
    """Real run_monitoring with its collaborators replaced by controllable stubs."""
    monkeypatch.setattr(
        check_and_retrain, "MONITORING_DB_PATH", str(tmp_path / "mon" / "m.duckdb")
    )
    monkeypatch.setattr(
        check_and_retrain, "get_latest_gold_snapshot_date", lambda conn: "2026-10-30"
    )
    monkeypatch.setattr(check_and_retrain, "_served_run_id", lambda: "run1")
    monkeypatch.setattr(
        check_and_retrain, "record_daily_predictions", lambda *a, **k: 5
    )
    monkeypatch.setattr(
        check_and_retrain, "daily_skill", lambda mon, conn: pd.DataFrame()
    )
    monkeypatch.setattr(
        check_and_retrain,
        "return_drift",
        lambda conn, latest: DriftResult(0.03, False, 100, 50, "a", "b"),
    )
    alerts = MagicMock()
    monkeypatch.setattr(check_and_retrain, "send_alert", alerts)
    return alerts


def test_run_monitoring_records_predictions_and_stays_quiet_when_healthy(
    monitoring_env,
):
    result = REAL_RUN_MONITORING(MagicMock())

    assert result.predictions_recorded == 5
    assert result.served_run_id == "run1"
    assert result.errors == []
    monitoring_env.assert_not_called()


def test_run_monitoring_survives_a_failed_step_and_still_checks_drift(
    monitoring_env, monkeypatch
):
    def _boom(*a, **k):
        raise RuntimeError("mlflow unreachable")

    monkeypatch.setattr(check_and_retrain, "record_daily_predictions", _boom)

    result = REAL_RUN_MONITORING(MagicMock())

    assert result.errors == ["record predictions: mlflow unreachable"]
    assert result.drift is not None  # the next step still ran
    subject = monitoring_env.call_args.args[0]
    assert subject == "Monitoring step failed: record predictions"


def test_run_monitoring_alerts_when_model_is_worse_than_naive(
    monitoring_env, monkeypatch
):
    worse = pd.DataFrame(
        {
            "snapshot_date": pd.date_range("2026-10-01", periods=3),
            "skill": [-0.2, -0.15, -0.1],
        }
    )
    monkeypatch.setattr(check_and_retrain, "daily_skill", lambda mon, conn: worse)

    result = REAL_RUN_MONITORING(MagicMock())

    assert result.skill.degraded is True
    assert result.summary()["model_worse_than_naive"] is True
    assert monitoring_env.call_args.args[0] == "Model worse than naive"


def test_run_monitoring_alerts_on_return_drift(monitoring_env, monkeypatch):
    monkeypatch.setattr(
        check_and_retrain,
        "return_drift",
        lambda conn, latest: DriftResult(
            0.9, True, 100, 50, "2026-10-17", "2026-10-23"
        ),
    )

    result = REAL_RUN_MONITORING(MagicMock())

    assert result.summary()["drift_detected"] is True
    assert monitoring_env.call_args.args[0] == "Price-return drift"


def test_main_passes_skill_to_the_retrain_decision_and_records_monitoring(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "cards.duckdb"
    db_path.touch()
    monkeypatch.setattr(check_and_retrain, "GOLD_DB_PATH", str(db_path))
    monkeypatch.setattr(check_and_retrain, "STATUS_PATH", tmp_path / "status.json")
    monkeypatch.setattr(
        check_and_retrain.duckdb, "connect", lambda *a, **k: MagicMock()
    )
    skill_df = pd.DataFrame({"snapshot_date": [], "skill": []})
    result = check_and_retrain.MonitoringResult(
        predictions_recorded=7,
        skill_df=skill_df,
        skill=SkillStatus(4, "2026-10-01", 0.004, False),
    )
    monkeypatch.setattr(check_and_retrain, "run_monitoring", lambda conn: result)
    seen = {}

    def _should_retrain(conn, skill):
        seen["skill_df"] = skill
        return False, "no_trigger"

    monkeypatch.setattr(check_and_retrain, "should_retrain", _should_retrain)

    assert check_and_retrain.main() == 0

    assert seen["skill_df"] is skill_df
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["monitoring"]["predictions_recorded"] == 7
    assert status["monitoring"]["skill_latest"] == 0.004
