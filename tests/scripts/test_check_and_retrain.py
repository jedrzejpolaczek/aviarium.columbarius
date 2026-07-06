"""Unit tests for scripts/check_and_retrain.py."""

import json
from unittest.mock import MagicMock

from scripts import check_and_retrain


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
        check_and_retrain, "should_retrain", lambda conn: (False, "no_trigger")
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

    fake_conn = MagicMock()
    fake_conn.execute.return_value.fetchone.return_value = ("2026-07-01",)
    monkeypatch.setattr(check_and_retrain.duckdb, "connect", lambda *a, **k: fake_conn)
    monkeypatch.setattr(
        check_and_retrain, "should_retrain", lambda conn: (True, "mape_threshold")
    )
    monkeypatch.setattr(
        check_and_retrain, "retrain", lambda conn, snapshot_date: "abc123"
    )

    exit_code = check_and_retrain.main()

    assert exit_code == 0
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["result"] == "retrained"
    assert status["reason"] == "mape_threshold"
    assert status["run_id"] == "abc123"


def test_main_writes_error_status_when_retrain_raises(tmp_path, monkeypatch):
    db_path = tmp_path / "cards.duckdb"
    db_path.touch()
    monkeypatch.setattr(check_and_retrain, "GOLD_DB_PATH", str(db_path))
    monkeypatch.setattr(check_and_retrain, "STATUS_PATH", tmp_path / "status.json")

    fake_conn = MagicMock()
    fake_conn.execute.return_value.fetchone.return_value = ("2026-07-01",)
    monkeypatch.setattr(check_and_retrain.duckdb, "connect", lambda *a, **k: fake_conn)
    monkeypatch.setattr(
        check_and_retrain, "should_retrain", lambda conn: (True, "mape_threshold")
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
