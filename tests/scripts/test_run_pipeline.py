"""Unit tests for scripts/run_pipeline.py."""

import json
from unittest.mock import MagicMock

from scripts import run_pipeline


def test_main_calls_daily_pipeline_with_config_path(tmp_path, monkeypatch):
    mock_daily_pipeline = MagicMock()
    monkeypatch.setattr(run_pipeline, "daily_pipeline", mock_daily_pipeline)
    monkeypatch.setattr(run_pipeline, "setup_logging", MagicMock())
    monkeypatch.setattr(run_pipeline, "STATUS_PATH", tmp_path / "status.json")

    exit_code = run_pipeline.main()

    mock_daily_pipeline.assert_called_once()
    assert mock_daily_pipeline.call_args.args[0] == "configs/data_sources.yaml"
    assert exit_code == 0


def test_main_writes_success_status(tmp_path, monkeypatch):
    monkeypatch.setattr(run_pipeline, "daily_pipeline", MagicMock())
    monkeypatch.setattr(run_pipeline, "setup_logging", MagicMock())
    status_path = tmp_path / "status.json"
    monkeypatch.setattr(run_pipeline, "STATUS_PATH", status_path)

    run_pipeline.main()

    status = json.loads(status_path.read_text())
    assert status["result"] == "success"
    assert "checked_at" in status


def test_main_returns_1_and_writes_error_status_when_pipeline_raises(
    tmp_path, monkeypatch
):
    def _raise(config_path):
        raise RuntimeError("mtgjson download failed")

    monkeypatch.setattr(run_pipeline, "daily_pipeline", _raise)
    monkeypatch.setattr(run_pipeline, "setup_logging", MagicMock())
    status_path = tmp_path / "status.json"
    monkeypatch.setattr(run_pipeline, "STATUS_PATH", status_path)
    mock_send_alert = MagicMock()
    monkeypatch.setattr(run_pipeline, "send_alert", mock_send_alert)

    exit_code = run_pipeline.main()

    assert exit_code == 1
    status = json.loads(status_path.read_text())
    assert status["result"] == "error"
    assert "mtgjson download failed" in status["error"]
    mock_send_alert.assert_called_once()
    assert "mtgjson download failed" in mock_send_alert.call_args.args[1]


def test_main_does_not_alert_on_success(tmp_path, monkeypatch):
    monkeypatch.setattr(run_pipeline, "daily_pipeline", MagicMock())
    monkeypatch.setattr(run_pipeline, "setup_logging", MagicMock())
    monkeypatch.setattr(run_pipeline, "STATUS_PATH", tmp_path / "status.json")
    mock_send_alert = MagicMock()
    monkeypatch.setattr(run_pipeline, "send_alert", mock_send_alert)

    run_pipeline.main()

    mock_send_alert.assert_not_called()
