"""Unit tests for src/monitoring/alerts.py."""

import json
from unittest.mock import MagicMock

import pytest

from src.monitoring import alerts


@pytest.fixture(autouse=True)
def _mock_plyer_notify(monkeypatch):
    """Never let a real OS notification pop up during the test suite."""
    mock_notify = MagicMock()
    monkeypatch.setattr("plyer.notification.notify", mock_notify)
    return mock_notify


def test_send_alert_appends_one_jsonl_record(tmp_path):
    log_path = tmp_path / "alerts.jsonl"

    alerts.send_alert(
        "Backup failed", "disk full", severity="error", alerts_log_path=log_path
    )

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["subject"] == "Backup failed"
    assert record["message"] == "disk full"
    assert record["severity"] == "error"
    assert "timestamp" in record


def test_send_alert_appends_without_truncating_previous_records(tmp_path):
    log_path = tmp_path / "alerts.jsonl"

    alerts.send_alert("First", "one", alerts_log_path=log_path)
    alerts.send_alert("Second", "two", alerts_log_path=log_path)

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["subject"] == "First"
    assert json.loads(lines[1])["subject"] == "Second"


def test_send_alert_creates_parent_directory(tmp_path):
    log_path = tmp_path / "nested" / "dir" / "alerts.jsonl"

    alerts.send_alert("Test", "msg", alerts_log_path=log_path)

    assert log_path.exists()


def test_send_alert_calls_desktop_notification(tmp_path, _mock_plyer_notify):
    log_path = tmp_path / "alerts.jsonl"

    alerts.send_alert("Retrain failed", "mlflow boom", alerts_log_path=log_path)

    _mock_plyer_notify.assert_called_once()
    _, kwargs = _mock_plyer_notify.call_args
    assert kwargs["title"] == "Retrain failed"
    assert kwargs["message"] == "mlflow boom"


def test_send_alert_does_not_raise_when_notification_backend_fails(
    tmp_path, monkeypatch
):
    def _raise(*_args, **_kwargs):
        raise RuntimeError("no display available")

    monkeypatch.setattr("plyer.notification.notify", _raise)
    log_path = tmp_path / "alerts.jsonl"

    alerts.send_alert("Test", "msg", alerts_log_path=log_path)  # must not raise

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1  # the durable log still got written
