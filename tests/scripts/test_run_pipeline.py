"""Unit tests for scripts/run_pipeline.py."""

import json
from unittest.mock import MagicMock

from scripts import run_pipeline
from src.data.cards.pipelines import PipelineOutcome
from src.data.cards.storage.health import CheckResult


def _healthy() -> PipelineOutcome:
    return PipelineOutcome(
        health=[CheckResult("silver_prices_history freshness", "silver", "PASS", "5")],
        failed_sources=[],
    )


def _mock_pipeline(outcome: PipelineOutcome) -> MagicMock:
    return MagicMock(return_value=outcome)


def test_main_calls_daily_pipeline_with_config_path(tmp_path, monkeypatch):
    mock_daily_pipeline = _mock_pipeline(_healthy())
    monkeypatch.setattr(run_pipeline, "daily_pipeline", mock_daily_pipeline)
    monkeypatch.setattr(run_pipeline, "setup_logging", MagicMock())
    monkeypatch.setattr(run_pipeline, "STATUS_PATH", tmp_path / "status.json")

    exit_code = run_pipeline.main()

    mock_daily_pipeline.assert_called_once()
    assert mock_daily_pipeline.call_args.args[0] == "configs/data_sources.yaml"
    assert exit_code == 0


def test_main_writes_success_status(tmp_path, monkeypatch):
    monkeypatch.setattr(run_pipeline, "daily_pipeline", _mock_pipeline(_healthy()))
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
    monkeypatch.setattr(run_pipeline, "daily_pipeline", _mock_pipeline(_healthy()))
    monkeypatch.setattr(run_pipeline, "setup_logging", MagicMock())
    monkeypatch.setattr(run_pipeline, "STATUS_PATH", tmp_path / "status.json")
    mock_send_alert = MagicMock()
    monkeypatch.setattr(run_pipeline, "send_alert", mock_send_alert)

    run_pipeline.main()

    mock_send_alert.assert_not_called()


# ---------------------------------------------------------------------------
# Degraded runs — the pipeline finished, but the day's data is incomplete.
# This is the state a 36-day Scryfall outage sat in while the status file
# kept reporting "success".
# ---------------------------------------------------------------------------


def test_main_reports_degraded_on_failed_health_check(tmp_path, monkeypatch):
    outcome = PipelineOutcome(
        health=[
            CheckResult("silver_prices_history freshness", "silver", "PASS", "5 rows"),
            CheckResult(
                "bronze_scryfall_prices_history freshness",
                "bronze",
                "FAIL",
                "no rows for 2026-08-10",
            ),
        ],
        failed_sources=[],
    )
    monkeypatch.setattr(run_pipeline, "daily_pipeline", _mock_pipeline(outcome))
    monkeypatch.setattr(run_pipeline, "setup_logging", MagicMock())
    status_path = tmp_path / "status.json"
    monkeypatch.setattr(run_pipeline, "STATUS_PATH", status_path)
    mock_send_alert = MagicMock()
    monkeypatch.setattr(run_pipeline, "send_alert", mock_send_alert)

    exit_code = run_pipeline.main()

    assert exit_code == 1
    status = json.loads(status_path.read_text())
    assert status["result"] == "degraded"
    assert "bronze_scryfall_prices_history freshness" in status["detail"]
    assert status["failed_checks"]
    mock_send_alert.assert_called_once()


def test_main_reports_degraded_on_source_with_no_records(tmp_path, monkeypatch):
    outcome = PipelineOutcome(health=[], failed_sources=["scryfall"])
    monkeypatch.setattr(run_pipeline, "daily_pipeline", _mock_pipeline(outcome))
    monkeypatch.setattr(run_pipeline, "setup_logging", MagicMock())
    status_path = tmp_path / "status.json"
    monkeypatch.setattr(run_pipeline, "STATUS_PATH", status_path)
    mock_send_alert = MagicMock()
    monkeypatch.setattr(run_pipeline, "send_alert", mock_send_alert)

    exit_code = run_pipeline.main()

    assert exit_code == 1
    status = json.loads(status_path.read_text())
    assert status["result"] == "degraded"
    assert status["failed_sources"] == ["scryfall"]
    mock_send_alert.assert_called_once()
    assert "scryfall" in mock_send_alert.call_args.args[1]


def test_main_overwrites_stale_success_status_on_degraded_run(tmp_path, monkeypatch):
    """The status file must never be left describing an earlier, healthier run.

    Regression guard: run_health_checks' SystemExit used to abort main()
    before _write_status(), so a failing day silently inherited the previous
    day's "success".
    """
    status_path = tmp_path / "status.json"
    status_path.write_text(
        json.dumps({"checked_at": "2026-08-09T08:00:00+00:00", "result": "success"})
    )
    outcome = PipelineOutcome(
        health=[CheckResult("silver freshness", "silver", "FAIL", "no rows")],
        failed_sources=[],
    )
    monkeypatch.setattr(run_pipeline, "daily_pipeline", _mock_pipeline(outcome))
    monkeypatch.setattr(run_pipeline, "setup_logging", MagicMock())
    monkeypatch.setattr(run_pipeline, "STATUS_PATH", status_path)
    monkeypatch.setattr(run_pipeline, "send_alert", MagicMock())

    run_pipeline.main()

    assert json.loads(status_path.read_text())["result"] == "degraded"


# ---------------------------------------------------------------------------
# Heartbeat — dead-man's-switch for the schedule itself.
# ---------------------------------------------------------------------------


def test_heartbeat_skipped_when_env_unset(tmp_path, monkeypatch):
    monkeypatch.delenv("HEARTBEAT_URL", raising=False)
    mock_get = MagicMock()
    monkeypatch.setattr(run_pipeline.httpx, "get", mock_get)
    monkeypatch.setattr(run_pipeline, "daily_pipeline", _mock_pipeline(_healthy()))
    monkeypatch.setattr(run_pipeline, "setup_logging", MagicMock())
    monkeypatch.setattr(run_pipeline, "STATUS_PATH", tmp_path / "status.json")

    run_pipeline.main()

    mock_get.assert_not_called()


def test_heartbeat_pinged_on_success(tmp_path, monkeypatch):
    monkeypatch.setenv("HEARTBEAT_URL", "https://hb.example/abc")
    mock_get = MagicMock()
    monkeypatch.setattr(run_pipeline.httpx, "get", mock_get)
    monkeypatch.setattr(run_pipeline, "daily_pipeline", _mock_pipeline(_healthy()))
    monkeypatch.setattr(run_pipeline, "setup_logging", MagicMock())
    monkeypatch.setattr(run_pipeline, "STATUS_PATH", tmp_path / "status.json")

    run_pipeline.main()

    assert mock_get.call_args.args[0] == "https://hb.example/abc"


def test_heartbeat_pings_fail_suffix_on_degraded(tmp_path, monkeypatch):
    monkeypatch.setenv("HEARTBEAT_URL", "https://hb.example/abc")
    mock_get = MagicMock()
    monkeypatch.setattr(run_pipeline.httpx, "get", mock_get)
    outcome = PipelineOutcome(health=[], failed_sources=["scryfall"])
    monkeypatch.setattr(run_pipeline, "daily_pipeline", _mock_pipeline(outcome))
    monkeypatch.setattr(run_pipeline, "setup_logging", MagicMock())
    monkeypatch.setattr(run_pipeline, "STATUS_PATH", tmp_path / "status.json")
    monkeypatch.setattr(run_pipeline, "send_alert", MagicMock())

    run_pipeline.main()

    assert mock_get.call_args.args[0] == "https://hb.example/abc/fail"
