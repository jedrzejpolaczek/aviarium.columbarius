"""Fixtures shared by the scripts/ entry-point tests.

These tests call each script's ``main()`` directly, which means every
side effect a real invocation has — writing a timestamped log file,
appending to the durable alert log — happens inside the test run unless it
is redirected. Two of those side effects used to land in the project's real
``logs/`` directory (see the autouse fixtures below).
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from scripts import check_and_retrain, run_pipeline


@pytest.fixture(autouse=True)
def _no_real_log_files(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stop main() from creating real logs/pipeline_*.log files.

    setup_logging is bound into each script's namespace by a module-level
    ``from src.logger import setup_logging``, so it has to be patched per
    module rather than at the source. Most tests already did this by hand;
    tests/scripts/test_check_and_retrain.py did not, which is why the repo
    accumulated log files timestamped with pytest runs rather than pipeline
    runs (e.g. three logs/pipeline_2026-09-25_12-52-4*.log within 4 seconds).
    """
    for module in (check_and_retrain, run_pipeline):
        monkeypatch.setattr(module, "setup_logging", MagicMock(return_value=None))


@pytest.fixture(autouse=True)
def _isolate_alert_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect the durable alert log into tmp_path.

    See tests/conftest.py for the same fixture applied suite-wide; this one
    exists because the scripts tests are the heaviest producers of alerts.
    """
    from src.monitoring import alerts

    monkeypatch.setattr(alerts, "ALERTS_LOG_PATH", tmp_path / "alerts.jsonl")
