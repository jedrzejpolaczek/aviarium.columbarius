"""Unit tests for scripts/check_health.py."""

from unittest.mock import MagicMock

import pytest

from scripts import check_health
from src.data.cards.storage.health import CheckResult


def test_main_calls_run_health_checks_with_config_paths(monkeypatch):
    # load_config and run_health_checks are both bound directly into
    # scripts.check_health's namespace via module-level `from ... import`
    # statements, so they must be patched there rather than at their
    # source modules.
    fake_config = {
        "storage": {
            "bronze_duckdb_path": "bronze.duckdb",
            "silver_duckdb_path": "silver.duckdb",
            "gold_duckdb_path": "gold.duckdb",
        }
    }
    monkeypatch.setattr(check_health, "load_config", lambda path: fake_config)
    mock_run_health_checks = MagicMock(return_value=[])
    monkeypatch.setattr(check_health, "run_health_checks", mock_run_health_checks)
    # setup_logging is also bound directly into scripts.check_health's
    # namespace via `from src.logger import setup_logging` at module level.
    # Mock it to avoid writing real timestamped log files into logs/ and
    # mutating the global logging root logger as a side effect of main().
    monkeypatch.setattr(check_health, "setup_logging", MagicMock())

    check_health.main()

    mock_run_health_checks.assert_called_once()
    call_kwargs = mock_run_health_checks.call_args.kwargs
    assert call_kwargs["bronze_path"] == "bronze.duckdb"
    assert call_kwargs["silver_path"] == "silver.duckdb"
    assert call_kwargs["gold_path"] == "gold.duckdb"


def _patched_main(monkeypatch, results):
    fake_config = {
        "storage": {
            "bronze_duckdb_path": "bronze.duckdb",
            "silver_duckdb_path": "silver.duckdb",
            "gold_duckdb_path": "gold.duckdb",
        }
    }
    monkeypatch.setattr(check_health, "load_config", lambda path: fake_config)
    monkeypatch.setattr(
        check_health, "run_health_checks", MagicMock(return_value=results)
    )
    monkeypatch.setattr(check_health, "setup_logging", MagicMock())


def test_main_exits_1_when_any_check_fails(monkeypatch):
    """The exit-on-FAIL decision moved here from run_health_checks().

    It stays a CLI-level concern: daily_pipeline() needs the same results as
    data so it can record a degraded run instead of being torn down mid-run.
    """
    _patched_main(
        monkeypatch,
        [
            CheckResult(
                "bronze_scryfall_prices_history freshness", "bronze", "PASS", ""
            ),
            CheckResult("silver_prices_history freshness", "silver", "FAIL", "no rows"),
        ],
    )

    with pytest.raises(SystemExit) as exc:
        check_health.main()

    assert exc.value.code == 1


def test_main_returns_normally_when_only_warnings(monkeypatch):
    _patched_main(
        monkeypatch,
        [
            CheckResult("schema drift", "bronze", "WARN", "1 new combo"),
            CheckResult("silver_prices_history freshness", "silver", "PASS", "5 rows"),
        ],
    )

    check_health.main()
