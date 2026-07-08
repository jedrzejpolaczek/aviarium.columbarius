"""Unit tests for scripts/check_health.py."""

from unittest.mock import MagicMock

from scripts import check_health


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

    check_health.main()

    mock_run_health_checks.assert_called_once()
    call_kwargs = mock_run_health_checks.call_args.kwargs
    assert call_kwargs["bronze_path"] == "bronze.duckdb"
    assert call_kwargs["silver_path"] == "silver.duckdb"
    assert call_kwargs["gold_path"] == "gold.duckdb"
