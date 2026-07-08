"""Unit tests for scripts/run_pipeline.py."""

from unittest.mock import MagicMock

from scripts import run_pipeline


def test_main_calls_daily_pipeline_with_config_path(monkeypatch):
    # daily_pipeline is bound directly into scripts.run_pipeline's namespace
    # via `from src.data.cards.pipelines import daily_pipeline` at module
    # level, so it must be patched there rather than at the source module.
    mock_daily_pipeline = MagicMock()
    monkeypatch.setattr(run_pipeline, "daily_pipeline", mock_daily_pipeline)
    # setup_logging is also bound directly into scripts.run_pipeline's
    # namespace via `from src.logger import setup_logging` at module level.
    # Mock it to avoid writing real timestamped log files into logs/ and
    # mutating the global logging root logger as a side effect of main().
    monkeypatch.setattr(run_pipeline, "setup_logging", MagicMock())

    run_pipeline.main()

    mock_daily_pipeline.assert_called_once()
    assert mock_daily_pipeline.call_args.args[0] == "configs/data_sources.yaml"
