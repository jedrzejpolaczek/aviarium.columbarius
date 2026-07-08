"""Unit tests for scripts/run_pipeline.py."""

from unittest.mock import MagicMock

from scripts import run_pipeline


def test_main_calls_daily_pipeline_with_config_path(monkeypatch):
    # daily_pipeline is bound directly into scripts.run_pipeline's namespace
    # via `from src.data.cards.pipelines import daily_pipeline` at module
    # level, so it must be patched there rather than at the source module.
    mock_daily_pipeline = MagicMock()
    monkeypatch.setattr(run_pipeline, "daily_pipeline", mock_daily_pipeline)

    run_pipeline.main()

    mock_daily_pipeline.assert_called_once()
    assert mock_daily_pipeline.call_args.args[0] == "configs/data_sources.yaml"
