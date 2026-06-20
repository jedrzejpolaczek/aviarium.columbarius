from unittest.mock import MagicMock, patch

import pytest
import yaml

from src.data.cards.pipelines import (
    _StageResult,
    _log_pipeline_summary,
    _run_timed,
    daily_bronze_pipeline,
    daily_silver_pipeline,
    initial_pipeline,
    load_config,
)

MINIMAL_PIPELINE_CONFIG = {
    "storage": {
        "bronze_duckdb_path": ":memory:",
        "silver_duckdb_path": ":memory:",
        "gold_duckdb_path": ":memory:",
        "silver_config_path": "configs/silver_config.json",
        "gold_config_path": "configs/gold_config.json",
        "bronze_config_path": "configs/bronze_config.json",
        "bronze_config_seed_path": "configs/bronze_config_seed.json",
    },
}


def test_load_config_storage_is_dict(tmp_path):
    """After the fix, config['storage'] must be a dict keyed by name, not a list."""
    yaml_content = """storage:
  bronze_duckdb_path: "data/bronze/cards.duckdb"
  silver_duckdb_path: "data/silver/cards.duckdb"
  gold_duckdb_path: "data/gold/cards.duckdb"
  silver_config_path: "configs/silver_config.json"
  gold_config_path: "configs/gold_config.json"
  bronze_config_path: "configs/bronze_config.json"
  bronze_config_seed_path: "configs/bronze_config_seed.json"
"""
    config_file = tmp_path / "data_sources.yaml"
    config_file.write_text(yaml_content)
    config = load_config(str(config_file))
    storage = config["storage"]
    assert isinstance(storage, dict), f"storage must be a dict, got {type(storage)}"
    assert storage["bronze_duckdb_path"] == "data/bronze/cards.duckdb"
    assert storage["bronze_config_seed_path"] == "configs/bronze_config_seed.json"
    assert storage["gold_config_path"] == "configs/gold_config.json"


@pytest.fixture
def config_path(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.dump(MINIMAL_PIPELINE_CONFIG))
    return str(path)


@pytest.fixture
def config():
    return MINIMAL_PIPELINE_CONFIG


def _mock_storage_ctx(mock_class):
    """Wire a MagicMock class to work as a context manager returning an instance."""
    instance = MagicMock()
    mock_class.return_value.__enter__ = MagicMock(return_value=instance)
    mock_class.return_value.__exit__ = MagicMock(return_value=False)
    return instance


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------


class TestLoadConfig:
    def test_returns_parsed_yaml(self, tmp_path):
        data = {"sources": [], "key": "value"}
        path = tmp_path / "cfg.yaml"
        path.write_text(yaml.dump(data))
        result = load_config(str(path))
        assert result["key"] == "value"
        assert result["sources"] == []

    def test_missing_file_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_config(str(tmp_path / "nonexistent.yaml"))


# ---------------------------------------------------------------------------
# initial_pipeline
# ---------------------------------------------------------------------------


class TestInitialPipeline:
    def test_calls_bronze_populate_then_silver_and_gold_populate(self, config_path):
        mock_results: dict[str, tuple[list, list]] = {"scryfall": ([], [])}
        with (
            patch(
                "src.data.cards.pipelines.ingesting_pipeline", return_value=mock_results
            ),
            patch("src.data.cards.pipelines.BronzeStorage") as MockBronze,
            patch("src.data.cards.pipelines.SilverStorage") as MockSilver,
            patch("src.data.cards.pipelines.GoldStorage") as MockGold,
        ):
            bronze = _mock_storage_ctx(MockBronze)
            silver = _mock_storage_ctx(MockSilver)
            gold = _mock_storage_ctx(MockGold)

            initial_pipeline(config_path)

            bronze.populate.assert_called_once_with(mock_results)
            silver.populate.assert_called_once()
            gold.populate.assert_called_once()

    def test_passes_correct_paths_to_storage(self, config_path):
        with (
            patch("src.data.cards.pipelines.ingesting_pipeline", return_value={}),
            patch("src.data.cards.pipelines.BronzeStorage") as MockBronze,
            patch("src.data.cards.pipelines.SilverStorage") as MockSilver,
            patch("src.data.cards.pipelines.GoldStorage") as MockGold,
        ):
            _mock_storage_ctx(MockBronze)
            _mock_storage_ctx(MockSilver)
            _mock_storage_ctx(MockGold)

            initial_pipeline(config_path)

            MockBronze.assert_called_once_with(":memory:")
            MockSilver.assert_called_once_with(
                ":memory:", ":memory:", "configs/silver_config.json"
            )
            MockGold.assert_called_once_with(
                ":memory:", ":memory:", "configs/gold_config.json"
            )


# ---------------------------------------------------------------------------
# daily_bronze_pipeline
# ---------------------------------------------------------------------------


class TestDailyBronzePipeline:
    def test_calls_bronze_daily_update(self, config):
        mock_results: dict[str, tuple[list, list]] = {}
        with (
            patch(
                "src.data.cards.pipelines.ingesting_pipeline", return_value=mock_results
            ),
            patch("src.data.cards.pipelines.BronzeStorage") as MockBronze,
        ):
            bronze = _mock_storage_ctx(MockBronze)

            daily_bronze_pipeline(config)

            bronze.daily_update.assert_called_once_with(mock_results)

    def test_passes_correct_paths_to_storage(self, config):
        with (
            patch("src.data.cards.pipelines.ingesting_pipeline", return_value={}),
            patch("src.data.cards.pipelines.BronzeStorage") as MockBronze,
        ):
            _mock_storage_ctx(MockBronze)

            daily_bronze_pipeline(config)

            MockBronze.assert_called_once_with(":memory:")


# ---------------------------------------------------------------------------
# daily_silver_pipeline
# ---------------------------------------------------------------------------


class TestDailySilverPipeline:
    def test_calls_silver_update(self, config):
        with patch("src.data.cards.pipelines.SilverStorage") as MockSilver:
            silver = _mock_storage_ctx(MockSilver)

            daily_silver_pipeline(config)

            silver.update.assert_called_once()

    def test_passes_correct_paths_to_silver_storage(self, config):
        with patch("src.data.cards.pipelines.SilverStorage") as MockSilver:
            _mock_storage_ctx(MockSilver)

            daily_silver_pipeline(config)

            MockSilver.assert_called_once_with(
                ":memory:", ":memory:", "configs/silver_config.json"
            )


# ---------------------------------------------------------------------------
# _run_timed
# ---------------------------------------------------------------------------


class TestRunTimed:
    def test_appends_ok_result_on_success(self) -> None:
        results: list[_StageResult] = []
        _run_timed("Bronze", lambda: None, results)
        assert len(results) == 1
        name, elapsed, status = results[0]
        assert name == "Bronze"
        assert status == "ok"
        assert elapsed >= 0

    def test_appends_error_result_on_failure(self) -> None:
        results: list[_StageResult] = []
        with pytest.raises(ValueError):
            _run_timed("Silver", lambda: (_ for _ in ()).throw(ValueError("boom")), results)
        assert len(results) == 1
        name, elapsed, status = results[0]
        assert name == "Silver"
        assert status == "error"
        assert elapsed >= 0

    def test_reraises_exception(self) -> None:
        results: list[_StageResult] = []
        with pytest.raises(RuntimeError, match="stage failed"):
            _run_timed("Gold", lambda: (_ for _ in ()).throw(RuntimeError("stage failed")), results)

    def test_elapsed_is_non_negative(self) -> None:
        results: list[_StageResult] = []
        _run_timed("Bronze", lambda: None, results)
        assert results[0][1] >= 0

    def test_multiple_stages_accumulate(self) -> None:
        results: list[_StageResult] = []
        _run_timed("Bronze", lambda: None, results)
        _run_timed("Silver", lambda: None, results)
        assert len(results) == 2
        assert results[0][0] == "Bronze"
        assert results[1][0] == "Silver"


# ---------------------------------------------------------------------------
# _log_pipeline_summary
# ---------------------------------------------------------------------------


class TestLogPipelineSummary:
    def test_logs_at_info_level(self, caplog) -> None:
        import logging
        with caplog.at_level(logging.INFO, logger="src.data.cards.pipelines"):
            _log_pipeline_summary([("Bronze", 1.5, "ok")], total=1.5)
        assert any(r.levelno == logging.INFO for r in caplog.records)

    def test_summary_contains_stage_name(self, caplog) -> None:
        import logging
        with caplog.at_level(logging.INFO, logger="src.data.cards.pipelines"):
            _log_pipeline_summary([("Bronze", 1.5, "ok")], total=1.5)
        assert "Bronze" in caplog.text

    def test_summary_contains_checkmark_for_ok(self, caplog) -> None:
        import logging
        with caplog.at_level(logging.INFO, logger="src.data.cards.pipelines"):
            _log_pipeline_summary([("Bronze", 1.5, "ok")], total=1.5)
        assert "✓" in caplog.text

    def test_summary_contains_cross_for_error(self, caplog) -> None:
        import logging
        with caplog.at_level(logging.INFO, logger="src.data.cards.pipelines"):
            _log_pipeline_summary([("Silver", 0.3, "error")], total=0.3)
        assert "✗" in caplog.text

    def test_summary_contains_total(self, caplog) -> None:
        import logging
        with caplog.at_level(logging.INFO, logger="src.data.cards.pipelines"):
            _log_pipeline_summary([("Bronze", 2.0, "ok"), ("Silver", 1.0, "ok")], total=3.0)
        assert "Total" in caplog.text

    def test_all_stages_appear_in_summary(self, caplog) -> None:
        import logging
        stages: list[_StageResult] = [
            ("Bronze", 1.0, "ok"),
            ("Silver", 2.0, "ok"),
            ("Gold", 3.0, "error"),
        ]
        with caplog.at_level(logging.INFO, logger="src.data.cards.pipelines"):
            _log_pipeline_summary(stages, total=6.0)
        for name in ("Bronze", "Silver", "Gold"):
            assert name in caplog.text
