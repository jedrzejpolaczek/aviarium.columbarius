"""Top-level pipeline orchestration for the Bronze ingestion tier.

Provides two entry points that cover the full lifecycle of the data pipeline:

    initial_pipeline  — full load, intended for first-time setup or a complete
                        rebuild of all Bronze tables.
    daily_pipeline    — incremental update, intended to run once per day to
                        upsert card data and append a daily price/meta snapshot.

Both functions read storage paths from a YAML file (configs/data_sources.yaml).
Each tier loads its own JSON config: Bronze from configs/bronze_config.json,
Silver from configs/silver_config.json, Gold from configs/gold_config.json.
"""

import asyncio
import datetime
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from src.logger import get_logger

import yaml

from src.data.cards.sources import ingesting_pipeline
from src.data.cards.storage.bronze import BronzeStorage
from src.data.cards.storage.silver import SilverStorage
from src.data.cards.storage.gold import GoldStorage
from src.data.cards.storage.health import run_health_checks


logger = get_logger(__name__)

_StageResult = tuple[str, float, str]  # (name, elapsed_s, "ok" | "error")


def _run_timed(
    name: str,
    fn: Callable[[], None],
    results: list[_StageResult],
) -> None:
    """Execute *fn* and append a timing result to *results*.

    Appends ``(name, elapsed_seconds, "ok")`` on success or
    ``(name, elapsed_seconds, "error")`` on failure, then re-raises the
    exception so the caller can decide whether to abort the pipeline.

    Args:
        name: Human-readable stage label used in the summary table.
        fn: Zero-argument callable that runs one pipeline stage.
        results: Accumulator list; one tuple is appended per call.

    Raises:
        Exception: Whatever *fn* raises, propagated unchanged after recording.
    """
    t0 = time.perf_counter()
    try:
        fn()
        results.append((name, time.perf_counter() - t0, "ok"))
    except Exception:
        results.append((name, time.perf_counter() - t0, "error"))
        raise


def _log_pipeline_summary(results: list[_StageResult], total: float) -> None:
    """Log a pytest-style summary table of all completed pipeline stages.

    Emits a single INFO record containing a bordered table with one row per
    stage (name, elapsed time, ✓/✗ status) and a total duration footer.
    The record is written by the module-level logger so it appears in both
    the console and the file handler when file logging is enabled.

    Args:
        results: Stage results produced by :func:`_run_timed`.
        total: Wall-clock seconds for the entire pipeline run.
    """
    width = 48
    sep = "═" * width
    lines = ["", sep, f"{'Pipeline Summary':^{width}}", sep]
    for name, elapsed, status in results:
        icon = "✓" if status == "ok" else "✗"
        lines.append(f"  {icon}  {name:<8}  {elapsed:>6.1f}s")
    lines += [sep, f"  Total: {total:>6.1f}s", sep, ""]
    logger.info("\n".join(lines))


def load_config(config_path: str) -> dict[str, Any]:
    """Load and return the YAML configuration file.

    Args:
        config_path: Path to the YAML config file (e.g. "configs/data_sources.yaml").

    Returns:
        Parsed configuration as a dict.

    Raises:
        FileNotFoundError: If config_path does not exist.
        ValueError: If the file contains invalid YAML.
    """
    try:
        with open(Path(config_path), "r", encoding="utf-8") as f:
            result: dict[str, Any] = yaml.safe_load(f)
            return result
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found: {config_path}") from None
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in {config_path}: {e}") from e


def initial_pipeline(config_path: str) -> None:
    """Run a full load of all Bronze tables from the configured data sources.

    Downloads (if flag=true in config), validates, and persists all sources
    via BronzeStorage.populate — dropping and recreating every Bronze table.
    Also writes the first daily snapshot rows for prices and metadata.

    Intended for first-time database setup or a complete rebuild.

    Args:
        config_path: Path to the YAML config file (e.g. "configs/data_sources.yaml").
    """
    config = load_config(config_path)
    pipeline_start = time.perf_counter()
    results: list[_StageResult] = []

    _run_timed("Bronze", lambda: initial_bronze_pipeline(config), results)
    _run_timed("Silver", lambda: initial_silver_pipeline(config), results)
    _run_timed("Gold", lambda: initial_gold_pipeline(config), results)

    _log_pipeline_summary(results, time.perf_counter() - pipeline_start)


def initial_bronze_pipeline(config: dict[str, Any]) -> None:
    """Run a full load of all Bronze tables from the configured data sources.

    Downloads (if flag=true in bronze_config), validates, and persists all
    sources via BronzeStorage.populate — dropping and recreating every Bronze
    table. Source definitions are read from the JSON file at
    config["storage"]["bronze_config_path"].

    Args:
        config: Parsed configuration dict (see configs/data_sources.yaml).
    """
    logger.info("Starting bronze initial pipeline")

    bronze_config_path = config["storage"]["bronze_config_seed_path"]
    try:
        bronze_config = json.loads(Path(bronze_config_path).read_text())
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Bronze config not found: {bronze_config_path}"
        ) from None
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {bronze_config_path}: {e}") from e
    results = asyncio.run(ingesting_pipeline(bronze_config))

    bronze_db_path = config["storage"]["bronze_duckdb_path"]

    with BronzeStorage(bronze_db_path) as storage:
        storage.populate(results)

    logger.info("Bronze initial pipeline finished")


def initial_silver_pipeline(config: dict[str, Any]) -> None:
    """Run a full load of all Silver tables from the current Bronze data.

    Reads Bronze tables and applies the config-driven transformation pipeline
    via SilverStorage.populate — dropping and recreating every Silver table.

    Args:
        config: Parsed configuration dict (see configs/data_sources.yaml).
    """
    logger.info("Starting silver initial pipeline")

    bronze_db_path = config["storage"]["bronze_duckdb_path"]
    silver_duckdb_path = config["storage"]["silver_duckdb_path"]
    silver_config_path = config["storage"]["silver_config_path"]

    with SilverStorage(
        bronze_db_path, silver_duckdb_path, silver_config_path
    ) as storage:
        storage.populate()

    logger.info("Silver initial pipeline finished")


def initial_gold_pipeline(config: dict[str, Any]) -> None:
    """Run a full load of all Gold tables from the current Silver data.

    Applies the config-driven aggregation pipeline via GoldStorage.populate —
    dropping and recreating every Gold table.

    Args:
        config: Parsed configuration dict (see configs/data_sources.yaml).
    """
    logger.info("Starting gold initial pipeline")

    silver_duckdb_path = config["storage"]["silver_duckdb_path"]
    gold_duckdb_path = config["storage"]["gold_duckdb_path"]
    gold_config_path = config["storage"]["gold_config_path"]

    with GoldStorage(silver_duckdb_path, gold_duckdb_path, gold_config_path) as storage:
        storage.populate()

    logger.info("Gold initial pipeline finished")


def daily_pipeline(config_path: str) -> None:
    """Run incremental updates across all three tiers (Bronze → Silver → Gold).

    Loads configuration from the given YAML file, then delegates to the
    per-tier daily pipeline functions in sequence.

    Args:
        config_path: Path to the YAML config file (e.g. "configs/data_sources.yaml").
    """
    config = load_config(config_path)
    pipeline_start = time.perf_counter()
    results: list[_StageResult] = []

    _run_timed("Bronze", lambda: daily_bronze_pipeline(config), results)
    _run_timed("Silver", lambda: daily_silver_pipeline(config), results)
    _run_timed("Gold", lambda: daily_gold_pipeline(config), results)

    _log_pipeline_summary(results, time.perf_counter() - pipeline_start)

    run_health_checks(
        bronze_path=config["storage"]["bronze_duckdb_path"],
        silver_path=config["storage"]["silver_duckdb_path"],
        gold_path=config["storage"]["gold_duckdb_path"],
        today=datetime.date.today(),
    )


def daily_bronze_pipeline(config: dict[str, Any]) -> None:
    """Run an incremental update of Bronze tables and append a daily snapshot.

    Downloads (if flag=true in bronze_config), validates, and upserts card data
    via BronzeStorage.daily_update. Appends one snapshot row per card to the
    price and metadata history tables. Safe to call multiple times on the
    same day — duplicate snapshots are skipped automatically. Source definitions
    are read from the JSON file at config["storage"]["bronze_config_path"].

    Intended to run once per day after initial_pipeline has been executed.

    Args:
        config: Parsed configuration dict (see configs/data_sources.yaml).
    """
    logger.info("Starting daily bronze data update pipeline")

    bronze_config_path = config["storage"]["bronze_config_path"]
    try:
        bronze_config = json.loads(Path(bronze_config_path).read_text())
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Bronze config not found: {bronze_config_path}"
        ) from None
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {bronze_config_path}: {e}") from e
    results = asyncio.run(ingesting_pipeline(bronze_config))

    total_records = sum(len(records) for records, _ in results.values())
    logger.info(
        "Ingestion complete — %d sources, %d total records; writing to DuckDB",
        len(results),
        total_records,
    )

    bronze_db_path = config["storage"]["bronze_duckdb_path"]

    with BronzeStorage(bronze_db_path) as storage:
        storage.daily_update(results)

    logger.info("Daily bronze pipeline finished")


def daily_silver_pipeline(config: dict[str, Any]) -> None:
    """Run an incremental Silver update from the current Bronze data.

    Reads all Bronze tables and applies the config-driven transformation
    pipeline via SilverStorage.update — upserts current-state Silver tables
    and appends daily snapshot rows. Safe to call multiple times on the same
    day — duplicate snapshots are skipped automatically.

    Intended to run once per day after daily_bronze_pipeline has populated
    the Bronze tier with fresh data.

    Args:
        config: Parsed configuration dict (see configs/data_sources.yaml).
    """
    logger.info("Starting daily silver data update pipeline")

    bronze_db_path = config["storage"]["bronze_duckdb_path"]
    silver_duckdb_path = config["storage"]["silver_duckdb_path"]
    silver_config_path = config["storage"]["silver_config_path"]

    with SilverStorage(
        bronze_db_path, silver_duckdb_path, silver_config_path
    ) as storage:
        storage.update()

    logger.info("Daily silver pipeline finished")


def daily_gold_pipeline(config: dict[str, Any]) -> None:
    """Run an incremental update of Gold tables from the current Silver data.

    Applies the config-driven aggregation pipeline via GoldStorage.update —
    upserts current-state Gold tables and appends daily snapshot rows. Safe to
    call multiple times on the same day — duplicate snapshots are skipped
    automatically.

    Intended to run once per day after daily_silver_pipeline has populated
    the Silver tier with fresh data.

    Args:
        config: Parsed configuration dict (see configs/data_sources.yaml).
    """
    logger.info("Starting daily gold data update pipeline")

    silver_duckdb_path = config["storage"]["silver_duckdb_path"]
    gold_duckdb_path = config["storage"]["gold_duckdb_path"]
    gold_config_path = config["storage"]["gold_config_path"]

    with GoldStorage(silver_duckdb_path, gold_duckdb_path, gold_config_path) as storage:
        storage.update()

    logger.info("Daily gold pipeline finished")
