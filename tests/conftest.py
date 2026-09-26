"""Shared fixtures for the top-level test suite.

Fixtures here are visible to every test package (unlike the
directory-scoped conftest.py files under tests/app/ and
tests/data/cards/storage/), so this file is reserved for fixtures needed
across multiple, otherwise-unrelated test directories.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import duckdb
import pytest


@pytest.fixture(autouse=True)
def _no_real_log_files(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stop any script's main() from creating real logs/pipeline_*.log files.

    All six scripts bind setup_logging into their own namespace with a
    module-level ``from src.logger import setup_logging``, so patching the
    source has no effect — each binding has to be replaced. Rather than listing
    them (the list went stale the moment a seventh script appeared), this walks
    the already-imported ``scripts.*`` modules and patches whatever it finds.

    Why it matters beyond tidiness: setup_logging prunes ``log_dir`` to
    ``keep_last_logs=90`` timestamped groups. Test-created logs therefore do not
    grow the directory — they *evict real pipeline logs* to stay under the cap.
    A stable file count is exactly what that looks like, which is how this went
    unnoticed once already.
    """
    for name, module in list(sys.modules.items()):
        if name.startswith("scripts") and hasattr(module, "setup_logging"):
            monkeypatch.setattr(module, "setup_logging", MagicMock(return_value=None))


@pytest.fixture(autouse=True)
def _isolate_alert_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep every test's send_alert() calls out of the real logs/alerts.jsonl.

    send_alert resolves its target from the module-level ALERTS_LOG_PATH at
    call time, so redirecting that constant is enough to cover callers that
    do not pass alerts_log_path explicitly.

    Without this, the production alert log is where test noise goes: all 190
    records it held on 2026-09-25 were written by pytest (124 of them
    "Monitor: Gold DB missing" naming pytest tmp paths). The log is documented
    as the one channel that always succeeds and is meant to be replayable —
    it cannot be that and a test scratch file at the same time.
    """
    from src.monitoring import alerts

    monkeypatch.setattr(alerts, "ALERTS_LOG_PATH", tmp_path / "alerts.jsonl")


@pytest.fixture
def tiny_gold_conn():
    """In-memory DuckDB connection pre-populated with a 2-card/2-snapshot
    gold_price_features + gold_card_features dataset.

    Deliberately tiny: it trips retrain()'s InsufficientDataError fallback
    instead of running a full walk-forward CV, which is what makes it
    usable for real-MLflow integration tests (tests/monitoring/
    test_retrain_integration.py, tests/scripts/test_check_and_retrain.py)
    without needing 50+ days of synthetic price history.
    """
    con = duckdb.connect(":memory:")
    con.execute("""
        CREATE TABLE gold_price_features AS
        SELECT * FROM (VALUES
            ('uuid-1', '2026-06-01', 1.5, 100.0, NULL),
            ('uuid-1', '2026-06-08', 1.8, 100.0, NULL),
            ('uuid-2', '2026-06-01', 0.3, 200.0, NULL),
            ('uuid-2', '2026-06-08', 0.4, 200.0, NULL)
        ) AS t(uuid, snapshot_date, eur, edhrec_rank, foil_premium)
    """)
    # edhrec_saltiness is required here (not in gold_price_features) because
    # IMPUTE_MEDIAN_COLS in src/ml/features/pipeline.py expects it, and in
    # production it is sourced from gold_card_features (see
    # GoldFeatureBuilders.build_card_features in
    # src/data/cards/storage/gold/features.py) — build_inference_features()
    # merges lag_df and card_df on uuid, so it must be present post-merge.
    con.execute("""
        CREATE TABLE gold_card_features AS
        SELECT * FROM (VALUES
            ('uuid-1', 'common', 3, 2.0, 1, false, false, true, NULL),
            ('uuid-2', 'rare',   1, 1.0, 1, false, false, true, NULL)
        ) AS t(uuid, rarity, print_count, mana_value, format_count,
                is_reserved, is_legendary, is_commander_legal, edhrec_saltiness)
    """)
    yield con
    con.close()
