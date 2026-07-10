"""Shared fixtures for the top-level test suite.

Fixtures here are visible to every test package (unlike the
directory-scoped conftest.py files under tests/app/ and
tests/data/cards/storage/), so this file is reserved for fixtures needed
across multiple, otherwise-unrelated test directories.
"""

import duckdb
import pytest


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
