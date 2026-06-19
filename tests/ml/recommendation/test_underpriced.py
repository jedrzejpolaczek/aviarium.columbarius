import duckdb
import numpy as np
import pandas as pd
import pytest

from src.ml.recommendation.underpriced import (
    TIER1_FLAG_THRESHOLD,
    TIER2_FLAG_THRESHOLD,
    backtest_underpriced,
    flag_underpriced,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_predictions_df(
    n_tier1: int = 10,
    n_tier2: int = 5,
    n_tier3: int = 2,
    confidence_multiplier: float = 1.5,
    seed: int = 0,
) -> pd.DataFrame:
    """DataFrame with cards across all three tiers, all flagged as underpriced."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_tier1):
        eur = rng.uniform(1.0, 50.0)
        rows.append(
            {
                "uuid": f"t1_{i}",
                "name": f"T1 Card {i}",
                "eur": eur,
                "predicted_eur": eur * confidence_multiplier,
            }
        )
    for i in range(n_tier2):
        eur = rng.uniform(100.0, 500.0)
        rows.append(
            {
                "uuid": f"t2_{i}",
                "name": f"T2 Card {i}",
                "eur": eur,
                "predicted_eur": eur * confidence_multiplier,
            }
        )
    for i in range(n_tier3):
        eur = rng.uniform(1100.0, 3000.0)
        rows.append(
            {
                "uuid": f"t3_{i}",
                "name": f"T3 Card {i}",
                "eur": eur,
                "predicted_eur": eur * confidence_multiplier,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def predictions_df():
    return _make_predictions_df()


@pytest.fixture
def backtest_conn():
    """In-memory DuckDB with two snapshots for backtest tests."""
    conn = duckdb.connect()
    conn.execute("""
        CREATE TABLE gold_price_features (
            uuid VARCHAR, snapshot_date DATE, eur DOUBLE
        )
    """)
    for uid, price_flag, price_check in [
        ("card_a", 10.0, 12.0),  # +20% → appreciated
        ("card_b", 20.0, 21.0),  # +5%  → did not appreciate (< 10%)
        ("card_c", 50.0, 60.0),  # +20% → appreciated
    ]:
        conn.execute(
            "INSERT INTO gold_price_features VALUES (?, ?, ?)",
            [uid, "2026-01-01", price_flag],
        )
        conn.execute(
            "INSERT INTO gold_price_features VALUES (?, ?, ?)",
            [uid, "2026-02-01", price_check],
        )
    yield conn
    conn.close()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def test_tier1_threshold_is_float():
    assert isinstance(TIER1_FLAG_THRESHOLD, float)


def test_tier2_threshold_is_float():
    assert isinstance(TIER2_FLAG_THRESHOLD, float)


def test_tier1_threshold_above_one():
    assert TIER1_FLAG_THRESHOLD > 1.0


# ---------------------------------------------------------------------------
# flag_underpriced()
# ---------------------------------------------------------------------------


def test_flag_underpriced_returns_dataframe(predictions_df):
    result = flag_underpriced(predictions_df)
    assert isinstance(result, pd.DataFrame)


def test_flag_underpriced_has_is_underpriced_column(predictions_df):
    result = flag_underpriced(predictions_df)
    assert "is_underpriced" in result.columns


def test_flag_underpriced_has_confidence_column(predictions_df):
    result = flag_underpriced(predictions_df)
    assert "confidence" in result.columns


def test_flag_underpriced_has_reason_column(predictions_df):
    result = flag_underpriced(predictions_df)
    assert "reason" in result.columns


def test_flag_underpriced_tier1_flagged_when_above_threshold():
    df = pd.DataFrame(
        {
            "uuid": ["x"],
            "name": ["X"],
            "eur": [5.0],
            "predicted_eur": [5.0 * 1.5],
        }
    )
    result = flag_underpriced(df)
    assert result.loc[0, "is_underpriced"]


def test_flag_underpriced_tier1_not_flagged_when_below_threshold():
    df = pd.DataFrame(
        {
            "uuid": ["x"],
            "name": ["X"],
            "eur": [5.0],
            "predicted_eur": [5.0 * 1.1],
        }
    )
    result = flag_underpriced(df)
    assert not result.loc[0, "is_underpriced"]


def test_flag_underpriced_tier2_flagged_when_above_threshold():
    df = pd.DataFrame(
        {
            "uuid": ["x"],
            "name": ["X"],
            "eur": [200.0],
            "predicted_eur": [200.0 * 1.5],
        }
    )
    result = flag_underpriced(df)
    assert result.loc[0, "is_underpriced"]


def test_flag_underpriced_tier3_never_flagged():
    df = pd.DataFrame(
        {
            "uuid": ["x"],
            "name": ["X"],
            "eur": [1500.0],
            "predicted_eur": [1500.0 * 2.0],
        }
    )
    result = flag_underpriced(df)
    assert not result.loc[0, "is_underpriced"]


def test_flag_underpriced_reason_non_empty_when_flagged():
    df = pd.DataFrame(
        {
            "uuid": ["x"],
            "name": ["X"],
            "eur": [5.0],
            "predicted_eur": [5.0 * 1.5],
        }
    )
    result = flag_underpriced(df)
    assert result.loc[0, "reason"] != ""


def test_flag_underpriced_reason_empty_when_not_flagged():
    df = pd.DataFrame(
        {
            "uuid": ["x"],
            "name": ["X"],
            "eur": [5.0],
            "predicted_eur": [5.0 * 1.0],
        }
    )
    result = flag_underpriced(df)
    assert result.loc[0, "reason"] == ""


def test_flag_underpriced_sorted_by_confidence_descending(predictions_df):
    result = flag_underpriced(predictions_df)
    assert list(result["confidence"]) == sorted(result["confidence"], reverse=True)


def test_flag_underpriced_confidence_equals_ratio():
    df = pd.DataFrame(
        {
            "uuid": ["x"],
            "name": ["X"],
            "eur": [10.0],
            "predicted_eur": [15.0],
        }
    )
    result = flag_underpriced(df)
    assert abs(float(result.loc[0, "confidence"]) - 1.5) < 1e-9  # type: ignore[arg-type]


def test_flag_underpriced_does_not_modify_input(predictions_df):
    original_cols = set(predictions_df.columns)
    flag_underpriced(predictions_df)
    assert set(predictions_df.columns) == original_cols


def test_flag_underpriced_reason_contains_percentage():
    df = pd.DataFrame(
        {
            "uuid": ["x"],
            "name": ["X"],
            "eur": [10.0],
            "predicted_eur": [15.0],
        }
    )
    result = flag_underpriced(df)
    # confidence = 1.5 → +50%
    assert "50%" in str(result.loc[0, "reason"])


# ---------------------------------------------------------------------------
# backtest_underpriced()
# ---------------------------------------------------------------------------


def test_backtest_returns_dict(backtest_conn):
    result = backtest_underpriced(
        backtest_conn, "2026-01-01", "2026-02-01", ["card_a", "card_b", "card_c"]
    )
    assert isinstance(result, dict)


def test_backtest_has_total_flagged_key(backtest_conn):
    result = backtest_underpriced(
        backtest_conn, "2026-01-01", "2026-02-01", ["card_a", "card_b", "card_c"]
    )
    assert "total_flagged" in result


def test_backtest_has_hit_rate_key(backtest_conn):
    result = backtest_underpriced(
        backtest_conn, "2026-01-01", "2026-02-01", ["card_a", "card_b", "card_c"]
    )
    assert "hit_rate" in result


def test_backtest_total_flagged_correct(backtest_conn):
    result = backtest_underpriced(
        backtest_conn, "2026-01-01", "2026-02-01", ["card_a", "card_b", "card_c"]
    )
    assert result["total_flagged"] == 3


def test_backtest_appreciated_count_correct(backtest_conn):
    # card_a: +20%, card_c: +20% → 2 appreciated; card_b: +5% → below threshold
    result = backtest_underpriced(
        backtest_conn,
        "2026-01-01",
        "2026-02-01",
        ["card_a", "card_b", "card_c"],
        appreciation_threshold=0.10,
    )
    assert result["appreciated"] == 2


def test_backtest_hit_rate_correct(backtest_conn):
    result = backtest_underpriced(
        backtest_conn,
        "2026-01-01",
        "2026-02-01",
        ["card_a", "card_b", "card_c"],
        appreciation_threshold=0.10,
    )
    assert abs(float(result["hit_rate"]) - 2 / 3) < 1e-9


def test_backtest_empty_uuids_returns_zero_hit_rate(backtest_conn):
    result = backtest_underpriced(backtest_conn, "2026-01-01", "2026-02-01", [])
    assert result["hit_rate"] == 0.0
    assert result["total_flagged"] == 0
