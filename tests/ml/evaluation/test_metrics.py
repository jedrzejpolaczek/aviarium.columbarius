import numpy as np
import pandas as pd
import pytest

from src.ml.evaluation.metrics import (
    evaluate_per_tier,
    mae,
    mape,
)


# ---------------------------------------------------------------------------
# mae()
# ---------------------------------------------------------------------------


def test_mae_perfect_prediction():
    y = np.array([0.1, 0.2, 0.3])
    assert mae(y, y) == 0.0


def test_mae_known_values():
    # |0.1-0.2| + |0.3-0.1| + |0.5-0.4| = 0.1 + 0.2 + 0.1 = 0.4 / 3 ≈ 0.1333
    y_true = np.array([0.1, 0.3, 0.5])
    y_pred = np.array([0.2, 0.1, 0.4])
    assert abs(mae(y_true, y_pred) - (0.1 + 0.2 + 0.1) / 3) < 1e-9


def test_mae_returns_float():
    assert isinstance(mae(np.array([0.1]), np.array([0.2])), float)


def test_mae_is_symmetric():
    # MAE(a, b) == MAE(b, a) — error direction does not matter
    y_true = np.array([0.1, 0.5, -0.2])
    y_pred = np.array([0.3, 0.2, 0.1])
    assert mae(y_true, y_pred) == mae(y_pred, y_true)


# ---------------------------------------------------------------------------
# mape()
# ---------------------------------------------------------------------------


def test_mape_perfect_prediction():
    y = np.array([0.1, 0.5, 1.0])
    assert mape(y, y) == 0.0


def test_mape_returns_percent_not_fraction():
    # 50% error on y_true=0.1, pred=0.15 → MAPE should be ~50.0, not ~0.5
    y_true = np.array([0.1])
    y_pred = np.array([0.15])
    result = mape(y_true, y_pred)
    assert result > 1.0


def test_mape_known_values():
    # y_true=0.5, y_pred=0.6 → |0.5-0.6|/0.5 * 100 = 20.0%
    y_true = np.array([0.5])
    y_pred = np.array([0.6])
    assert abs(mape(y_true, y_pred) - 20.0) < 1e-9


def test_mape_returns_float():
    assert isinstance(mape(np.array([0.1]), np.array([0.2])), float)


def test_mape_clip_prevents_explosion_near_zero():
    # Without clipping, y_true=0.0001 would give MAPE = 100 000%.
    # clip_min=0.01 caps the denominator so the result stays finite and reasonable.
    y_true = np.array([0.0001])
    y_pred = np.array([0.0101])
    result = mape(y_true, y_pred)
    assert result < 10_000.0


def test_mape_negative_y_true_treated_as_positive():
    # log_return_7d can be negative (price drop). A -0.2 return has the same
    # scale as a +0.2 return — abs() normalises before dividing.
    y_true_pos = np.array([0.2])
    y_true_neg = np.array([-0.2])
    y_pred = np.array([0.0])
    assert mape(y_true_pos, y_pred) == mape(y_true_neg, y_pred)


def test_mape_custom_clip_min():
    # With clip_min=0.01: |0.005-0.015| / 0.01 * 100 = 100.0%
    y_true = np.array([0.005])
    y_pred = np.array([0.015])
    assert abs(mape(y_true, y_pred, clip_min=0.01) - 100.0) < 1e-9


# ---------------------------------------------------------------------------
# evaluate_per_tier() — new API: (y_true, predictions, tiers)
# ---------------------------------------------------------------------------


@pytest.fixture
def tier_inputs():
    """y_true, predictions dict, and tiers Series — one card per tier."""
    y_true = pd.Series([0.05, 0.10, 0.20])
    predictions = {
        "naive": pd.Series([0.0, 0.0, 0.0]),
        "lightgbm": pd.Series([0.06, 0.08, 0.15]),
    }
    tiers = pd.Series([1, 2, 3])
    return y_true, predictions, tiers


def test_evaluate_per_tier_returns_dataframe(tier_inputs):
    y_true, predictions, tiers = tier_inputs
    result = evaluate_per_tier(y_true, predictions, tiers)
    assert isinstance(result, pd.DataFrame)


def test_evaluate_per_tier_columns(tier_inputs):
    y_true, predictions, tiers = tier_inputs
    result = evaluate_per_tier(y_true, predictions, tiers)
    assert list(result.columns) == ["model", "tier", "n_cards", "mae", "mape"]


def test_evaluate_per_tier_row_count(tier_inputs):
    # 2 models × 3 tiers = 6 rows (each tier has exactly 1 card here)
    y_true, predictions, tiers = tier_inputs
    result = evaluate_per_tier(y_true, predictions, tiers)
    assert len(result) == 6


def test_evaluate_per_tier_model_names_present(tier_inputs):
    y_true, predictions, tiers = tier_inputs
    result = evaluate_per_tier(y_true, predictions, tiers)
    assert set(result["model"]) == {"naive", "lightgbm"}


def test_evaluate_per_tier_omits_empty_tiers():
    # Only Tier 1 cards — tiers 2 and 3 must be absent from the result.
    y_true = pd.Series([0.1, 0.2, 0.3])
    predictions = {"naive": pd.Series([0.0, 0.0, 0.0])}
    tiers = pd.Series([1, 1, 1])
    result = evaluate_per_tier(y_true, predictions, tiers)
    assert set(result["tier"]) == {1}


def test_evaluate_per_tier_mae_value_correct():
    # Single card in Tier 1: y_true=0.2, y_pred=0.1 → MAE = 0.1
    y_true = pd.Series([0.2])
    predictions = {"model_a": pd.Series([0.1])}
    tiers = pd.Series([1])
    result = evaluate_per_tier(y_true, predictions, tiers)
    assert abs(result.iloc[0]["mae"] - 0.1) < 1e-9


def test_evaluate_per_tier_sorted_by_tier_then_mape():
    # Two models in two tiers — result must be ordered tier 1 first, then tier 2,
    # and within each tier the lower mape model must appear first.
    y_true = pd.Series([0.2, 0.2, 0.5, 0.5])
    predictions = {
        "good": pd.Series([0.2, 0.2, 0.5, 0.5]),  # perfect → mape 0
        "bad": pd.Series([0.0, 0.0, 0.0, 0.0]),  # large errors → higher mape
    }
    tiers = pd.Series([1, 1, 2, 2])
    result = evaluate_per_tier(y_true, predictions, tiers)
    tier1 = result[result["tier"] == 1]["mape"].tolist()
    tier2 = result[result["tier"] == 2]["mape"].tolist()
    assert tier1 == sorted(tier1)
    assert tier2 == sorted(tier2)
    assert list(result["tier"]) == sorted(result["tier"].tolist())


def test_evaluate_per_tier_n_cards_counts_correctly():
    y_true = pd.Series([0.1, 0.2, 0.3, 0.4])
    predictions = {"m": pd.Series([0.0, 0.0, 0.0, 0.0])}
    tiers = pd.Series([1, 1, 1, 2])  # 3 in tier 1, 1 in tier 2
    result = evaluate_per_tier(y_true, predictions, tiers)
    assert result[result["tier"] == 1]["n_cards"].iloc[0] == 3
    assert result[result["tier"] == 2]["n_cards"].iloc[0] == 1
