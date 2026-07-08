"""
Model quality metrics, always reported per price tier.

WHY PER TIER, NOT AGGREGATE:
A model with global MAE = 0.05 can have Tier 1 MAE = 0.02 and Tier 3 MAE = 0.45.
The aggregate hides errors that carry the most financial weight.

METRICS:
- MAE  (Mean Absolute Error): mean absolute error on the log1p scale.
       Good for comparing models against each other.
- MAPE (Mean Absolute Percentage Error): scale-independent, in percent.
       Official threshold for LightGBM to beat the Naive baseline.

MAPE NEAR ZERO:
Cards priced at €0.01 cause division by a very small number.
y_true is always clipped to clip_min=0.01 before computing MAPE.
"""

import numpy as np
import pandas as pd

MAPE_CLIP_MIN = 0.01
"""Lower bound for the denominator in percent-error calculations, avoiding
division-by-zero blowup for near-zero actual prices. Shared by mape() here
and worst_predictions() in error_analysis.py — see that module's docstring
cross-reference.
"""


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error on the log1p scale.

    Args:
        y_true: Observed log_return_7d values.
        y_pred: Predicted log_return_7d values.

    Returns:
        Scalar MAE value.
    """
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, clip_min: float = MAPE_CLIP_MIN) -> float:
    """Mean Absolute Percentage Error, returned as a percentage (e.g. 12.5 = 12.5%).

    y_true is clipped to clip_min before division to avoid exploding errors
    on cards priced below €0.01.

    Args:
        y_true:   Observed values.
        y_pred:   Predicted values.
        clip_min: Minimum absolute value of y_true used as denominator.

    Returns:
        Scalar MAPE value in percent.
    """
    # log_return_7d can be negative (price drop) or near zero (no change).
    # Dividing by a raw log_return would give nonsense: a tiny return of 0.0001
    # turns a small prediction error into a 10 000% MAPE.
    # abs() removes the sign — "3% error" means the same regardless of direction.
    # clip() enforces a minimum denominator of 0.01 so near-zero returns
    # don't inflate MAPE into meaningless numbers.
    y_true_safe = np.clip(np.abs(y_true), clip_min, None)
    return float(100 * np.mean(np.abs(y_true - y_pred) / y_true_safe))


def evaluate_per_tier(
    y_true: pd.Series,
    predictions: dict[str, pd.Series],
    tiers: pd.Series,
) -> pd.DataFrame:
    """Compute MAE and MAPE per model per price tier.

    Replaces the previous single-model DataFrame-based API. Accepts multiple
    models at once so notebook 02 can compare Naive / MA7d / LightGBM in one call.

    Args:
        y_true:      True log_return_7d values.
        predictions: Dict mapping model name to its predicted log_return_7d Series.
                     All Series must share the same index as y_true.
        tiers:       Integer tier (1, 2, or 3) for each row, same index as y_true.
                     Use assign_tier() from src.ml.models.tiered to build this.

    Returns:
        DataFrame with columns: model, tier, n_cards, mae, mape,
        sorted by tier then mape ascending. Tiers with zero rows are omitted.
    """
    rows = []
    for model_name, y_pred in predictions.items():
        for tier in [1, 2, 3]:
            mask = tiers == tier
            if mask.sum() == 0:
                continue
            rows.append(
                {
                    "model": model_name,
                    "tier": tier,
                    "n_cards": int(mask.sum()),
                    "mae": mae(y_true[mask].to_numpy(), y_pred[mask].to_numpy()),
                    "mape": mape(y_true[mask].to_numpy(), y_pred[mask].to_numpy()),
                }
            )

    return pd.DataFrame(rows).sort_values(["tier", "mape"]).reset_index(drop=True)
