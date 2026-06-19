"""
Identifies which cards the model predicts worst and why.

WHY ERROR ANALYSIS BEFORE FIXING THE MODEL:
A model with global MAE = 0.05 can hide a subset of cards with MAPE = 80%.
Knowing WHERE the model fails guides the next feature engineering iteration
rather than blindly tuning hyperparameters.

PATTERNS TO LOOK FOR:
- Cards from a new set (set_age_days < 30) — model has no price history yet.
  Signal: add feature 'days_since_set_release' in the next iteration.
- Cards with missing lag features — too short a price history in the database.
- Recently banned cards — price dropped sharply; the model has no ban signal.
- Systematic under-prediction for a specific rarity — the rarity_ord encoding
  may not capture the full price premium (e.g. Mythic vs Rare is non-linear).

TARGET OUTCOME OF ERROR ANALYSIS:
End with a concrete finding, e.g.:
  "Model errors are highest for cards with set_age_days < 14 — MAPE 45%
   vs 12% for the rest. Add days_since_set_release as a feature."

RESIDUAL PATTERNS:
  Random scatter around y=0 → model has no systematic bias.
  Fan shape (errors grow with predicted value) → heteroscedasticity;
    consider log-transforming predictions or using a quantile loss.
  Curve pattern → a non-linear relationship was missed by the model.
"""

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def worst_predictions(
    df: pd.DataFrame,
    y_true_col: str = "log_return_7d",
    y_pred_col: str = "predicted",
    n: int = 20,
) -> pd.DataFrame:
    """Return the N cards with the largest percentage prediction error.

    Percentage error uses the same clip_min=0.01 as mape() in metrics.py to
    prevent near-zero true returns from producing meaningless 100 000% errors.

    Args:
        df:         DataFrame with columns 'name', 'eur', 'tier', y_true_col,
                    y_pred_col. Must have at least those columns present.
        y_true_col: Column name for the observed log_return_7d.
        y_pred_col: Column name for the model's predicted log_return_7d.
        n:          Number of worst-performing cards to return.

    Returns:
        DataFrame of at most n rows, columns:
        ['name', 'eur', 'tier', y_true_col, y_pred_col, 'pct_error'],
        sorted by pct_error descending.
    """
    df = df.copy()
    df["pct_error"] = (df[y_pred_col] - df[y_true_col]).abs() / df[
        y_true_col
    ].abs().clip(lower=0.01)
    return df.nlargest(n, "pct_error")[
        ["name", "eur", "tier", y_true_col, y_pred_col, "pct_error"]
    ]


def error_by_group(
    df: pd.DataFrame,
    group_col: str,
    y_true_col: str = "log_return_7d",
    y_pred_col: str = "predicted",
) -> pd.DataFrame:
    """Compute mean absolute error grouped by a categorical column.

    Use this to find systematic failure modes, e.g.:
      error_by_group(df, "rarity") → "Model fails most on Mythic Rare"
      error_by_group(df, "set_type") → "New sets dominate the error list"

    Args:
        df:         DataFrame containing group_col, y_true_col, y_pred_col.
        group_col:  Column to group by (e.g. 'rarity', 'set_type', 'tier').
        y_true_col: Column name for observed log_return_7d.
        y_pred_col: Column name for predicted log_return_7d.

    Returns:
        DataFrame indexed by group_col with columns:
        ['mean', 'count', 'std', 'share_of_total'],
        sorted by mean descending (worst-performing groups first).
        share_of_total = fraction of all cards belonging to this group.
    """
    df = df.copy()
    df["abs_error"] = (df[y_pred_col] - df[y_true_col]).abs()
    result = (
        df.groupby(group_col)["abs_error"]
        .agg(["mean", "count", "std"])
        .sort_values("mean", ascending=False)
        .reset_index()
    )
    result["share_of_total"] = result["count"] / result["count"].sum()
    return result


def residual_plot(
    y_true: pd.Series,
    y_pred: np.ndarray,
    price_col: pd.Series | None = None,
) -> matplotlib.figure.Figure:
    """Draw residual diagnostic plots.

    Left panel — residuals vs predicted value:
      Random scatter around y=0 is ideal. A fan or curve indicates the model
      has a systematic error that hyperparameter tuning alone cannot fix.

    Right panel (only when price_col is provided) — residuals vs current EUR price:
      Checks for heteroscedasticity: errors that grow with price signal that a
      separate model (or stronger regularisation) is needed for expensive cards.

    Args:
        y_true:    Observed log_return_7d values.
        y_pred:    Predicted log_return_7d values (numpy array, same length).
        price_col: Optional Series of current EUR prices for the right panel.

    Returns:
        matplotlib Figure — call fig.savefig() or fig.show() yourself.
    """
    residuals = y_true.values - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(y_pred, residuals, alpha=0.3, s=5)
    axes[0].axhline(0, color="red", linestyle="--")
    axes[0].set_xlabel("Predicted log_return_7d")
    axes[0].set_ylabel("Residual (true − pred)")
    axes[0].set_title("Residuals vs Predicted")

    if price_col is not None:
        axes[1].scatter(price_col.values, residuals, alpha=0.3, s=5)
        axes[1].axhline(0, color="red", linestyle="--")
        axes[1].set_xlabel("Current price EUR")
        axes[1].set_title("Residuals vs Price")
    else:
        axes[1].set_visible(False)

    plt.tight_layout()
    return fig
