import matplotlib
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.ml.evaluation.error_analysis import (
    error_by_group,
    residual_plot,
    worst_predictions,
)

matplotlib.use("Agg")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def prediction_df():
    """Small DataFrame with three cards spanning different error magnitudes.

    Card 'A': perfect prediction (pct_error = 0)
    Card 'B': 100% error on a near-zero return (clipped to clip_min=0.01)
    Card 'C': 50% error on a normal return
    """
    return pd.DataFrame(
        {
            "name": ["A", "B", "C"],
            "eur": [10.0, 5.0, 200.0],
            "tier": [1, 1, 2],
            "log_return_7d": [0.2, 0.001, 0.4],
            "predicted": [0.2, 0.011, 0.6],
        }
    )


@pytest.fixture
def group_df():
    """DataFrame with three rarity groups of different error levels."""
    return pd.DataFrame(
        {
            "rarity": ["Common", "Common", "Rare", "Rare", "Mythic"],
            "log_return_7d": [0.1, 0.1, 0.2, 0.2, 0.3],
            "predicted": [0.15, 0.12, 0.4, 0.5, 0.3],
        }
    )


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# worst_predictions()
# ---------------------------------------------------------------------------


def test_worst_predictions_returns_dataframe(prediction_df):
    result = worst_predictions(prediction_df)
    assert isinstance(result, pd.DataFrame)


def test_worst_predictions_columns(prediction_df):
    result = worst_predictions(prediction_df)
    assert list(result.columns) == [
        "name",
        "eur",
        "tier",
        "log_return_7d",
        "predicted",
        "pct_error",
    ]


def test_worst_predictions_sorted_descending_by_pct_error(prediction_df):
    result = worst_predictions(prediction_df)
    assert list(result["pct_error"]) == sorted(result["pct_error"], reverse=True)


def test_worst_predictions_n_limits_rows(prediction_df):
    result = worst_predictions(prediction_df, n=2)
    assert len(result) == 2


def test_worst_predictions_n_larger_than_rows_returns_all(prediction_df):
    # DataFrame has 3 rows; requesting n=10 should return all 3.
    result = worst_predictions(prediction_df, n=10)
    assert len(result) == 3


def test_worst_predictions_perfect_prediction_has_zero_error(prediction_df):
    result = worst_predictions(prediction_df)
    row_a = result[result["name"] == "A"]
    assert abs(row_a["pct_error"].iloc[0]) < 1e-9


def test_worst_predictions_clip_prevents_explosion_near_zero():
    # y_true = 0.0001 (near zero) — without clip this would be MAPE ≈ 99 900%.
    # clip_min=0.01 must cap the denominator so pct_error stays finite.
    df = pd.DataFrame(
        {
            "name": ["X"],
            "eur": [1.0],
            "tier": [1],
            "log_return_7d": [0.0001],
            "predicted": [0.0101],
        }
    )
    result = worst_predictions(df, n=1)
    assert result["pct_error"].iloc[0] < 10_000.0


def test_worst_predictions_does_not_mutate_input(prediction_df):
    original_cols = list(prediction_df.columns)
    worst_predictions(prediction_df)
    assert list(prediction_df.columns) == original_cols


def test_worst_predictions_custom_column_names():
    df = pd.DataFrame(
        {
            "name": ["X"],
            "eur": [5.0],
            "tier": [1],
            "actual": [0.2],
            "pred": [0.5],
        }
    )
    result = worst_predictions(df, y_true_col="actual", y_pred_col="pred", n=1)
    assert "actual" in result.columns
    assert "pred" in result.columns
    assert "pct_error" in result.columns


# ---------------------------------------------------------------------------
# error_by_group()
# ---------------------------------------------------------------------------


def test_error_by_group_returns_dataframe(group_df):
    result = error_by_group(group_df, group_col="rarity")
    assert isinstance(result, pd.DataFrame)


def test_error_by_group_columns(group_df):
    result = error_by_group(group_df, group_col="rarity")
    assert list(result.columns) == ["rarity", "mean", "count", "std", "share_of_total"]


def test_error_by_group_row_count_equals_unique_groups(group_df):
    result = error_by_group(group_df, group_col="rarity")
    assert len(result) == group_df["rarity"].nunique()


def test_error_by_group_sorted_descending_by_mean(group_df):
    result = error_by_group(group_df, group_col="rarity")
    assert list(result["mean"]) == sorted(result["mean"], reverse=True)


def test_error_by_group_share_of_total_sums_to_one(group_df):
    result = error_by_group(group_df, group_col="rarity")
    assert abs(result["share_of_total"].sum() - 1.0) < 1e-9


def test_error_by_group_count_values_correct(group_df):
    result = error_by_group(group_df, group_col="rarity")
    common_row = result[result["rarity"] == "Common"]
    assert common_row["count"].iloc[0] == 2


def test_error_by_group_mean_values_correct():
    # Two cards with abs_error 0.1 each → mean = 0.1.
    df = pd.DataFrame(
        {
            "rarity": ["Rare", "Rare"],
            "log_return_7d": [0.2, 0.3],
            "predicted": [0.1, 0.4],
        }
    )
    result = error_by_group(df, group_col="rarity")
    assert abs(result.iloc[0]["mean"] - 0.1) < 1e-9


def test_error_by_group_does_not_mutate_input(group_df):
    original_cols = list(group_df.columns)
    error_by_group(group_df, group_col="rarity")
    assert list(group_df.columns) == original_cols


def test_error_by_group_custom_column_names():
    df = pd.DataFrame(
        {
            "set_type": ["Core", "Core", "Expansion"],
            "actual": [0.1, 0.2, 0.3],
            "pred": [0.2, 0.1, 0.6],
        }
    )
    result = error_by_group(
        df, group_col="set_type", y_true_col="actual", y_pred_col="pred"
    )
    assert "set_type" in result.columns


# ---------------------------------------------------------------------------
# residual_plot()
# ---------------------------------------------------------------------------


@pytest.fixture
def residual_inputs():
    rng = np.random.default_rng(0)
    n = 50
    y_true = pd.Series(rng.normal(0, 0.1, n))
    y_pred = rng.normal(0, 0.1, n)
    price_col = pd.Series(rng.uniform(1, 500, n))
    return y_true, y_pred, price_col


def test_residual_plot_returns_figure(residual_inputs):
    y_true, y_pred, _ = residual_inputs
    fig = residual_plot(y_true, y_pred)
    assert isinstance(fig, matplotlib.figure.Figure)


def test_residual_plot_without_price_col_does_not_raise(residual_inputs):
    y_true, y_pred, _ = residual_inputs
    residual_plot(y_true, y_pred)


def test_residual_plot_with_price_col_returns_figure(residual_inputs):
    y_true, y_pred, price_col = residual_inputs
    fig = residual_plot(y_true, y_pred, price_col=price_col)
    assert isinstance(fig, matplotlib.figure.Figure)


def test_residual_plot_left_axis_has_title(residual_inputs):
    y_true, y_pred, _ = residual_inputs
    fig = residual_plot(y_true, y_pred)
    visible_axes = [ax for ax in fig.axes if ax.get_visible()]
    titles = [ax.get_title() for ax in visible_axes]
    assert any("Residuals vs Predicted" in t for t in titles)


def test_residual_plot_right_axis_has_title_when_price_provided(residual_inputs):
    y_true, y_pred, price_col = residual_inputs
    fig = residual_plot(y_true, y_pred, price_col=price_col)
    titles = [ax.get_title() for ax in fig.axes if ax.get_visible()]
    assert any("Residuals vs Price" in t for t in titles)
