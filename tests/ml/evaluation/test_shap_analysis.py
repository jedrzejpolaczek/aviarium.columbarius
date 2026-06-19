import matplotlib
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.ml.evaluation.shap_analysis import (
    compute_shap_values,
    plot_summary,
    plot_waterfall,
    run_optuna_tuning,
)
from src.ml.models.lightgbm_model import LightGBMParams, LightGBMPriceModel

# Use non-interactive backend so plt calls work in CI without a display.
matplotlib.use("Agg")

# Fast params keep each test under ~100 ms (same rationale as test_lightgbm_model.py).
FAST_PARAMS = LightGBMParams(
    n_estimators=10,
    num_leaves=4,
    min_child_samples=5,
    learning_rate=0.3,
    subsample=1.0,
    colsample_bytree=1.0,
    random_state=0,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def training_data():
    """60-row DataFrame with three features and a log_return_7d target.

    Split 30/30 so both train and val sets satisfy min_child_samples=5.
    """
    rng = np.random.default_rng(42)
    n = 60
    X = pd.DataFrame(
        {
            "log_eur": rng.normal(0, 1, n),
            "rarity_ord": rng.integers(0, 4, n).astype(float),
            "edhrec_rank": rng.uniform(0, 30_000, n),
        }
    )
    y = pd.Series(rng.normal(0, 0.05, n), name="log_return_7d")
    return X, y


@pytest.fixture
def fitted_model(training_data):
    """LightGBMPriceModel trained on the first 30 rows, validated on the last 30."""
    X, y = training_data
    mid = len(X) // 2
    model = LightGBMPriceModel(params=FAST_PARAMS)
    model.fit(X.iloc[:mid], y.iloc[:mid], X.iloc[mid:], y.iloc[mid:])
    return model


@pytest.fixture
def shap_result(fitted_model, training_data):
    """(shap_values, explainer) tuple reused across plot tests."""
    X, _ = training_data
    return compute_shap_values(fitted_model, X)


@pytest.fixture(autouse=True)
def close_figures():
    """Close all matplotlib figures after each test to prevent resource leaks."""
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# compute_shap_values()
# ---------------------------------------------------------------------------


def test_compute_shap_values_returns_tuple(fitted_model, training_data):
    X, _ = training_data
    result = compute_shap_values(fitted_model, X)
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_compute_shap_values_first_element_is_explanation(fitted_model, training_data):
    import shap

    X, _ = training_data
    shap_values, _ = compute_shap_values(fitted_model, X)
    assert isinstance(shap_values, shap.Explanation)


def test_compute_shap_values_second_element_is_explainer(fitted_model, training_data):
    import shap

    X, _ = training_data
    _, explainer = compute_shap_values(fitted_model, X)
    assert isinstance(explainer, shap.TreeExplainer)


def test_compute_shap_values_shape_matches_features(fitted_model, training_data):
    # shap_values.values must be (n_samples, n_features).
    X, _ = training_data
    shap_values, _ = compute_shap_values(fitted_model, X)
    assert shap_values.values.shape[1] == X.shape[1]


def test_compute_shap_values_samples_large_X(fitted_model, training_data):
    # For X with >2000 rows the function must cap the sample at 2000.
    X, _ = training_data
    large_X = pd.concat([X] * 40, ignore_index=True)  # 2400 rows
    assert len(large_X) > 2_000
    shap_values, _ = compute_shap_values(fitted_model, large_X)
    assert shap_values.values.shape[0] == 2_000


def test_compute_shap_values_small_X_uses_all_rows(fitted_model, training_data):
    X, _ = training_data
    small_X = X.iloc[:10]
    shap_values, _ = compute_shap_values(fitted_model, small_X)
    assert shap_values.values.shape[0] == 10


def test_compute_shap_values_are_finite(fitted_model, training_data):
    X, _ = training_data
    shap_values, _ = compute_shap_values(fitted_model, X)
    assert np.all(np.isfinite(shap_values.values))


# ---------------------------------------------------------------------------
# plot_summary()  — takes (shap_values, X: DataFrame)
# ---------------------------------------------------------------------------


def test_plot_summary_returns_figure(shap_result, training_data):
    shap_values, _ = shap_result
    X, _ = training_data
    fig = plot_summary(shap_values, X)
    assert isinstance(fig, matplotlib.figure.Figure)


def test_plot_summary_default_max_display_does_not_raise(shap_result, training_data):
    shap_values, _ = shap_result
    X, _ = training_data
    plot_summary(shap_values, X)


def test_plot_summary_custom_max_display(shap_result, training_data):
    shap_values, _ = shap_result
    X, _ = training_data
    fig = plot_summary(shap_values, X, max_display=2)
    assert isinstance(fig, matplotlib.figure.Figure)


# ---------------------------------------------------------------------------
# plot_waterfall()  — takes (shap_values, X: DataFrame, card_index)
# ---------------------------------------------------------------------------


def test_plot_waterfall_returns_figure(shap_result, training_data):
    shap_values, _ = shap_result
    X, _ = training_data
    fig = plot_waterfall(shap_values, X, card_index=0)
    assert isinstance(fig, matplotlib.figure.Figure)


def test_plot_waterfall_with_card_name_sets_title(shap_result, training_data):
    shap_values, _ = shap_result
    X, _ = training_data
    fig = plot_waterfall(shap_values, X, card_index=0, card_name="Black Lotus")
    # The figure's axes should have a title containing the card name.
    titles = [ax.get_title() for ax in fig.axes]
    assert any("Black Lotus" in t for t in titles)


def test_plot_waterfall_without_card_name_does_not_raise(shap_result, training_data):
    shap_values, _ = shap_result
    X, _ = training_data
    plot_waterfall(shap_values, X, card_index=0, card_name="")


def test_plot_waterfall_different_card_index(shap_result, training_data):
    shap_values, _ = shap_result
    X, _ = training_data
    fig = plot_waterfall(shap_values, X, card_index=5)
    assert isinstance(fig, matplotlib.figure.Figure)


def test_plot_waterfall_label_index(shap_result, training_data):
    # card_index can be a DataFrame index label, not just a positional integer.
    shap_values, _ = shap_result
    X, _ = training_data
    label = X.index[3]
    fig = plot_waterfall(shap_values, X, card_index=label)
    assert isinstance(fig, matplotlib.figure.Figure)


# ---------------------------------------------------------------------------
# run_optuna_tuning()  — returns (best_params, study) tuple
# ---------------------------------------------------------------------------


def test_run_optuna_tuning_returns_tuple(training_data):
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    X, y = training_data
    result = run_optuna_tuning(X, y, n_trials=1)
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_run_optuna_tuning_first_element_is_dict(training_data):
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    X, y = training_data
    best_params, _ = run_optuna_tuning(X, y, n_trials=1)
    assert isinstance(best_params, dict)


def test_run_optuna_tuning_second_element_is_study(training_data):
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    X, y = training_data
    _, study = run_optuna_tuning(X, y, n_trials=1)
    assert isinstance(study, optuna.Study)


def test_run_optuna_tuning_returns_expected_keys(training_data):
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    X, y = training_data
    best_params, _ = run_optuna_tuning(X, y, n_trials=1)
    expected_keys = {"num_leaves", "learning_rate", "min_child_samples", "subsample"}
    assert expected_keys == set(best_params.keys())


def test_run_optuna_tuning_learning_rate_in_valid_range(training_data):
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    X, y = training_data
    best_params, _ = run_optuna_tuning(X, y, n_trials=1)
    assert 0.01 <= best_params["learning_rate"] <= 0.3


def test_run_optuna_tuning_study_has_completed_trials(training_data):
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    X, y = training_data
    _, study = run_optuna_tuning(X, y, n_trials=2)
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    assert len(completed) >= 1
