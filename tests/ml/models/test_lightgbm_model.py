from unittest.mock import MagicMock, patch

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from src.ml.models.lightgbm_model import LightGBMParams, LightGBMPriceModel
from tests.ml.conftest import FAST_LIGHTGBM_PARAMS as FAST_PARAMS


@pytest.fixture
def training_data():
    """60-row DataFrame with three representative features and a log_return_7d target.

    log_return_7d: log(price_in_7_days / price_today) — the model target.
    Split 30/30 so both train and val sets have enough rows for min_child_samples=5.
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
    """LightGBMPriceModel already trained; returns (model, full X) for prediction tests."""
    X, y = training_data
    mid = len(X) // 2
    model = LightGBMPriceModel(params=FAST_PARAMS)
    model.fit(X.iloc[:mid], y.iloc[:mid], X.iloc[mid:], y.iloc[mid:])
    return model, X


# ---------------------------------------------------------------------------
# LightGBMParams
# ---------------------------------------------------------------------------


def test_default_params_objective_is_mae():
    # MAE is required because MTG prices follow Pareto(α=1.303) with infinite
    # variance — MSE would let a single €2,000 card dominate the gradient.
    assert LightGBMParams().objective == "mae"


def test_custom_params_are_stored():
    params = LightGBMParams(num_leaves=31, learning_rate=0.1)
    model = LightGBMPriceModel(params=params)
    assert model.params.num_leaves == 31
    assert model.params.learning_rate == 0.1


def test_default_params_used_when_none_passed():
    model = LightGBMPriceModel(params=None)
    assert isinstance(model.params, LightGBMParams)


# ---------------------------------------------------------------------------
# fit()
# ---------------------------------------------------------------------------


def test_model_is_none_before_fit():
    model = LightGBMPriceModel(params=FAST_PARAMS)
    assert model.model is None


def test_fit_returns_self(training_data):
    # Enables chaining: model.fit(...).predict(X_test)
    X, y = training_data
    mid = len(X) // 2
    model = LightGBMPriceModel(params=FAST_PARAMS)
    result = model.fit(X.iloc[:mid], y.iloc[:mid], X.iloc[mid:], y.iloc[mid:])
    assert result is model


def test_model_is_booster_after_fit(fitted_model):
    model, _ = fitted_model
    assert isinstance(model.model, lgb.Booster)


def test_best_iteration_is_set_after_fit(fitted_model):
    # best_iteration marks the round with lowest validation loss.
    # predict() uses it to avoid including trees built after overfitting starts.
    model, _ = fitted_model
    assert model.model.best_iteration > 0


def test_n_estimators_not_passed_to_lgb_train(training_data):
    # n_estimators is sklearn naming; the native lgb.train() API uses the
    # separate num_boost_round argument instead. Passing n_estimators as a
    # model param would trigger an "Unknown parameter" warning from LightGBM.
    # random_state must be renamed to seed for the same reason.
    X, y = training_data
    mid = len(X) // 2
    model = LightGBMPriceModel(params=FAST_PARAMS)

    with patch("src.ml.models.lightgbm_model.lgb.train") as mock_train:
        mock_booster = MagicMock()
        mock_booster.best_iteration = 5
        mock_train.return_value = mock_booster
        model.fit(X.iloc[:mid], y.iloc[:mid], X.iloc[mid:], y.iloc[mid:])

    passed_params = mock_train.call_args[0][0]
    assert "n_estimators" not in passed_params
    assert "random_state" not in passed_params
    assert "seed" in passed_params


# ---------------------------------------------------------------------------
# predict()
# ---------------------------------------------------------------------------


def test_predict_returns_ndarray(fitted_model):
    model, X = fitted_model
    assert isinstance(model.predict(X), np.ndarray)


def test_predict_length_matches_input(fitted_model):
    model, X = fitted_model
    assert len(model.predict(X)) == len(X)


def test_predict_values_are_finite(fitted_model):
    # Guards against NaN/Inf propagation from unexpected feature values.
    model, X = fitted_model
    assert np.all(np.isfinite(model.predict(X)))


# ---------------------------------------------------------------------------
# feature_importance()
# ---------------------------------------------------------------------------


def test_feature_importance_returns_series(fitted_model):
    model, X = fitted_model
    result = model.feature_importance(list(X.columns))
    assert isinstance(result, pd.Series)


def test_feature_importance_index_matches_names(fitted_model):
    model, X = fitted_model
    result = model.feature_importance(list(X.columns))
    assert set(result.index) == set(X.columns)


def test_feature_importance_length_matches_features(fitted_model):
    model, X = fitted_model
    result = model.feature_importance(list(X.columns))
    assert len(result) == X.shape[1]


def test_feature_importance_sorted_descending(fitted_model):
    # shap_analysis.py and MLflow consumers expect the Series pre-sorted
    # so the most informative feature appears first.
    model, X = fitted_model
    result = model.feature_importance(list(X.columns))
    assert result.is_monotonic_decreasing
