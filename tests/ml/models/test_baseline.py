import numpy as np
import pandas as pd
import pytest
from src.ml.models.baseline import (
    AR1Forecast,
    MeanForecast,
    MovingAverageForecast,
    NaiveForecast,
)


@pytest.fixture
def sample_data():
    X = pd.DataFrame({"log_eur": [1.0, 2.0, 3.0]})
    y = pd.Series([0.05, -0.02, 0.10])
    return X, y


@pytest.fixture
def ma_data():
    X = pd.DataFrame(
        {
            "log_eur": [1.0, 2.0, 3.0],
            "rolling_mean_7d": [1.1, 1.8, 3.2],
        }
    )
    y = pd.Series([0.05, -0.02, 0.10])
    return X, y


def test_naive_predict_returns_zeros(sample_data):
    X, y = sample_data
    model = NaiveForecast().fit(X, y)
    np.testing.assert_array_equal(model.predict(X), np.zeros(len(X)))


def test_naive_predict_length_matches_input(sample_data):
    X, y = sample_data
    model = NaiveForecast().fit(X, y)
    assert len(model.predict(X)) == len(X)


def test_naive_fit_returns_self(sample_data):
    X, y = sample_data
    model = NaiveForecast()
    assert model.fit(X, y) is model


def test_mean_forecast_predict_returns_mean(sample_data):
    X, y = sample_data
    model = MeanForecast().fit(X, y)
    np.testing.assert_array_almost_equal(model.predict(X), np.full(len(X), y.mean()))


def test_mean_forecast_predict_length_matches_input(sample_data):
    X, y = sample_data
    model = MeanForecast().fit(X, y)
    assert len(model.predict(X)) == len(X)


def test_mean_forecast_fit_returns_self(sample_data):
    X, y = sample_data
    model = MeanForecast()
    assert model.fit(X, y) is model


def test_ma_forecast_predict_values(ma_data):
    X, y = ma_data
    model = MovingAverageForecast().fit(X, y)
    expected = (X["rolling_mean_7d"] - X["log_eur"]).values
    np.testing.assert_array_almost_equal(model.predict(X), expected)


def test_ma_forecast_predict_length_matches_input(ma_data):
    X, y = ma_data
    model = MovingAverageForecast().fit(X, y)
    assert len(model.predict(X)) == len(X)


def test_ma_forecast_fit_returns_self(ma_data):
    X, y = ma_data
    model = MovingAverageForecast()
    assert model.fit(X, y) is model


def test_ma_forecast_predicts_positive_when_price_below_mean():
    # price below rolling mean → model predicts price will rise
    X = pd.DataFrame({"log_eur": [2.0], "rolling_mean_7d": [2.5]})
    y = pd.Series([0.0])
    pred = MovingAverageForecast().fit(X, y).predict(X)
    assert pred[0] > 0


def test_ma_forecast_predicts_negative_when_price_above_mean():
    # price above rolling mean → model predicts price will fall
    X = pd.DataFrame({"log_eur": [3.0], "rolling_mean_7d": [2.5]})
    y = pd.Series([0.0])
    pred = MovingAverageForecast().fit(X, y).predict(X)
    assert pred[0] < 0


def test_ma_forecast_predicts_zero_when_price_equals_mean():
    # price exactly at rolling mean → no predicted movement
    X = pd.DataFrame({"log_eur": [2.0], "rolling_mean_7d": [2.0]})
    y = pd.Series([0.0])
    pred = MovingAverageForecast().fit(X, y).predict(X)
    assert pred[0] == 0.0


def test_ma_forecast_returns_ndarray(ma_data):
    X, y = ma_data
    pred = MovingAverageForecast().fit(X, y).predict(X)
    assert isinstance(pred, np.ndarray)


@pytest.fixture
def ar1_data():
    X = pd.DataFrame({"lag_1d_return": [0.05, -0.02, 0.10, 0.0, -0.05]})
    y = pd.Series([0.10, -0.04, 0.20, 0.0, -0.10])
    return X, y


def test_ar1_fit_returns_self(ar1_data):
    X, y = ar1_data
    model = AR1Forecast()
    assert model.fit(X, y) is model


def test_ar1_predict_length_matches_input(ar1_data):
    X, y = ar1_data
    model = AR1Forecast().fit(X, y)
    assert len(model.predict(X)) == len(X)


def test_ar1_predict_returns_ndarray(ar1_data):
    X, y = ar1_data
    pred = AR1Forecast().fit(X, y).predict(X)
    assert isinstance(pred, np.ndarray)


def test_ar1_predict_handles_nan():
    # NaNs in lag_1d_return should be treated as 0 (no momentum)
    X_train = pd.DataFrame({"lag_1d_return": [0.05, -0.02, 0.10]})
    y_train = pd.Series([0.10, -0.04, 0.20])
    X_test = pd.DataFrame({"lag_1d_return": [float("nan")]})
    pred = AR1Forecast().fit(X_train, y_train).predict(X_test)
    assert len(pred) == 1
    assert np.isfinite(pred[0])


def test_ar1_learns_positive_relationship():
    # perfect linear relationship: return_7d = 2 * lag_1d_return
    X = pd.DataFrame({"lag_1d_return": [0.1, 0.2, 0.3, 0.4]})
    y = pd.Series([0.2, 0.4, 0.6, 0.8])
    model = AR1Forecast().fit(X, y)
    pred = model.predict(pd.DataFrame({"lag_1d_return": [0.5]}))
    assert abs(pred[0] - 1.0) < 1e-6
