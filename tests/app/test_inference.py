"""Tests for inference helpers extracted from app/routers/predict.py."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock

from app.routers.predict import inverse_log_return
from app.schemas.responses import PredictionResponse


def test_inverse_log_return_zero_return():
    eur = np.array([10.0])
    result = inverse_log_return(eur, np.array([0.0]))
    assert result[0] == pytest.approx(10.0, rel=1e-5)


def test_inverse_log_return_positive_return():
    eur = np.array([20.0])
    log_return = np.array([0.1])
    expected = float(np.expm1(np.log1p(20.0) + 0.1))
    result = inverse_log_return(eur, log_return)
    assert result[0] == pytest.approx(expected, rel=1e-5)


def test_inverse_log_return_nan_price():
    eur = np.array([np.nan, 10.0])
    log_return = np.array([0.1, 0.1])
    result = inverse_log_return(eur, log_return)
    assert np.isnan(result[0])
    assert not np.isnan(result[1])


def test_inverse_log_return_batch():
    eur = np.array([5.0, 10.0, np.nan])
    log_returns = np.array([0.0, 0.1, 0.2])
    result = inverse_log_return(eur, log_returns)
    assert result[0] == pytest.approx(5.0, rel=1e-5)
    assert not np.isnan(result[1])
    assert np.isnan(result[2])


def _make_test_state():
    X_all = pd.DataFrame(
        {
            "uuid": ["uuid-1"],
            "name": ["Lightning Bolt"],
            "eur": [5.0],
        }
    )
    X_all_t = pd.DataFrame({"feature_1": [0.5]})
    model = MagicMock()
    model.predict.return_value = np.array([0.1])
    return X_all, X_all_t, model


def test_predict_from_index_tier1():
    from app.routers.predict import _predict_from_index

    X_all, X_all_t, model = _make_test_state()
    result = _predict_from_index(0, "Lightning Bolt", X_all, X_all_t, model, "run-1")
    assert isinstance(result, PredictionResponse)
    assert result.card_name == "Lightning Bolt"
    assert result.current_price == pytest.approx(5.0)
    assert result.tier == 1
    assert result.log_return_7d == pytest.approx(0.1, rel=1e-5)
    assert result.predicted_price is not None
    assert result.model_run_id == "run-1"


def test_predict_from_index_tier3_returns_null_prediction():
    from app.routers.predict import _predict_from_index

    X_all = pd.DataFrame(
        {
            "uuid": ["uuid-1"],
            "name": ["Black Lotus"],
            "eur": [5000.0],
        }
    )
    X_all_t = pd.DataFrame({"feature_1": [0.5]})
    model = MagicMock()
    result = _predict_from_index(0, "Black Lotus", X_all, X_all_t, model, "run-1")
    assert result.tier == 3
    assert result.predicted_price is None
    assert result.log_return_7d is None
    model.predict.assert_not_called()
