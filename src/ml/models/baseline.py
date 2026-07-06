"""
Baseline models — the simplest possible predictors for log_return_7d.

log_return_7d: the target variable across the whole ML pipeline,
defined as log(price_in_7_days / price_today). Zero means no price change;
positive values indicate a price increase, negative a decrease.

Each class exposes fit(X, y) / predict(X) matching the sklearn interface so
they can be evaluated with the same functions in metrics.py as LightGBM.
fit() always accepts (X, y) even when the model ignores X entirely.

BASELINES AND WHAT THEY TELL YOU:
  NaiveForecast   — always predicts 0 (no price change).
                    If LightGBM loses to this, something is wrong with the model or data.
  MeanForecast    — always predicts the training mean of log_return_7d.
                    If LightGBM loses to this, it learns no signal at all.
  MovingAverageForecast — predicts mean reversion toward rolling_mean_7d.
                    Beats Naive → market reverts to mean → rolling features are useful.
                    Loses to Naive → prices have momentum → last price is the best estimate.
  AR1Forecast     — linear regression on lag_1d_return (yesterday's price change).
                    Beats Naive → lag-1 autocorrelation carries signal for a 7-day horizon.
                    Losing to Naive → autocorrelation does not transfer to 7-day predictions.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


class NaiveForecast:
    """
    Baseline: predicts that the card price will not change over the next 7 days,
    i.e. log_return_7d = 0 for every card.

    log_return_7d: the target variable — log(price_in_7_days / price_today).
    A value of 0 means no price change. Positive = price increase, negative = decrease.

    If LightGBM does not beat this baseline, the model learns nothing useful.
    """

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "NaiveForecast":
        """No training needed. Returns self for API consistency.

        Args:
            X: Feature matrix (not used).
            y: True log_return_7d values from the training set (not used).

        Returns:
            self
        """
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict zero log-return for every card.

        Args:
            X: Feature matrix. Only its length is used.

        Returns:
            np.ndarray of zeros, shape (len(X),).
        """
        return np.zeros(len(X))


class MeanForecast:
    """
    Baseline: always predicts the mean log_return_7d observed in training data.

    log_return_7d: log(price_in_7_days / price_today) — the target variable.
    mean_: the arithmetic mean of log_return_7d across all training cards,
           stored after fit() and reused as a constant prediction.

    Sanity check — if LightGBM does not beat this, the model is broken.
    Equivalent to DummyRegressor(strategy='mean') from sklearn.
    """

    def __init__(self) -> None:
        self.mean_: float | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MeanForecast":
        """Compute and store the mean log-return from training labels.

        Args:
            X: Feature matrix (not used).
            y: True log_return_7d values from the training set.

        Returns:
            self
        """
        self.mean_ = y.mean()
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return the training mean as prediction for every card.

        Args:
            X: Feature matrix. Only its length is used.

        Returns:
            np.ndarray filled with self.mean_, shape (len(X),).
        """
        return np.full(len(X), self.mean_)


class MovingAverageForecast:
    """
    Baseline: predicts log_return_7d as the gap between the rolling average
    and today's log price — assumes prices revert to their recent mean.

    window: lookback window in days. Determines which rolling_mean_{window}d column
            is read from X. lag.py must have computed that column beforehand.
    log_eur: log1p(current EUR price) — today's price in the same scale.

    The prediction rolling_mean_{window}d - log_eur says: "the price will move back
    toward its recent average." If this beats NaiveForecast, the market exhibits mean
    reversion and rolling features are useful inputs for LightGBM.
    """

    def __init__(self, window: int = 7) -> None:
        self.window = window

    def fit(
        self, X: pd.DataFrame, y: pd.Series | None = None
    ) -> "MovingAverageForecast":
        """No training needed. Returns self for API consistency.

        Args:
            X: Feature matrix (not used).
            y: True log_return_7d values (not used).

        Returns:
            self
        """
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict log-return as deviation of rolling mean from current log price.

        Args:
            X: Feature matrix. Must contain columns:
               - rolling_mean_{window}d: log1p mean EUR price over the past window days.
               - log_eur: log1p current EUR price.

        Returns:
            np.ndarray of shape (len(X),) with values rolling_mean_{window}d - log_eur.
        """
        col = f"rolling_mean_{self.window}d"
        return (X[col] - X["log_eur"]).to_numpy()


class AR1Forecast:
    """
    Baseline: first-order autoregression (AR(1)) — predicts log_return_7d from
    the previous day's price change (lag_1d_return). AR(1) means the model uses
    exactly one lagged value of the series as its sole predictor.

    lag_1d_return: (price_today - price_yesterday) / price_yesterday — the 1-day
                momentum feature from lag.py. Captures whether the card gained
                or lost value yesterday.

    Rationale: the Ljung-Box test (statistical_properties/02 notebook) checks
    whether lag-1 autocorrelation is statistically significant, i.e. whether
    yesterday's price move is non-randomly correlated with future moves. If
    significant, yesterday's move should predict the next 7-day move and AR1
    will beat NaiveForecast. If not — the autocorrelation does not carry signal
    at a 7-day horizon and AR1 will lose to Naive.
    """

    def __init__(self) -> None:
        self.model: LinearRegression | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "AR1Forecast":
        """Fit a LinearRegression on the lag_1d_return feature.

        Args:
            X: Feature matrix. Must contain column lag_1d_return (NaNs filled with 0).
            y: True log_return_7d values from the training set.

        Returns:
            self
        """
        self.model = LinearRegression()
        self.model.fit(X[["lag_1d_return"]].fillna(0), y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict log_return_7d using the fitted AR(1) linear model.

        Args:
            X: Feature matrix. Must contain column lag_1d_return (NaNs filled with 0).

        Returns:
            np.ndarray of predicted log_return_7d values, shape (len(X),).
        """
        if self.model is None:
            raise RuntimeError("Call fit() before predict().")
        return np.asarray(self.model.predict(X[["lag_1d_return"]].fillna(0)))
