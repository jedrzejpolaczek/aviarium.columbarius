"""
Primary price prediction model — LightGBM with MAE/Huber loss.

log_return_7d: the target variable across the whole ML pipeline,
defined as log(price_in_7_days / price_today). Zero means no price change;
positive values indicate a price increase, negative a decrease.

WHY MAE INSTEAD OF MSE:
MTG card prices follow a Pareto distribution with α = 1.303 < 2
(confirmed in statistical_properties/01). Because 1 < α < 2 the distribution
has finite mean but infinite theoretical variance. MSE squares errors — a single
€2,000 outlier would dominate the entire gradient signal. MAE penalises errors
linearly, making it robust to the heavy tail. Set objective='mae' or 'huber'.

HOW EARLY STOPPING WORKS:
LightGBM evaluates the validation loss every N rounds. If it does not improve
for 50 consecutive rounds, training stops. This prevents overfitting without
manually guessing the number of trees.

HYPERPARAMETERS:
Default values are a starting point. Optuna in evaluation/shap_analysis.py
will optimise them later (T8).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import lightgbm as lgb


@dataclass
class LightGBMParams:
    """
    Model hyperparameters. Optuna modifies these values during tuning (T8).

    Stored as a dataclass so vars(params) converts directly to the dict that
    lgb.train() expects and mlflow.log_params() accepts without extra work.
    """

    objective: str = "mae"  # MAE because of infinite variance (Pareto α=1.303)
    num_leaves: int = 63  # Tree complexity (Optuna range: 32–256)
    learning_rate: float = 0.05  # Gradient step size (Optuna range: 0.01–0.3)
    n_estimators: int = 1000  # Max trees; early stopping will halt sooner
    min_child_samples: int = 50  # Min samples per leaf (Optuna range: 20–200)
    subsample: float = 0.8  # Fraction of rows sampled per tree (Optuna: 0.6–1.0)
    colsample_bytree: float = 0.8  # Fraction of features sampled per tree
    random_state: int = 42


class LightGBMPriceModel:
    """
    LightGBM wrapper with a fit/predict interface consistent with baseline models.

    Trains a gradient-boosted tree ensemble on log_return_7d using the native
    lgb.train() API with early stopping on a held-out validation set.

    model: lgb.Booster after fit(), None before.
    """

    def __init__(self, params: LightGBMParams | None = None) -> None:
        self.params = params or LightGBMParams()
        self.model: lgb.Booster | None = None  # lgb.Booster after fit()

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> "LightGBMPriceModel":
        """Train the model with early stopping on the validation set.

        If X_val / y_val are omitted, the last 20% of X_train is used as the
        validation set automatically (temporal order is preserved).

        Args:
            X_train: Feature matrix for training.
            y_train: True log_return_7d values for training.
            X_val:   Feature matrix for validation (early stopping monitor).
                     Optional — if None, an internal 80/20 split is used.
            y_val:   True log_return_7d values for validation. Optional.

        Returns:
            self — allows chaining: model.fit(...).predict(X_test).
        """
        if X_val is None or y_val is None:
            split = int(len(X_train) * 0.8)
            X_val = X_train.iloc[split:]
            y_val = y_train.iloc[split:]
            X_train = X_train.iloc[:split]
            y_train = y_train.iloc[:split]
        params_dict = vars(self.params).copy()
        params_dict.pop("n_estimators")  # passed separately as num_boost_round
        params_dict["seed"] = params_dict.pop("random_state")  # native API uses 'seed'

        train_set = lgb.Dataset(X_train, y_train)
        val_set = lgb.Dataset(X_val, y_val, reference=train_set)

        self.model = lgb.train(
            params_dict,
            train_set,
            num_boost_round=self.params.n_estimators,
            valid_sets=[train_set, val_set],
            valid_names=["train", "val"],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=100),
            ],
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict log_return_7d using trees up to the best validation round.

        Args:
            X: Feature matrix. Must contain the same columns used in fit().

        Returns:
            np.ndarray of predicted log_return_7d values, shape (len(X),).
        """
        if self.model is None:
            raise RuntimeError("Call fit() before predict().")
        return np.asarray(
            self.model.predict(X, num_iteration=self.model.best_iteration)
        )

    def feature_importance(self, feature_names: list[str]) -> pd.Series:
        """Return feature importances by total gain, sorted descending.

        gain: total reduction in loss contributed by each feature across all trees.
        Used by shap_analysis.py and logged to MLflow.

        Args:
            feature_names: Column names matching the order used in fit().

        Returns:
            pd.Series indexed by feature name, sorted from most to least important.
        """
        if self.model is None:
            raise RuntimeError("Call fit() before feature_importance().")
        importances = self.model.feature_importance(importance_type="gain")
        return pd.Series(importances, index=feature_names).sort_values(ascending=False)
