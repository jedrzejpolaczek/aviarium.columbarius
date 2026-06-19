"""
Explains model predictions using SHAP (SHapley Additive exPlanations).

WHAT SHAP DOES:
A trained model knows THAT a card will increase in price — SHAP explains WHY.
For each card it assigns a contribution score to every feature:
"this card will rise because: edhrec_saltiness is high (+0.8 log-return units),
is_reserved (+2.2), top8_appearances is growing (+0.3)."
Without SHAP you only see the prediction; with SHAP you see the reasoning.

HOW TREESHAP WORKS:
shap.TreeExplainer uses the exact TreeSHAP algorithm — it walks every decision
tree in the LightGBM ensemble and computes each feature's marginal contribution
by averaging over all possible feature orderings. This is exact (not sampled)
and runs in polynomial time for trees.

EXPECTED FEATURE IMPORTANCE ORDER (from Bayesian Analysis 02):
edhrec_saltiness (β=+0.222) > is_reserved (β=+2.0) > log_top8 (β=+0.091)

CONFOUND CHECK:
Verify that print_count has low SHAP importance when edhrec_saltiness is in the
model. Bayesian Analysis 02 showed that print_count is a confounder of saltiness
(popular cards are reprinted more AND tend to be saltier). If print_count still
dominates SHAP, saltiness may have been dropped or mis-encoded.

PLOT TYPES:
1. Summary plot  — global feature importance across all cards in the dataset.
2. Waterfall plot — per-card breakdown: base value + each feature's contribution
                    = final prediction.
"""

import matplotlib.figure
import matplotlib.pyplot as plt
import optuna
import pandas as pd
import shap
from typing import Any

from src.ml.evaluation.metrics import mae
from src.ml.models.lightgbm_model import LightGBMParams, LightGBMPriceModel


def compute_shap_values(
    model: "LightGBMPriceModel", X: pd.DataFrame
) -> tuple[Any, Any]:
    """Compute SHAP values for the given feature matrix using TreeSHAP.

    For X with more than 2,000 rows a random sample of 2,000 is used to keep
    computation time reasonable; the sample is reproducible via random_state=42.

    Args:
        model: Fitted LightGBMPriceModel. model.model must be a lgb.Booster.
        X:     Feature DataFrame — the same columns used during training.

    Returns:
        (shap_values, explainer) tuple where:
          shap_values: shap.Explanation object — use shap_values[i] for a single card.
          explainer:   shap.TreeExplainer — reuse to explain new cards without refitting.
    """
    sample = X.sample(min(2_000, len(X)), random_state=42)
    explainer = shap.TreeExplainer(model.model)
    shap_values = explainer(sample)
    return shap_values, explainer


def plot_summary(
    shap_values: Any,
    X: pd.DataFrame,
    max_display: int = 20,
) -> matplotlib.figure.Figure:
    """Draw a SHAP summary plot showing global feature importance.

    Each dot is one card. Colour encodes feature value (red = high, blue = low).
    Features are ranked by mean |SHAP value| so the most influential appear first.

    Args:
        shap_values:  shap.Explanation returned by compute_shap_values().
        X:            Feature DataFrame — column names are used as axis labels.
        max_display:  Maximum number of features shown on the y-axis.

    Returns:
        matplotlib Figure (not displayed — call fig.show() or fig.savefig() yourself).
    """
    plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        feature_names=list(X.columns),
        max_display=max_display,
        show=False,
    )
    return plt.gcf()


def plot_waterfall(
    shap_values: Any,
    X: pd.DataFrame,
    card_index: int | object,
    card_name: str = "",
) -> matplotlib.figure.Figure:
    """Draw a SHAP waterfall plot for a single card.

    Shows: base value (average model output) + each feature's push up or down
    = final prediction for this card. Useful for explaining individual pricing
    decisions — e.g. why a specific card is predicted to rise next week.

    Args:
        shap_values: shap.Explanation returned by compute_shap_values().
        X:           Feature DataFrame used to resolve label-based indices.
        card_index:  Row position (int) or DataFrame index label to explain.
        card_name:   Optional card name added to the plot title.

    Returns:
        matplotlib Figure (not displayed — call fig.show() or fig.savefig() yourself).
    """
    if isinstance(card_index, int):
        pos = card_index
    else:
        pos = list(X.index).index(card_index)

    plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(shap_values[pos], show=False)
    if card_name:
        plt.title(f"SHAP: {card_name}")
    return plt.gcf()


def run_optuna_tuning(
    X: pd.DataFrame,
    y: pd.Series,
    n_trials: int = 50,
    experiment_name: str = "lightgbm_tuning",
) -> tuple[dict[str, Any], optuna.Study]:
    """Search for the best LightGBM hyperparameters using Optuna.

    Splits X/y 80/20 internally (temporal order preserved) to create train and
    validation sets. Each trial trains a model with early stopping and reports
    MAE on the validation set. The MedianPruner stops unpromising trials early.

    Args:
        X:               Full feature matrix (train + validation combined).
        y:               Full log_return_7d target Series.
        n_trials:        Number of hyperparameter combinations to try.
        experiment_name: Label shown in the MLflow UI if MLflow is active.

    Returns:
        (best_params, study) where best_params is a dict of optimal values
        and study is the Optuna Study object (use study.best_value for best MAE).
    """
    split = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split], X.iloc[split:]
    y_train, y_val = y.iloc[:split], y.iloc[split:]

    def objective(trial: optuna.Trial) -> float:
        params = LightGBMParams(
            num_leaves=trial.suggest_int("num_leaves", 32, 256),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            min_child_samples=trial.suggest_int("min_child_samples", 20, 200),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
        )
        model = LightGBMPriceModel(params)
        model.fit(X_train, y_train, X_val, y_val)
        return mae(y_val.to_numpy(), model.predict(X_val))

    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(),
    )
    study.optimize(objective, n_trials=n_trials)
    return study.best_params, study
