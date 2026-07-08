"""Shared pytest fixtures/constants for tests/ml/."""

from src.ml.models.lightgbm_model import LightGBMParams

# Fast params keep each LightGBM fit under ~100 ms. Default params
# (n_estimators=1000, min_child_samples=50) would make the ML test suite
# take minutes for no coverage gain. Shared by test_lightgbm_model.py,
# test_tiered.py, and test_shap_analysis.py, which previously each defined
# this identically.
FAST_LIGHTGBM_PARAMS = LightGBMParams(
    n_estimators=10,
    num_leaves=4,
    min_child_samples=5,
    learning_rate=0.3,
    subsample=1.0,
    colsample_bytree=1.0,
    random_state=0,
)
