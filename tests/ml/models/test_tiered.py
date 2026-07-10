import numpy as np
import pandas as pd
import pytest

from src.ml.models.tiered import TIER1_MAX_EUR, TIER2_MAX_EUR, TieredRouter, assign_tier
from tests.ml.conftest import FAST_LIGHTGBM_PARAMS as FAST_PARAMS

FEATURE_COLS = ["f1", "f2"]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tiered_data():
    """Train/val split with cards covering all three tiers.

    - 120 Tier 1 cards (eur < 100)
    - 60  Tier 2 cards (eur 100–1000) — above MIN_TIER2_ROWS=50
    - 10  Tier 3 cards (eur > 1000)

    Split 70/30 preserving temporal order.
    """
    rng = np.random.default_rng(42)
    # n_t2=80 → after 70% train split: ~56 rows in train, safely above MIN_TIER2_ROWS=50.
    n_t1, n_t2, n_t3 = 120, 80, 10
    n = n_t1 + n_t2 + n_t3

    eur = np.concatenate(
        [
            rng.uniform(1, 99, n_t1),
            rng.uniform(100, 999, n_t2),
            rng.uniform(1_001, 5_000, n_t3),
        ]
    )
    df = pd.DataFrame(
        {
            "eur": eur,
            "f1": rng.normal(0, 1, n),
            "f2": rng.normal(0, 1, n),
            "log_return_7d": rng.normal(0, 0.05, n),
        }
    )
    # Shuffle so every tier appears in both train and val after the split.
    df = df.sample(frac=1, random_state=99).reset_index(drop=True)
    split = int(n * 0.7)
    return df.iloc[:split].reset_index(drop=True), df.iloc[split:].reset_index(
        drop=True
    )


@pytest.fixture
def tier1_only_data():
    """Train/val with only Tier 1 cards — Tier 2 model must be skipped."""
    rng = np.random.default_rng(7)
    n = 60
    df = pd.DataFrame(
        {
            "eur": rng.uniform(1, 50, n),
            "f1": rng.normal(0, 1, n),
            "f2": rng.normal(0, 1, n),
            "log_return_7d": rng.normal(0, 0.05, n),
        }
    )
    split = int(n * 0.7)
    return df.iloc[:split].reset_index(drop=True), df.iloc[split:].reset_index(
        drop=True
    )


@pytest.fixture
def fitted_router(tiered_data):
    """TieredRouter fitted on the full tiered dataset."""
    train_df, val_df = tiered_data
    router = TieredRouter(params=FAST_PARAMS)
    router.fit(train_df, val_df, FEATURE_COLS)
    return router


# ---------------------------------------------------------------------------
# assign_tier()
# ---------------------------------------------------------------------------


def test_assign_tier_below_100_is_tier_1():
    assert assign_tier(0.01) == 1
    assert assign_tier(99.99) == 1


def test_assign_tier_exactly_100_is_tier_2():
    assert assign_tier(TIER1_MAX_EUR) == 2


def test_assign_tier_between_100_and_1000_is_tier_2():
    assert assign_tier(500.0) == 2
    assert assign_tier(999.99) == 2


def test_assign_tier_exactly_1000_is_tier_3():
    assert assign_tier(TIER2_MAX_EUR) == 3


def test_assign_tier_above_1000_is_tier_3():
    assert assign_tier(50_000.0) == 3


def test_assign_tier_none_defaults_to_tier_1():
    # Missing price is treated as a cheap card to avoid defaulting to Cardmarket lookup.
    assert assign_tier(None) == 1  # type: ignore[arg-type]


def test_assign_tier_nan_defaults_to_tier_1():
    assert assign_tier(float("nan")) == 1


# ---------------------------------------------------------------------------
# TieredRouter — construction
# ---------------------------------------------------------------------------


def test_init_models_are_none_before_fit():
    router = TieredRouter(params=FAST_PARAMS)
    assert router.model_tier1 is None
    assert router.model_tier2 is None


def test_init_accepts_none_params():
    # None → uses default LightGBMParams during fit(); must not raise.
    TieredRouter(params=None)


# ---------------------------------------------------------------------------
# TieredRouter.fit()
# ---------------------------------------------------------------------------


def test_fit_returns_self(tiered_data):
    train_df, val_df = tiered_data
    router = TieredRouter(params=FAST_PARAMS)
    result = router.fit(train_df, val_df, FEATURE_COLS)
    assert result is router


def test_fit_trains_tier1_model(tiered_data):
    train_df, val_df = tiered_data
    router = TieredRouter(params=FAST_PARAMS)
    router.fit(train_df, val_df, FEATURE_COLS)
    assert router.model_tier1 is not None
    assert router.model_tier1.model is not None


def test_fit_trains_tier2_model_when_enough_rows(tiered_data):
    # tiered_data fixture has 60 Tier 2 train rows → above MIN_TIER2_ROWS=50.
    train_df, val_df = tiered_data
    router = TieredRouter(params=FAST_PARAMS)
    router.fit(train_df, val_df, FEATURE_COLS)
    assert router.model_tier2 is not None
    assert router.model_tier2.model is not None


def test_fit_skips_tier2_when_insufficient_rows(tier1_only_data):
    # No Tier 2 cards → model_tier2 must remain None.
    train_df, val_df = tier1_only_data
    router = TieredRouter(params=FAST_PARAMS)
    router.fit(train_df, val_df, FEATURE_COLS)
    assert router.model_tier2 is None


def test_fit_does_not_mutate_input_dataframes(tiered_data):
    train_df, val_df = tiered_data
    original_train_cols = list(train_df.columns)
    original_val_cols = list(val_df.columns)
    router = TieredRouter(params=FAST_PARAMS)
    router.fit(train_df, val_df, FEATURE_COLS)
    assert list(train_df.columns) == original_train_cols
    assert list(val_df.columns) == original_val_cols


# ---------------------------------------------------------------------------
# TieredRouter.predict()
# ---------------------------------------------------------------------------


def test_predict_returns_series(fitted_router, tiered_data):
    _, val_df = tiered_data
    result = fitted_router.predict(val_df, FEATURE_COLS)
    assert isinstance(result, pd.Series)


def test_predict_index_matches_input(fitted_router, tiered_data):
    _, val_df = tiered_data
    result = fitted_router.predict(val_df, FEATURE_COLS)
    assert list(result.index) == list(val_df.index)


def test_predict_length_matches_input(fitted_router, tiered_data):
    _, val_df = tiered_data
    result = fitted_router.predict(val_df, FEATURE_COLS)
    assert len(result) == len(val_df)


def test_predict_tier3_cards_return_nan(fitted_router):
    # Tier 3 has no ML model — result must be NaN so the caller uses Cardmarket.
    tier3_df = pd.DataFrame(
        {
            "eur": [1_500.0, 3_000.0],
            "f1": [0.1, -0.2],
            "f2": [0.3, 0.5],
        }
    )
    result = fitted_router.predict(tier3_df, FEATURE_COLS)
    assert result.isna().all()


def test_predict_tier1_cards_return_finite_values(fitted_router):
    tier1_df = pd.DataFrame(
        {
            "eur": [5.0, 20.0, 80.0],
            "f1": [0.1, -0.3, 0.7],
            "f2": [0.5, 0.2, -0.1],
        }
    )
    result = fitted_router.predict(tier1_df, FEATURE_COLS)
    assert np.all(np.isfinite(result.values))


def test_predict_mixed_tiers_nan_only_for_tier3(fitted_router):
    # fitted_router has a tier2 model (fixture has ~56 tier2 train rows > 50 threshold).
    df = pd.DataFrame(
        {
            "eur": [10.0, 500.0, 2_000.0],
            "f1": [0.1, 0.2, 0.3],
            "f2": [-0.1, 0.5, -0.2],
        }
    )
    result = fitted_router.predict(df, FEATURE_COLS)
    assert np.isfinite(result.iloc[0])  # Tier 1 — model trained
    assert np.isfinite(result.iloc[1])  # Tier 2 — model trained (≥50 train rows)
    assert np.isnan(result.iloc[2])  # Tier 3 — no model, caller uses Cardmarket
