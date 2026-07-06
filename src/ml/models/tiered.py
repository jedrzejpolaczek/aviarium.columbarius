"""
Routes each card to the appropriate model based on its current EUR price.

WHY THREE TIERS (from CDA and Statistical Properties):
Log-price variance differs 4.6× between Rare and Common (Levene W=2981, p≈0).
A single global model has miscalibrated errors for expensive cards.

TIER BOUNDARIES:
  Tier 1 (<€100):      LightGBM — 99.15% of cards, sufficient volume.
  Tier 2 (€100–€1000): LightGBM — 0.68% of cards, train only if ≥50 rows.
  Tier 3 (>€1000):     No ML model — ~139 cards, power analysis shows 60%
                        statistical power (below the 80% threshold from MP-03).
                        Returns NaN; caller falls back to Cardmarket lookup.

Tier is assigned from the CURRENT price, not the predicted future price.
"""

import numpy as np
import pandas as pd

from src.logger import get_logger
from src.ml.models.lightgbm_model import LightGBMParams, LightGBMPriceModel

logger = get_logger(__name__)


TIER1_MAX_EUR = 100.0
TIER2_MAX_EUR = 1_000.0


def assign_tier(eur: float) -> int:
    """Map a card's current EUR price to its pricing tier.

    NaN and None are treated as Tier 1 (safe default — assume cheap card).
    Tier boundires are described in module docstring.

    Args:
        eur: Current EUR price of the card.

    Returns:
        1, 2, or 3.
    """
    if eur is None or (isinstance(eur, float) and np.isnan(eur)):
        return 1
    if eur >= TIER2_MAX_EUR:
        return 3
    if eur >= TIER1_MAX_EUR:
        return 2
    return 1


class TieredRouter:
    """
    Trains and stores separate LightGBM models for Tier 1 and Tier 2.
    Tier 3 always returns NaN — the API layer falls back to Cardmarket.

    model_tier1: LightGBMPriceModel for cards priced <€100.
    model_tier2: LightGBMPriceModel for cards priced €100–€1000.
                 May be None if fewer than 50 training rows were available.
    """

    MIN_TIER2_ROWS = 50

    def __init__(self, params: LightGBMParams | None = None) -> None:
        self._params = params  # None → default LightGBMParams used inside fit()
        self.model_tier1: LightGBMPriceModel | None = None
        self.model_tier2: LightGBMPriceModel | None = None

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        feature_cols: list[str],
        target_col: str = "log_return_7d",
    ) -> "TieredRouter":
        """Train per-tier models on training data, validate on val data.

        Args:
            train_df:     Training DataFrame including 'eur' and target_col.
            val_df:       Validation DataFrame with the same columns.
            feature_cols: Feature column names passed to LightGBM.
            target_col:   Column to predict (default: log_return_7d).

        Returns:
            self
        """
        train_df = train_df.copy()
        val_df = val_df.copy()
        train_df["_tier"] = train_df["eur"].apply(assign_tier)
        val_df["_tier"] = val_df["eur"].apply(assign_tier)

        for tier, attr in [(1, "model_tier1"), (2, "model_tier2")]:
            t_mask = train_df["_tier"] == tier
            v_mask = val_df["_tier"] == tier

            if t_mask.sum() < (self.MIN_TIER2_ROWS if tier == 2 else 1):
                logger.info(
                    "[TieredRouter] Tier %d: only %d training rows — skipping.",
                    tier,
                    t_mask.sum(),
                )
                continue

            model = LightGBMPriceModel(self._params)
            model.fit(
                train_df.loc[t_mask, feature_cols],
                train_df.loc[t_mask, target_col],
                val_df.loc[v_mask, feature_cols],
                val_df.loc[v_mask, target_col],
            )
            setattr(self, attr, model)
            logger.info(
                "[TieredRouter] Tier %d: trained on %d rows.", tier, t_mask.sum()
            )

        return self

    def predict(self, df: pd.DataFrame, feature_cols: list[str]) -> pd.Series:
        """Predict log_return_7d for every card. Tier 3 returns NaN.

        Args:
            df:           DataFrame including 'eur' column for tier assignment.
            feature_cols: Feature columns matching those used in fit().

        Returns:
            pd.Series of predicted log_return_7d, indexed like df.
            NaN for Tier 3 cards.
        """
        result = pd.Series(np.nan, index=df.index)
        tiers = df["eur"].apply(assign_tier)

        for tier, attr in [(1, "model_tier1"), (2, "model_tier2")]:
            model = getattr(self, attr)
            if model is None:
                continue
            mask = tiers == tier
            if mask.sum() == 0:
                continue
            result.loc[mask] = model.predict(df.loc[mask, feature_cols])

        return result
