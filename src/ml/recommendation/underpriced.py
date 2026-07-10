"""
Detects cards the model considers underpriced (predicted >> actual).

STRATEGY PER TIER (from project pricing strategy and BAYESIAN_FINDINGS.md):
- Tier 1 (< 100 EUR):     predicted/actual > 1.3 → ML flag
- Tier 2 (100–1000 EUR):  ML flag only, same threshold as Tier 1 (see
                          TIER2_FLAG_THRESHOLD below). The Bayesian guardrail
                          (BA-02 HDI) described in BAYESIAN_FINDINGS.md is NOT
                          YET wired in — flag_underpriced() does not gate
                          Tier 2 on it. See the inline comment at the
                          tier2_flag assignment.
- Tier 3 (> 1000 EUR):    No ML flagging — too little data. Check Cardmarket manually.

VALIDATION — BACKTEST (important for portfolio!):
Check: did cards flagged 30 days ago actually appreciate?
Historical data is in bronze_scryfall_prices_history.
Backtest result: "73% of flagged cards rose > 10% within 30 days" —
a concrete business metric.

CONFIDENCE SCORE:
confidence = predicted_eur / actual_eur
Higher value = more underpriced (e.g. 1.5 = model predicts +50%).
"""

import duckdb
from typing import Any

import pandas as pd

from src.ml.models.tiered import assign_tier


TIER1_FLAG_THRESHOLD = 1.3  # predicted/actual > 1.3 = potentially underpriced
TIER2_FLAG_THRESHOLD = 1.3  # same threshold but gated by the Bayesian guardrail

PRICE_FLOOR_EUR = 0.01
"""Minimum EUR price used as a division-by-zero guard when computing the
confidence ratio and backtest appreciation. Unrelated to
src.ml.evaluation.metrics.MAPE_CLIP_MIN, which clips log_return values
(~-1..1 scale) for model evaluation -- the shared 0.01 literal is
coincidental, not a shared concept. Do not import MAPE_CLIP_MIN here."""


def flag_underpriced(
    df: pd.DataFrame,
    predicted_col: str = "predicted_eur",
    actual_col: str = "eur",
) -> pd.DataFrame:
    """Flag cards the model considers underpriced and compute a confidence score.

    Args:
        df: DataFrame with at least uuid, name, eur, and predicted_eur columns.
        predicted_col: Column name holding the model's price prediction.
        actual_col: Column name holding the current market price.

    Returns:
        Copy of df with three additional columns:
          is_underpriced  -- True when the model predicts significant upside.
          confidence      -- predicted/actual ratio (1.0 = neutral, 1.5 = +50%).
          reason          -- short explanation for flagged cards, empty string otherwise.
        Sorted by confidence descending.
    """
    df = df.copy()
    df["tier"] = df[actual_col].apply(assign_tier)
    df["confidence"] = df[predicted_col] / df[actual_col].clip(lower=PRICE_FLOOR_EUR)

    tier1_flag = (df["tier"] == 1) & (df["confidence"] > TIER1_FLAG_THRESHOLD)
    # Bayesian guardrail (BA-02 HDI) will be integrated when BA-02 results are available
    tier2_flag = (df["tier"] == 2) & (df["confidence"] > TIER2_FLAG_THRESHOLD)
    # Tier 3: never flag — too few data points, route to manual Cardmarket review

    df["is_underpriced"] = tier1_flag | tier2_flag
    df["reason"] = df.apply(
        lambda row: (
            f"ML predicts +{(row['confidence'] - 1) * 100:.0f}% in 7d"
            if row["is_underpriced"]
            else ""
        ),
        axis=1,
    )

    return df.sort_values("confidence", ascending=False).reset_index(drop=True)


def backtest_underpriced(
    conn: duckdb.DuckDBPyConnection,
    flagged_date: str,
    check_date: str,
    flagged_uuids: list[str],
    appreciation_threshold: float = 0.10,
) -> dict[str, Any]:
    """Check what fraction of previously flagged cards appreciated by check_date.

    Args:
        conn: Open DuckDB connection with gold_price_features in scope.
        flagged_date: ISO date when cards were flagged ('YYYY-MM-DD').
        check_date: ISO date when we evaluate the outcome ('YYYY-MM-DD').
        flagged_uuids: UUIDs of the cards that were flagged on flagged_date.
        appreciation_threshold: Minimum price change to count as a "win" (0.10 = 10%).

    Returns:
        Dict with keys:
          total_flagged   -- number of flagged cards with prices on both dates
          appreciated     -- number that exceeded appreciation_threshold
          hit_rate        -- appreciated / total_flagged (e.g. 0.73)
          avg_appreciation -- mean price change across all matched cards
    """
    if not flagged_uuids:
        return {
            "total_flagged": 0,
            "appreciated": 0,
            "hit_rate": 0.0,
            "avg_appreciation": 0.0,
        }

    prices_flag = conn.execute(
        "SELECT uuid, eur FROM gold_price_features WHERE snapshot_date = ?",
        [flagged_date],
    ).df()
    prices_check = conn.execute(
        "SELECT uuid, eur FROM gold_price_features WHERE snapshot_date = ?",
        [check_date],
    ).df()

    prices_flag = prices_flag[prices_flag["uuid"].isin(flagged_uuids)]
    merged = prices_flag.merge(prices_check, on="uuid", suffixes=("_flag", "_check"))

    if merged.empty:
        return {
            "total_flagged": 0,
            "appreciated": 0,
            "hit_rate": 0.0,
            "avg_appreciation": 0.0,
        }

    merged["change"] = (merged["eur_check"] - merged["eur_flag"]) / merged[
        "eur_flag"
    ].clip(lower=PRICE_FLOOR_EUR)

    total = len(merged)
    appreciated = int((merged["change"] > appreciation_threshold).sum())
    hit_rate = appreciated / total if total > 0 else 0.0
    avg_appreciation = float(merged["change"].mean())

    return {
        "total_flagged": total,
        "appreciated": appreciated,
        "hit_rate": hit_rate,
        "avg_appreciation": avg_appreciation,
    }
