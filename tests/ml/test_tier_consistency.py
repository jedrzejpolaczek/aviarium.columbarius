"""Cross-module regression: assign_tier() output must agree everywhere it's consumed.

Round-4 maintainability audit (2026-07-09) found that TIER1_MAX_EUR/TIER2_MAX_EUR
(tiered.py), the tier column used in evaluate_per_tier (trainer.py), and
flag_underpriced's tier assignment (underpriced.py) all depend on assign_tier()
returning consistent results, with no test spanning all three consumers at once.
This pins the contract so a future change to tier boundaries can't silently
desync flag_underpriced() from assign_tier().
"""

import pandas as pd

from src.ml.models.tiered import TIER1_MAX_EUR, TIER2_MAX_EUR, assign_tier
from src.ml.recommendation.underpriced import flag_underpriced


def test_flag_underpriced_uses_same_tier_boundaries_as_assign_tier():
    df = pd.DataFrame(
        {
            "uuid": ["a", "b", "c"],
            "name": ["Card A", "Card B", "Card C"],
            "eur": [TIER1_MAX_EUR - 1, TIER1_MAX_EUR + 1, TIER2_MAX_EUR + 1],
            "predicted_eur": [2.0, 200.0, 2000.0],
        }
    )
    expected_tiers = df.set_index("uuid")["eur"].apply(assign_tier)

    result = flag_underpriced(df)

    actual_tiers = result.set_index("uuid")["tier"]
    assert actual_tiers.equals(expected_tiers.loc[actual_tiers.index])


def test_tier_3_is_never_flagged_regardless_of_confidence():
    df = pd.DataFrame(
        {
            "uuid": ["a"],
            "name": ["Expensive Card"],
            "eur": [TIER2_MAX_EUR + 1],
            "predicted_eur": [(TIER2_MAX_EUR + 1) * 5],  # huge confidence ratio
        }
    )

    result = flag_underpriced(df)

    assert (result["tier"] == 3).all()
    assert not result["is_underpriced"].any()
