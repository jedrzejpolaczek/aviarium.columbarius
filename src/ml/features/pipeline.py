"""
Assembles card features into a single sklearn Pipeline ready for training.

WHY NO SCALER:
LightGBM uses decision trees — they split on threshold values, not distances.
Scaling features with StandardScaler or MinMaxScaler has no effect on tree-based
models and adds unnecessary complexity. Exception: CardSimilarityIndex
(recommendation/similarity.py) uses cosine distance and does need scaling.

IMPUTATION STRATEGY:
- edhrec_rank:        82% NaN for cards without MTGJson data → median per rarity
- foil_premium:       ~0.4% NaN (card has no foil printing) → 1.0 (no premium)
- edhrec_saltiness:   NaN for unranked cards → median
- top8_appearances_30d: NaN means zero tournament appearances
- deck_pct:           NaN means card appears in no EDHREC decks

LEAKAGE COLUMNS (confirmed in model_preparation/01):
price_ath, price_atl, days_with_price use OVER (PARTITION BY uuid) without a
ROWS frame — they scan the entire partition including future rows.
remainder='drop' in ColumnTransformer silently excludes them.

ENRICHMENT HELPERS:
enrich_card_df() and enrich_lag_df() are public, shared between
build_inference_features() (this module) and walk_forward_cv()
(src/ml/training/trainer.py), keeping the serving and training feature
matrices identical (no training/serving skew). Public because both are
legitimate cross-module consumers, not internal-only helpers.
"""

import duckdb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.ml.features.lag import build_lag_features


# Leakage confirmed in notebooks/model_preparation/01_leakage_audit.ipynb.
# ColumnTransformer remainder='drop' excludes any column not listed below.
LEAKAGE_COLS = ["price_ath", "price_atl", "days_with_price"]

# Columns imputed with the training-set median (MNAR — missing not at random).
IMPUTE_MEDIAN_COLS = ["edhrec_rank", "edhrec_saltiness", "foil_premium"]

# Columns imputed with zero (NaN semantically means "zero occurrences").
IMPUTE_ZERO_COLS = ["top8_appearances_30d", "deck_pct"]

# Numeric columns passed through without transformation.
# LightGBM handles NaN natively, so lag features with short history are safe.
# lag_1d_return is intentionally excluded — computed by enrich_lag_df for exploratory use,
# not selected as a model feature.
NUMERIC_PASS_COLS = [
    "rarity_ord",
    "print_count",
    "mana_value",
    "format_count",
    "lag_1d",
    "lag_7d",
    "lag_14d",
    "lag_30d",
    "rolling_mean_7d",
    "rolling_std_14d",
    "rolling_min_30d",
    "rolling_max_30d",
    "momentum_7d",
    "log_eur",
]

# Boolean columns (0/1); no imputation needed — pipeline fills missing booleans
# with False at the Silver layer before this point.
BOOL_COLS = [
    "is_reserved",
    "is_legendary",
    "is_commander_legal",
    "has_mtgjson_data",
]


def build_feature_pipeline() -> Pipeline:
    """Return a fitted-ready sklearn Pipeline with imputation and passthrough steps.

    The ColumnTransformer uses remainder='drop', which automatically excludes
    LEAKAGE_COLS and any column not listed in the transformers — no explicit
    drop step is needed.

    Returns:
        Unfitted Pipeline with a single 'features' step (ColumnTransformer).
    """
    ct = ColumnTransformer(
        transformers=[
            ("impute_median", SimpleImputer(strategy="median"), IMPUTE_MEDIAN_COLS),
            (
                "impute_zero",
                SimpleImputer(strategy="constant", fill_value=0),
                IMPUTE_ZERO_COLS,
            ),
            ("passthrough", "passthrough", NUMERIC_PASS_COLS + BOOL_COLS),
        ],
        remainder="drop",
    )
    return Pipeline([("features", ct)])


def get_feature_names(pipeline: Pipeline) -> list[str]:
    """Return feature names in the order produced by the pipeline's ColumnTransformer.

    Strips the transformer prefix added by sklearn (e.g. 'impute_median__edhrec_rank'
    becomes 'edhrec_rank') so the names are usable directly in SHAP and MLflow.

    Args:
        pipeline: Fitted Pipeline returned by build_feature_pipeline().

    Returns:
        List of clean feature name strings.
    """
    ct = pipeline.named_steps["features"]
    return [name.split("__", 1)[-1] for name in ct.get_feature_names_out()]


def prepare_training_data(
    lag_df: pd.DataFrame,
    card_df: pd.DataFrame,
    target_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Join lag features, static card features, and target into X and y.

    Uses inner joins on uuid — only cards present in all three DataFrames are
    returned. Cards missing lag history, card metadata, or a target value are
    silently excluded.

    Args:
        lag_df:    Output of build_lag_features(): one row per card per snapshot.
        card_df:   gold_card_features: static card attributes (rarity, foil, etc.).
        target_df: Output of build_target(): uuid + log_return_7d.

    Returns:
        (X, y) where X is the feature DataFrame and y is the log_return_7d Series.
        LEAKAGE_COLS are dropped from X if present.
    """
    df = lag_df.merge(card_df, on="uuid", how="inner").merge(
        target_df, on="uuid", how="inner"
    )
    df = df.drop(columns=[c for c in LEAKAGE_COLS if c in df.columns])
    y = df["log_return_7d"]
    X = df.drop(columns=["log_return_7d"])
    return X, y


def _normalise_nullable_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Convert pandas nullable extension types to numpy-native types for sklearn."""
    df = df.copy()
    for col in df.columns:
        dtype = df[col].dtype
        if hasattr(dtype, "numpy_dtype"):
            ndtype = dtype.numpy_dtype
            if ndtype == np.bool_:
                df[col] = df[col].fillna(False).astype(bool)
            elif np.issubdtype(ndtype, np.integer):
                df[col] = df[col].fillna(0).astype(ndtype)
            else:
                df[col] = df[col].astype(float)
    return df


_RARITY_MAP = {"common": 0, "uncommon": 1, "rare": 2, "mythic": 3}


def enrich_card_df(card_df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns to a gold_card_features DataFrame.

    Applies the same enrichments in both training (walk_forward_cv) and serving
    (build_inference_features) to prevent training/serving skew.

    Columns added:
        rarity_ord          — ordinal encoding of rarity (common=0 … mythic=3).
        has_mtgjson_data    — True for all gold cards (MTGJson coverage is a
                              prerequisite for a card to reach the Gold layer).
        top8_appearances_30d — stub 0.0 (tournament data not yet integrated).
        deck_pct            — stub 0.0 (EDHREC deck percentage not yet integrated).

    BOOL_COLS present on input (is_reserved, is_legendary, is_commander_legal)
    are cast to float64, matching has_mtgjson_data below. build_feature_pipeline()
    passes BOOL_COLS through the ColumnTransformer in the same block as
    NUMERIC_PASS_COLS; pandas' DataFrame.to_numpy() silently degrades that block
    to dtype=object whenever it mixes bool with numeric dtypes (unlike plain
    numpy, which upcasts bool+float to float64 without complaint), and once one
    ColumnTransformer block is object-dtype, hstack propagates object to the
    *entire* output array — which LightGBM's dtype validation then rejects with
    "pandas dtypes must be int, float or bool" even though every value is
    numeric. Casting here, the single enrichment choke point shared by
    walk_forward_cv() and build_inference_features(), keeps the passthrough
    block uniformly numeric so this never bites downstream.

    Args:
        card_df: gold_card_features DataFrame (one row per card printing).

    Returns:
        New DataFrame with the four extra columns appended. The input is not
        mutated.
    """
    card_df = card_df.copy()
    if "rarity" in card_df.columns:
        card_df["rarity_ord"] = card_df["rarity"].map(_RARITY_MAP)
    else:
        card_df["rarity_ord"] = np.nan
    card_df["has_mtgjson_data"] = True
    card_df["top8_appearances_30d"] = 0.0
    card_df["deck_pct"] = 0.0
    for col in BOOL_COLS:
        if col in card_df.columns:
            card_df[col] = card_df[col].astype(float)
    return card_df


def enrich_lag_df(lag_df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns to a lag-features DataFrame.

    Applies the same log transforms and return calculation in both training
    (walk_forward_cv) and serving (build_inference_features) to prevent
    training/serving skew.

    Columns added / modified:
        log_eur          — log1p(eur); log-scale current price for the pipeline.
        rolling_mean_7d  — replaced in-place with log1p(rolling_mean_7d) so the
                           pipeline sees a log-scale rolling average consistent
                           with log_eur.
        lag_1d_return    — (eur - lag_1d) / lag_1d; day-over-day return ratio.
                           NaN when lag_1d is zero or missing. Not currently used
                           by the sklearn pipeline but retained for exploratory
                           analysis.

    Args:
        lag_df: Output of build_lag_features() for a single snapshot date.

    Returns:
        New DataFrame with the derived columns appended / updated. The input is
        not mutated.
    """
    lag_df = lag_df.copy()
    if "eur" in lag_df.columns:
        lag_df["log_eur"] = np.log1p(lag_df["eur"])
    else:
        lag_df["log_eur"] = np.nan
    # rolling_mean_7d is overwritten (not kept alongside log_eur) because the pipeline
    # uses only the log-scale version and retaining both would create a redundant feature.
    if "rolling_mean_7d" in lag_df.columns:
        lag_df["rolling_mean_7d"] = np.log1p(lag_df["rolling_mean_7d"])
    else:
        lag_df["rolling_mean_7d"] = np.nan
    if "eur" in lag_df.columns and "lag_1d" in lag_df.columns:
        lag_df["lag_1d_return"] = (lag_df["eur"] - lag_df["lag_1d"]) / lag_df[
            "lag_1d"
        ].replace(0, np.nan)
    else:
        lag_df["lag_1d_return"] = np.nan
    return lag_df


def build_inference_features(
    conn: duckdb.DuckDBPyConnection,
    snapshot_date: str,
) -> pd.DataFrame:
    """Build the full feature matrix for inference at a given snapshot date.

    Constructs the same matrix used during training: lag features joined with
    static card attributes, with log transforms applied and stub columns for
    tournament/EDHREC data not yet integrated into the Gold layer.

    Enrichment is delegated to enrich_lag_df() and enrich_card_df(), which are
    also called by walk_forward_cv() in trainer.py, guaranteeing that training
    and serving see identical features (no training/serving skew).

    Used by both app/main.py lifespan startup and src/monitoring/retraining.py
    to guarantee that training and serving see identical features.

    Args:
        conn: Open DuckDB connection with gold_price_features and
              gold_card_features in scope.
        snapshot_date: ISO date string for the snapshot to build features at.

    Returns:
        DataFrame with one row per card, ready for sklearn pipeline transform.
        Pandas nullable extension types (BooleanDtype, Int64Dtype) are
        normalised to numpy-native types.
    """
    lag_df = enrich_lag_df(build_lag_features(conn, snapshot_date))
    card_df = enrich_card_df(conn.execute("SELECT * FROM gold_card_features").df())

    X = lag_df.merge(card_df, on="uuid", how="inner").reset_index(drop=True)
    return _normalise_nullable_dtypes(X)
