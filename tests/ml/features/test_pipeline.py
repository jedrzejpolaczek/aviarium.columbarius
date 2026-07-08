import duckdb
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.ml.features.pipeline import (
    BOOL_COLS,
    IMPUTE_MEDIAN_COLS,
    IMPUTE_ZERO_COLS,
    LEAKAGE_COLS,
    NUMERIC_PASS_COLS,
    build_feature_pipeline,
    build_inference_features,
    enrich_card_df,
    enrich_lag_df,
    get_feature_names,
    prepare_training_data,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _minimal_X(n: int = 3) -> pd.DataFrame:
    """DataFrame with exactly the columns the pipeline expects plus leakage columns.

    All NaN slots are intentional — the fixture verifies that imputation fills them.
    """
    data: dict = {}

    for col in IMPUTE_MEDIAN_COLS:
        # First row is NaN; the other rows provide a non-trivial median.
        data[col] = [np.nan, 2.0, 4.0][:n]

    for col in IMPUTE_ZERO_COLS:
        data[col] = [np.nan, 3.0, 1.0][:n]

    for col in NUMERIC_PASS_COLS:
        data[col] = [1.0, 2.0, 3.0][:n]

    for col in BOOL_COLS:
        data[col] = [0, 1, 0][:n]

    for col in LEAKAGE_COLS:
        data[col] = [99.0, 99.0, 99.0][:n]

    return pd.DataFrame(data)


@pytest.fixture
def sample_X():
    return _minimal_X()


@pytest.fixture
def fitted_pipeline(sample_X):
    pipe = build_feature_pipeline()
    pipe.fit(sample_X)
    return pipe


# ---------------------------------------------------------------------------
# build_feature_pipeline()
# ---------------------------------------------------------------------------


def test_build_feature_pipeline_returns_pipeline():
    assert isinstance(build_feature_pipeline(), Pipeline)


def test_build_feature_pipeline_has_features_step():
    pipe = build_feature_pipeline()
    assert "features" in pipe.named_steps


def test_fit_transform_returns_array(sample_X):
    pipe = build_feature_pipeline()
    result = pipe.fit_transform(sample_X)
    assert isinstance(result, np.ndarray)


def test_output_row_count_matches_input(sample_X):
    pipe = build_feature_pipeline()
    result = pipe.fit_transform(sample_X)
    assert result.shape[0] == len(sample_X)


def test_leakage_columns_excluded_from_output(sample_X):
    # remainder='drop' must silently exclude LEAKAGE_COLS.
    # The expected column count is everything except leakage columns.
    pipe = build_feature_pipeline()
    result = pipe.fit_transform(sample_X)
    expected_cols = (
        len(IMPUTE_MEDIAN_COLS)
        + len(IMPUTE_ZERO_COLS)
        + len(NUMERIC_PASS_COLS)
        + len(BOOL_COLS)
    )
    assert result.shape[1] == expected_cols


def test_median_imputation_fills_nan(sample_X):
    # IMPUTE_MEDIAN_COLS rows: [NaN, 2.0, 4.0] → median of non-NaN = 3.0
    # After fit_transform the NaN in row 0 should become 3.0.
    pipe = build_feature_pipeline()
    result = pipe.fit_transform(sample_X)
    # Median columns are first in the transformer order.
    median_col_idx = 0
    assert result[0, median_col_idx] == pytest.approx(3.0)


def test_zero_imputation_fills_nan(sample_X):
    # IMPUTE_ZERO_COLS rows: [NaN, 3.0, 1.0] → NaN in row 0 should become 0.
    pipe = build_feature_pipeline()
    result = pipe.fit_transform(sample_X)
    # Zero-imputed columns follow median-imputed columns.
    zero_col_idx = len(IMPUTE_MEDIAN_COLS)
    assert result[0, zero_col_idx] == pytest.approx(0.0)


def test_no_nans_in_imputed_columns_after_transform(sample_X):
    pipe = build_feature_pipeline()
    result = pipe.fit_transform(sample_X)
    # Imputed columns (first len(IMPUTE_MEDIAN_COLS) + len(IMPUTE_ZERO_COLS)) must be NaN-free.
    n_imputed = len(IMPUTE_MEDIAN_COLS) + len(IMPUTE_ZERO_COLS)
    assert not np.any(np.isnan(result[:, :n_imputed]))


# ---------------------------------------------------------------------------
# get_feature_names()
# ---------------------------------------------------------------------------


def test_get_feature_names_returns_list(fitted_pipeline, sample_X):
    names = get_feature_names(fitted_pipeline)
    assert isinstance(names, list)


def test_get_feature_names_all_strings(fitted_pipeline, sample_X):
    names = get_feature_names(fitted_pipeline)
    assert all(isinstance(n, str) for n in names)


def test_get_feature_names_no_transformer_prefix(fitted_pipeline, sample_X):
    # sklearn prefixes names like "impute_median__edhrec_rank".
    # get_feature_names() must strip that prefix for clean SHAP column labels.
    names = get_feature_names(fitted_pipeline)
    assert not any("__" in name for name in names)


def test_get_feature_names_length_matches_output_columns(fitted_pipeline, sample_X):
    names = get_feature_names(fitted_pipeline)
    output_cols = fitted_pipeline.transform(sample_X).shape[1]
    assert len(names) == output_cols


def test_get_feature_names_contains_imputed_columns(fitted_pipeline):
    names = get_feature_names(fitted_pipeline)
    for col in IMPUTE_MEDIAN_COLS + IMPUTE_ZERO_COLS:
        assert col in names


def test_get_feature_names_does_not_contain_leakage_columns(fitted_pipeline):
    names = get_feature_names(fitted_pipeline)
    for col in LEAKAGE_COLS:
        assert col not in names


# ---------------------------------------------------------------------------
# prepare_training_data()
# ---------------------------------------------------------------------------


@pytest.fixture
def three_source_dfs():
    """Minimal lag_df, card_df, and target_df sharing two UUIDs.

    uuid_a and uuid_b appear in all three DataFrames → both survive the inner join.
    uuid_orphan exists only in card_df → excluded from X, y.
    """
    lag_df = pd.DataFrame(
        {
            "uuid": ["uuid_a", "uuid_b"],
            "snapshot_date": ["2026-01-08", "2026-01-08"],
            "log_eur": [0.5, 1.2],
            "lag_1d": [0.4, 1.1],
            "rolling_mean_7d": [0.45, 1.15],
        }
    )
    card_df = pd.DataFrame(
        {
            "uuid": ["uuid_a", "uuid_b", "uuid_orphan"],
            "rarity_ord": [1.0, 3.0, 2.0],
            "is_reserved": [0, 1, 0],
            "price_ath": [1.0, 5.0, 2.0],  # leakage column
        }
    )
    target_df = pd.DataFrame(
        {
            "uuid": ["uuid_a", "uuid_b"],
            "log_return_7d": [0.05, -0.03],
        }
    )
    return lag_df, card_df, target_df


def test_prepare_training_data_returns_tuple(three_source_dfs):
    lag_df, card_df, target_df = three_source_dfs
    result = prepare_training_data(lag_df, card_df, target_df)
    assert isinstance(result, tuple) and len(result) == 2


def test_prepare_training_data_X_is_dataframe(three_source_dfs):
    lag_df, card_df, target_df = three_source_dfs
    X, _ = prepare_training_data(lag_df, card_df, target_df)
    assert isinstance(X, pd.DataFrame)


def test_prepare_training_data_y_is_series(three_source_dfs):
    lag_df, card_df, target_df = three_source_dfs
    _, y = prepare_training_data(lag_df, card_df, target_df)
    assert isinstance(y, pd.Series)


def test_prepare_training_data_row_count_is_inner_join(three_source_dfs):
    # uuid_orphan exists only in card_df → excluded; only uuid_a and uuid_b remain.
    lag_df, card_df, target_df = three_source_dfs
    X, y = prepare_training_data(lag_df, card_df, target_df)
    assert len(X) == 2
    assert len(y) == 2


def test_prepare_training_data_leakage_cols_dropped(three_source_dfs):
    lag_df, card_df, target_df = three_source_dfs
    X, _ = prepare_training_data(lag_df, card_df, target_df)
    for col in LEAKAGE_COLS:
        assert col not in X.columns


def test_prepare_training_data_target_not_in_X(three_source_dfs):
    lag_df, card_df, target_df = three_source_dfs
    X, _ = prepare_training_data(lag_df, card_df, target_df)
    assert "log_return_7d" not in X.columns


def test_prepare_training_data_y_values_correct(three_source_dfs):
    lag_df, card_df, target_df = three_source_dfs
    _, y = prepare_training_data(lag_df, card_df, target_df)
    assert set(y.values) == {0.05, -0.03}


# ---------------------------------------------------------------------------
# build_inference_features()
# ---------------------------------------------------------------------------


def _make_gold_db():
    """Create minimal in-memory Gold DuckDB with required tables.

    gold_card_features now includes is_legendary, which is derived from
    original_supertypes by GoldFeatureBuilders.build_card_features() and stored
    in the Gold table. Tests that call build_inference_features() require this
    column to be present so the pipeline can read it directly from the table.
    """
    con = duckdb.connect(":memory:")
    con.execute("""
        CREATE TABLE gold_price_features (
            uuid VARCHAR, snapshot_date DATE, eur DOUBLE,
            edhrec_rank DOUBLE, foil_premium DOUBLE
        )
    """)
    con.execute("""
        INSERT INTO gold_price_features VALUES
        ('uuid-1', '2025-01-10', 10.0, 100.0, 1.1),
        ('uuid-2', '2025-01-10', 20.0, 50.0, 1.0)
    """)
    con.execute("""
        CREATE TABLE gold_card_features (
            uuid VARCHAR, name VARCHAR, rarity VARCHAR,
            print_count INTEGER, mana_value DOUBLE, format_count INTEGER,
            is_reserved BOOLEAN, is_legendary BOOLEAN, is_commander_legal BOOLEAN,
            edhrec_rank DOUBLE, edhrec_saltiness DOUBLE, foil_premium DOUBLE
        )
    """)
    con.execute("""
        INSERT INTO gold_card_features VALUES
        ('uuid-1', 'Lightning Bolt', 'common', 50, 1.0, 3, false, false, true, 100.0, 0.5, 1.1),
        ('uuid-2', 'Atraxa', 'mythic', 10, 4.0, 5, false, true, true, 50.0, 0.8, 1.0)
    """)
    return con


def test_build_inference_features_returns_dataframe():
    con = _make_gold_db()
    result = build_inference_features(con, "2025-01-10")
    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert "uuid" in result.columns
    assert "log_eur" in result.columns
    assert "rarity_ord" in result.columns


def test_build_inference_features_no_pandas_nullable_dtypes():
    con = _make_gold_db()
    result = build_inference_features(con, "2025-01-10")
    for col in result.columns:
        assert not hasattr(result[col].dtype, "numpy_dtype"), (
            f"Column {col!r} has pandas nullable dtype {result[col].dtype!r}"
        )


def test_build_inference_features_stub_columns_present():
    con = _make_gold_db()
    result = build_inference_features(con, "2025-01-10")
    assert "top8_appearances_30d" in result.columns
    assert "deck_pct" in result.columns
    assert (result["top8_appearances_30d"] == 0.0).all()
    assert (result["deck_pct"] == 0.0).all()


def test_build_inference_features_is_legendary_from_gold_table():
    """is_legendary must come from gold_card_features, not a stub False."""
    con = _make_gold_db()
    # Confirm the table rows have correct is_legendary values (not stub overrides)
    result = build_inference_features(con, "2025-01-10")
    assert "is_legendary" in result.columns
    # uuid-1 (Lightning Bolt) is not legendary
    bolt_row = result[result["uuid"] == "uuid-1"]
    assert bool(bolt_row.iloc[0]["is_legendary"]) is False
    # uuid-2 (Atraxa) is legendary — verifies True values pass through correctly
    atraxa_row = result[result["uuid"] == "uuid-2"]
    assert bool(atraxa_row.iloc[0]["is_legendary"]) is True


# ---------------------------------------------------------------------------
# enrich_card_df()
# ---------------------------------------------------------------------------


def test_enrich_card_df_adds_rarity_ord():
    card_df = pd.DataFrame({"uuid": ["u1"], "rarity": ["rare"]})
    result = enrich_card_df(card_df)
    assert "rarity_ord" in result.columns
    assert result.iloc[0]["rarity_ord"] == 2


def test_enrich_card_df_rarity_ord_all_values():
    card_df = pd.DataFrame(
        {
            "uuid": ["u1", "u2", "u3", "u4"],
            "rarity": ["common", "uncommon", "rare", "mythic"],
        }
    )
    result = enrich_card_df(card_df)
    assert list(result["rarity_ord"]) == [0, 1, 2, 3]


def test_enrich_card_df_adds_has_mtgjson_data_true():
    card_df = pd.DataFrame({"uuid": ["u1"], "rarity": ["common"]})
    result = enrich_card_df(card_df)
    assert "has_mtgjson_data" in result.columns
    assert result.iloc[0]["has_mtgjson_data"] == True  # noqa: E712


def test_enrich_card_df_adds_stub_zero_columns():
    card_df = pd.DataFrame({"uuid": ["u1"], "rarity": ["common"]})
    result = enrich_card_df(card_df)
    assert result.iloc[0]["top8_appearances_30d"] == 0.0
    assert result.iloc[0]["deck_pct"] == 0.0


def test_enrich_card_df_does_not_mutate_input():
    card_df = pd.DataFrame({"uuid": ["u1"], "rarity": ["common"]})
    _ = enrich_card_df(card_df)
    assert "rarity_ord" not in card_df.columns


# ---------------------------------------------------------------------------
# enrich_lag_df()
# ---------------------------------------------------------------------------


def test_enrich_lag_df_adds_log_eur():
    lag_df = pd.DataFrame({"eur": [9.0], "lag_1d": [8.0], "rolling_mean_7d": [8.5]})
    result = enrich_lag_df(lag_df)
    assert "log_eur" in result.columns
    assert result.iloc[0]["log_eur"] == pytest.approx(np.log1p(9.0))


def test_enrich_lag_df_applies_log_to_rolling_mean_7d():
    lag_df = pd.DataFrame({"eur": [5.0], "lag_1d": [4.0], "rolling_mean_7d": [4.5]})
    result = enrich_lag_df(lag_df)
    assert result.iloc[0]["rolling_mean_7d"] == pytest.approx(np.log1p(4.5))


def test_enrich_lag_df_adds_lag_1d_return():
    lag_df = pd.DataFrame({"eur": [5.0], "lag_1d": [4.0], "rolling_mean_7d": [4.5]})
    result = enrich_lag_df(lag_df)
    assert "lag_1d_return" in result.columns
    assert result.iloc[0]["lag_1d_return"] == pytest.approx((5.0 - 4.0) / 4.0)


def test_enrich_lag_df_lag_1d_return_nan_when_lag_is_zero():
    lag_df = pd.DataFrame({"eur": [5.0], "lag_1d": [0.0], "rolling_mean_7d": [4.5]})
    result = enrich_lag_df(lag_df)
    assert pd.isna(result.iloc[0]["lag_1d_return"])


def test_enrich_lag_df_does_not_mutate_input():
    lag_df = pd.DataFrame({"eur": [5.0], "lag_1d": [4.0], "rolling_mean_7d": [4.5]})
    original_rolling = lag_df.iloc[0]["rolling_mean_7d"]
    _ = enrich_lag_df(lag_df)
    assert lag_df.iloc[0]["rolling_mean_7d"] == original_rolling


def test_enrich_lag_df_rolling_mean_7d_absent_adds_nan_column():
    """When rolling_mean_7d is missing, enrich_lag_df must add a NaN-filled column.

    rolling_mean_7d is listed in NUMERIC_PASS_COLS; a missing column would cause a
    ColumnTransformer key error at transform time.  The fallback prevents that.
    """
    lag_df = pd.DataFrame({"eur": [5.0], "lag_1d": [4.0]})  # no rolling_mean_7d
    result = enrich_lag_df(lag_df)
    assert "rolling_mean_7d" in result.columns
    assert pd.isna(result.iloc[0]["rolling_mean_7d"])


# ---------------------------------------------------------------------------
# Alignment: enrich_card_df and enrich_lag_df produce the same columns
# that build_inference_features adds inline — no training/serving skew.
# ---------------------------------------------------------------------------


def test_build_inference_features_uses_enrich_helpers_consistently():
    """Columns added by helpers must appear in the build_inference_features output."""
    con = _make_gold_db()
    result = build_inference_features(con, "2025-01-10")
    for col in ("rarity_ord", "has_mtgjson_data", "top8_appearances_30d", "deck_pct"):
        assert col in result.columns, f"Missing enriched card column: {col}"
    for col in ("log_eur", "rolling_mean_7d", "lag_1d_return"):
        assert col in result.columns, f"Missing enriched lag column: {col}"
