import numpy as np
import pandas as pd
import pytest

from src.ml.recommendation.similarity import SIMILARITY_FEATURES, CardSimilarityIndex


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_card_df(n: int = 30, seed: int = 0) -> pd.DataFrame:
    """Minimal card DataFrame with all required columns."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "uuid": [f"card_{i}" for i in range(n)],
            "name": [f"Card {i}" for i in range(n)],
            "eur": rng.uniform(0.5, 50.0, n),
            "rarity_ord": rng.integers(0, 4, n),
            "mana_value": rng.integers(0, 10, n),
            "color_count": rng.integers(0, 5, n),
            "color_identity_count": rng.integers(0, 5, n),
            "format_count": rng.integers(0, 8, n),
            "is_legendary": rng.integers(0, 2, n).astype(float),
            "is_commander_legal": rng.integers(0, 2, n).astype(float),
            "is_modern_legal": rng.integers(0, 2, n).astype(float),
        }
    )


@pytest.fixture
def small_df():
    return _make_card_df(30)


@pytest.fixture
def fitted_index(small_df):
    return CardSimilarityIndex(n_neighbors=5).fit(small_df)


# ---------------------------------------------------------------------------
# SIMILARITY_FEATURES constant
# ---------------------------------------------------------------------------


def test_similarity_features_is_list():
    assert isinstance(SIMILARITY_FEATURES, list)


def test_similarity_features_non_empty():
    assert len(SIMILARITY_FEATURES) > 0


def test_similarity_features_contains_mana_value():
    assert "mana_value" in SIMILARITY_FEATURES


def test_similarity_features_contains_rarity_ord():
    assert "rarity_ord" in SIMILARITY_FEATURES


# ---------------------------------------------------------------------------
# CardSimilarityIndex.fit()
# ---------------------------------------------------------------------------


def test_fit_returns_self(small_df):
    idx = CardSimilarityIndex()
    result = idx.fit(small_df)
    assert result is idx


def test_fit_sets_knn(small_df):
    idx = CardSimilarityIndex().fit(small_df)
    assert idx.knn is not None


def test_fit_sets_scaler(small_df):
    idx = CardSimilarityIndex().fit(small_df)
    assert idx.scaler is not None


def test_fit_sets_cards_df(small_df):
    idx = CardSimilarityIndex().fit(small_df)
    assert idx.cards_df is not None
    assert len(idx.cards_df) == len(small_df)


def test_fit_sets_x_scaled(small_df):
    idx = CardSimilarityIndex().fit(small_df)
    assert idx.X_scaled is not None
    assert idx.X_scaled.shape == (len(small_df), len(SIMILARITY_FEATURES))


def test_fit_cards_df_has_uuid_column(small_df):
    idx = CardSimilarityIndex().fit(small_df)
    assert idx.cards_df is not None
    assert "uuid" in idx.cards_df.columns


def test_fit_cards_df_has_eur_column(small_df):
    idx = CardSimilarityIndex().fit(small_df)
    assert idx.cards_df is not None
    assert "eur" in idx.cards_df.columns


def test_fit_handles_nan_features():
    df = _make_card_df(20)
    df.loc[0, "mana_value"] = float("nan")
    idx = CardSimilarityIndex().fit(df)
    assert idx.X_scaled is not None  # NaN filled with 0, no error raised


# ---------------------------------------------------------------------------
# CardSimilarityIndex.find_similar()
# ---------------------------------------------------------------------------


def test_find_similar_returns_dataframe(fitted_index):
    result = fitted_index.find_similar("card_0")
    assert isinstance(result, pd.DataFrame)


def test_find_similar_returns_n_neighbors_rows(fitted_index):
    result = fitted_index.find_similar("card_0")
    assert len(result) == 5


def test_find_similar_has_similarity_score_column(fitted_index):
    result = fitted_index.find_similar("card_0")
    assert "similarity_score" in result.columns


def test_find_similar_scores_between_0_and_1(fitted_index):
    result = fitted_index.find_similar("card_0")
    assert (result["similarity_score"] >= -1e-9).all()
    assert (result["similarity_score"] <= 1.0 + 1e-9).all()


def test_find_similar_does_not_include_query_card(fitted_index):
    result = fitted_index.find_similar("card_0")
    assert "card_0" not in result["uuid"].values


def test_find_similar_raises_value_error_for_unknown_uuid(fitted_index):
    with pytest.raises(ValueError, match="not found"):
        fitted_index.find_similar("nonexistent_uuid")


def test_find_similar_raises_runtime_error_before_fit():
    idx = CardSimilarityIndex()
    with pytest.raises(RuntimeError, match="fit()"):
        idx.find_similar("card_0")


def test_find_similar_has_uuid_column(fitted_index):
    result = fitted_index.find_similar("card_0")
    assert "uuid" in result.columns


def test_find_similar_result_index_is_reset(fitted_index):
    result = fitted_index.find_similar("card_0")
    assert list(result.index) == list(range(len(result)))
