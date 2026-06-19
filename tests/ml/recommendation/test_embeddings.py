import numpy as np
import pandas as pd
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

from src.ml.recommendation.embeddings import (
    build_tfidf_embeddings,
    combine_with_card_features,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_oracle_texts(n: int = 50) -> pd.Series:
    """Diverse MTG-like ability texts, repeated enough to pass min_df=5."""
    templates = [
        "Draw a card at the beginning of your upkeep.",
        "Counter target spell unless its controller pays two.",
        "Destroy target creature. It cannot be regenerated.",
        "Flying. When this creature enters the battlefield, draw a card.",
        "Trample. This creature gets plus two plus two until end of turn.",
        "Exile target permanent. Its controller gains three life.",
        "Search your library for a basic land card and put it into play.",
        "Add one mana of any color to your mana pool.",
        "Create a token that is a copy of target creature you control.",
        "Each player discards their hand then draws seven cards.",
    ]
    texts = [templates[i % len(templates)] for i in range(n)]
    return pd.Series(texts)


@pytest.fixture
def oracle_texts():
    return _make_oracle_texts(50)


# ---------------------------------------------------------------------------
# build_tfidf_embeddings()
# ---------------------------------------------------------------------------


def test_build_tfidf_returns_tuple(oracle_texts):
    result = build_tfidf_embeddings(oracle_texts, max_features=20)
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_build_tfidf_first_element_is_ndarray(oracle_texts):
    embeddings, _ = build_tfidf_embeddings(oracle_texts, max_features=20)
    assert isinstance(embeddings, np.ndarray)


def test_build_tfidf_second_element_is_vectorizer(oracle_texts):
    _, vectorizer = build_tfidf_embeddings(oracle_texts, max_features=20)
    assert isinstance(vectorizer, TfidfVectorizer)


def test_build_tfidf_embeddings_shape_n_rows(oracle_texts):
    embeddings, _ = build_tfidf_embeddings(oracle_texts, max_features=20)
    assert embeddings.shape[0] == len(oracle_texts)


def test_build_tfidf_embeddings_shape_n_cols_max_features(oracle_texts):
    max_f = 20
    embeddings, _ = build_tfidf_embeddings(oracle_texts, max_features=max_f)
    # Columns <= max_features (vocabulary may be smaller than max_features)
    assert embeddings.shape[1] <= max_f


def test_build_tfidf_embeddings_no_nan(oracle_texts):
    embeddings, _ = build_tfidf_embeddings(oracle_texts, max_features=20)
    assert not np.isnan(embeddings).any()


def test_build_tfidf_handles_nan_in_input():
    texts = pd.Series(["Draw a card", None, "Counter target spell"] * 20)
    embeddings, _ = build_tfidf_embeddings(texts, max_features=10)
    assert embeddings.shape[0] == len(texts)


def test_build_tfidf_all_zeros_row_is_nan_input():
    # NaN → empty string → no tokens → zero vector row
    texts = pd.Series([None] + ["Draw a card"] * 49)
    embeddings, _ = build_tfidf_embeddings(texts, max_features=10)
    assert embeddings[0].sum() == 0.0


def test_build_tfidf_vectorizer_can_transform_new_texts(oracle_texts):
    _, vectorizer = build_tfidf_embeddings(oracle_texts, max_features=20)
    new_texts = ["Draw a card at the beginning"]
    transformed = vectorizer.transform(new_texts).toarray()
    assert transformed.shape[0] == 1


def test_build_tfidf_values_non_negative(oracle_texts):
    embeddings, _ = build_tfidf_embeddings(oracle_texts, max_features=20)
    assert (embeddings >= 0).all()


# ---------------------------------------------------------------------------
# combine_with_card_features()
# ---------------------------------------------------------------------------


def test_combine_returns_ndarray():
    features = np.ones((5, 3))
    embeddings = np.ones((5, 10))
    result = combine_with_card_features(features, embeddings)
    assert isinstance(result, np.ndarray)


def test_combine_correct_shape():
    features = np.ones((5, 3))
    embeddings = np.ones((5, 10))
    result = combine_with_card_features(features, embeddings)
    assert result.shape == (5, 13)


def test_combine_concatenates_columns():
    features = np.array([[1.0, 2.0], [3.0, 4.0]])
    embeddings = np.array([[0.5, 0.5, 0.5], [0.1, 0.2, 0.3]])
    result = combine_with_card_features(features, embeddings)
    expected = np.hstack([features, embeddings])
    np.testing.assert_array_almost_equal(result, expected)


def test_combine_single_row():
    features = np.array([[1.0, 2.0, 3.0]])
    embeddings = np.array([[0.5, 0.5]])
    result = combine_with_card_features(features, embeddings)
    assert result.shape == (1, 5)


def test_combine_preserves_feature_values():
    features = np.array([[10.0, 20.0]])
    embeddings = np.array([[0.1, 0.2]])
    result = combine_with_card_features(features, embeddings)
    assert result[0, 0] == 10.0
    assert result[0, 1] == 20.0


def test_combine_preserves_embedding_values():
    features = np.array([[1.0, 2.0]])
    embeddings = np.array([[0.7, 0.3]])
    result = combine_with_card_features(features, embeddings)
    assert abs(result[0, 2] - 0.7) < 1e-9
    assert abs(result[0, 3] - 0.3) < 1e-9
