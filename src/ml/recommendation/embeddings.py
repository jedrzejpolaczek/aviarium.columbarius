"""
Builds vector representations of cards from their oracle_text ability text.

WHY TEXT EMBEDDINGS:
Static attributes (rarity, CMC) don't differentiate cards with similar abilities.
TF-IDF distinguishes "Draw a card" from "Draw two cards" and "Counter target spell".
MTG vocabulary is small and repetitive — TF-IDF works very well here.

APPROACH 1 — TF-IDF (start here):
- sklearn TfidfVectorizer on oracle_text
- max_features=500 is sufficient for the MTG vocabulary
- Fast, deterministic, scales to 300k cards

APPROACH 2 — sentence-transformers (optional, later):
- from sentence_transformers import SentenceTransformer
- model = SentenceTransformer('all-MiniLM-L6-v2')
- Understands semantics: "remove from the game" ≈ "exile"
- Slower, benefits from GPU for large corpora

COMBINING WITH CARD ATTRIBUTES (optional):
combined = np.hstack([card_features_normalized, text_embeddings])
This combined matrix can be passed to CardSimilarityIndex instead of attributes alone.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def build_tfidf_embeddings(
    oracle_texts: pd.Series,
    max_features: int = 500,
) -> tuple[np.ndarray, TfidfVectorizer]:
    """Build TF-IDF embeddings for card ability texts.

    Args:
        oracle_texts: Series of card oracle texts (may contain NaN).
        max_features: Vocabulary size to retain (500 covers the MTG lexicon well).

    Returns:
        Tuple of (embeddings_array, fitted_vectorizer):
          embeddings_array -- dense np.ndarray of shape (n_cards, max_features)
          vectorizer       -- fitted TfidfVectorizer for transforming new cards
    """
    texts = oracle_texts.fillna("").tolist()

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words="english",  # remove "the", "a", "of" etc.
        min_df=5,  # ignore words appearing fewer than 5 times
        sublinear_tf=True,  # apply log(tf) instead of raw tf
    )
    matrix = vectorizer.fit_transform(texts)
    return matrix.toarray(), vectorizer


def combine_with_card_features(
    card_features: np.ndarray,
    text_embeddings: np.ndarray,
) -> np.ndarray:
    """Concatenate static card features with text embeddings into one vector.

    Args:
        card_features: Scaled static features array of shape (n_cards, n_features).
            Must be normalised with StandardScaler before calling — otherwise
            numeric features will dominate the TF-IDF dimensions.
        text_embeddings: TF-IDF embedding array of shape (n_cards, n_tfidf).

    Returns:
        Combined array of shape (n_cards, n_features + n_tfidf).
    """
    return np.hstack([card_features, text_embeddings])
