"""
Finds cards similar to a given card based on static attributes.

WHY COSINE SIMILARITY:
Measures the angle between vectors, not Euclidean distance.
A card with CMC=3 and a card with CMC=4 can still be "similar" if they share
the same color profile and legality flags — cosine captures that.

WHY SCALER (needed here, unlike LightGBM):
CMC (0–16) and rarity_ord (0–3) live on different scales.
Without StandardScaler, mana_value dominates cosine similarity.

WHY NearestNeighbors instead of a full matrix:
A full 300k × 300k matrix = 360 GB of memory — impossible.
NearestNeighbors computes similarity only for one query card — O(n) not O(n²).
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


SIMILARITY_FEATURES = [
    "rarity_ord",
    "mana_value",
    "color_count",
    "color_identity_count",
    "format_count",
    "is_legendary",
    "is_commander_legal",
    "is_modern_legal",
]


class CardSimilarityIndex:
    """Nearest-neighbour similarity index for MTG cards using cosine distance.

    After a one-time fit() call the index answers find_similar() queries
    without recomputing the full distance matrix.
    """

    def __init__(self, n_neighbors: int = 10) -> None:
        # Default of 10 is for standalone/notebook use. Production overrides
        # this to 50 at startup (app/main.py::_build_similarity_index) — see
        # ADR-023 Decision 2 for why 50 was chosen as the /similar endpoint's
        # max result count.
        self.n_neighbors = n_neighbors
        self.scaler: StandardScaler | None = None
        self.knn: NearestNeighbors | None = None
        self.cards_df: pd.DataFrame | None = None
        self.X_scaled: np.ndarray | None = None

    def fit(self, df: pd.DataFrame) -> "CardSimilarityIndex":
        """Build the similarity index from a card feature DataFrame.

        Args:
            df: DataFrame containing at least 'uuid', 'name', 'eur', and all
                columns in SIMILARITY_FEATURES. Missing values are filled with 0.

        Returns:
            self, allowing method chaining.
        """
        X = df[SIMILARITY_FEATURES].fillna(0)

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # brute-force is preferred for cosine; tree-based indices don't help here
        self.knn = NearestNeighbors(
            n_neighbors=self.n_neighbors + 1,  # +1 because the query card is included
            metric="cosine",
            algorithm="brute",
        )
        self.knn.fit(X_scaled)

        self.cards_df = (
            df[["uuid", "name", "eur"] + SIMILARITY_FEATURES]
            .copy()
            .reset_index(drop=True)
        )
        self.X_scaled = X_scaled
        return self

    def find_similar(self, card_uuid: str) -> pd.DataFrame:
        """Return the n_neighbors most similar cards for the given card UUID.

        Args:
            card_uuid: UUID of the query card; must be present in the fitted index.

        Returns:
            DataFrame with columns uuid, name, eur, SIMILARITY_FEATURES, and
            similarity_score (1.0 = identical, 0.0 = no similarity).

        Raises:
            RuntimeError: fit() has not been called yet.
            ValueError: card_uuid not found in the fitted index.
        """
        if self.knn is None or self.cards_df is None or self.X_scaled is None:
            raise RuntimeError("Call fit() before find_similar().")

        matches = self.cards_df[self.cards_df["uuid"] == card_uuid]
        if matches.empty:
            raise ValueError(f"Card UUID {card_uuid!r} not found in the index.")

        idx = matches.index[0]
        card_vector = self.X_scaled[idx].reshape(1, -1)
        distances, indices = self.knn.kneighbors(card_vector)

        # Index 0 is the query card itself (distance ≈ 0) — skip it
        neighbor_indices = indices[0][1:]
        neighbor_distances = distances[0][1:]

        result = self.cards_df.iloc[neighbor_indices].copy()
        result["similarity_score"] = (
            1.0 - neighbor_distances
        )  # cosine dist → similarity
        result_df: pd.DataFrame = result.reset_index(drop=True)
        return result_df
