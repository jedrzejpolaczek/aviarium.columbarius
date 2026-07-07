"""Similar-cards recommendation endpoint.

Finds the N most similar cards to a given card using cosine distance over
static card attributes (rarity, mana cost, colour profile, format legality).
See ``src.ml.recommendation.similarity`` for the full feature list and the
reasoning behind cosine distance vs Euclidean.

The ``CardSimilarityIndex`` is built once at startup (see ``app.main.lifespan``)
with ``n_neighbors=50``.  Every request to this endpoint is O(card_count) —
a single nearest-neighbour query against the pre-scaled feature matrix.

URL encoding:
    Card names with spaces must be percent-encoded in the URL, e.g.
    ``GET /similar/Force%20of%20Will``.
"""

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query

from app.dependencies import get_similarity_index, require_match
from app.schemas.responses import SimilarCard, SimilarCardsResponse
from src.ml.recommendation.similarity import CardSimilarityIndex


router = APIRouter(prefix="/similar", tags=["recommendation"])


@router.get("/{card_name}", response_model=SimilarCardsResponse)
def find_similar_cards(
    card_name: str,
    n: int = Query(
        default=10, ge=1, le=50, description="Number of similar cards to return"
    ),
    similarity_index: CardSimilarityIndex | None = Depends(get_similarity_index),
) -> SimilarCardsResponse:
    """Return the N most similar cards to the requested card.

    Looks up the card's UUID in the similarity index's internal card DataFrame,
    then delegates to ``CardSimilarityIndex.find_similar`` which runs a
    brute-force cosine nearest-neighbour query.  Results are sorted by
    similarity_score descending (1.0 = identical, 0.0 = completely dissimilar).

    The ``n`` query parameter caps the returned list; at most 50 cards are ever
    returned (the index was built with ``n_neighbors=50`` at startup).

    Args:
        card_name:        Exact card name, URL-decoded by FastAPI.
        n:                Number of results to return. Between 1 and 50,
                          default 10.
        similarity_index: ``CardSimilarityIndex`` instance injected via
                          ``get_similarity_index`` dependency.

    Returns:
        SimilarCardsResponse containing the queried card name and a list of
        SimilarCard objects (name, uuid, current_price, similarity_score).

    Raises:
        HTTPException 404: Card not found in the similarity index.
        HTTPException 503: Similarity index not available (startup failed).
    """
    if similarity_index is None or similarity_index.cards_df is None:
        raise HTTPException(503, detail="Similarity index not available.")

    matches = require_match(similarity_index.cards_df, "name", card_name, "Card")

    card_uuid = str(matches.iloc[0]["uuid"])

    try:
        similar_df = similarity_index.find_similar(card_uuid)
    except ValueError as exc:
        raise HTTPException(404, detail=str(exc)) from exc

    cards = [
        SimilarCard(
            name=str(row["name"]),
            uuid=str(row["uuid"]),
            current_price=float(row["eur"]) if pd.notna(row["eur"]) else None,
            similarity_score=float(row["similarity_score"]),
        )
        for _, row in similar_df.head(n).iterrows()
    ]

    return SimilarCardsResponse(card_name=card_name, similar_cards=cards)
