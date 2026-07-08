# ADR-027: TF-IDF for Card-Similarity Text Embeddings

**Date:** 2026-07-08
**Status:** Accepted

## Context

`src/ml/recommendation/similarity.py`'s `CardSimilarityIndex` (ADR-023) ranks
cards by static attributes only (rarity, mana value, color identity, format
legality) — two cards with identical stats but completely different abilities
(e.g. a 2/2 white creature vs. a 2/2 blue creature) rank as equally similar.
`src/ml/recommendation/embeddings.py` was built to add a text-based similarity
signal from each card's `oracle_text`, but the choice of embedding method
(TF-IDF vs. a pretrained sentence-transformer model) was never written down
outside the module's own docstring.

## Decision

Use TF-IDF (`sklearn.feature_extraction.text.TfidfVectorizer`,
`max_features=500`) on `oracle_text`, not a sentence-transformers model.

MTG's rules-text vocabulary is small and highly repetitive across tens of
thousands of printings (a few hundred distinct keywords/templated phrases
cover the vast majority of card text), which is exactly the regime where
TF-IDF's bag-of-words counts perform well and a heavier contextual embedding
model gains little. TF-IDF is deterministic, has no GPU dependency, and fits
the project's existing dependency footprint (`scikit-learn` is already a core
dependency; `sentence-transformers` would be a new one).

`build_tfidf_embeddings()` and `combine_with_card_features()` exist in
`embeddings.py` but are **not yet wired into** `CardSimilarityIndex` — the
production `/similar` endpoint uses static attributes only today (see
ADR-023's "Negative consequences"). This ADR documents the embedding-method
choice for when that integration happens, not the integration itself.

## Consequences

### Positive

- No new dependency, no GPU requirement, deterministic output (same input
  text always produces the same vector, useful for reproducible tests).
- Fast enough to run over the full card catalogue without batching/GPU
  infrastructure.

### Negative

- TF-IDF is purely lexical — "remove from the game" and "exile" (which mean
  the same rules concept, phrased differently across MTG's templating
  history) get no similarity credit from token overlap alone. A
  sentence-transformer model would capture that semantic equivalence.
- `max_features=500` caps the vocabulary; rare keywords outside the top 500
  by document frequency contribute nothing to the embedding.

## Alternatives Considered

| Approach | Reason rejected |
|---|---|
| sentence-transformers (`all-MiniLM-L6-v2` or similar) | Understands semantic equivalence TF-IDF misses, but adds a new dependency, needs GPU to run at catalogue scale in reasonable time, and non-deterministic-feeling model updates (embedding drift across model versions) complicate reproducibility. Revisit only if TF-IDF similarity quality proves insufficient in practice — see `embeddings.py`'s docstring. |
| No text signal at all (static attributes only, current production state) | Leaves the "identical stats, different abilities" gap ADR-023 already flags as a known negative consequence; `embeddings.py` exists specifically to eventually close that gap. |

## Affected ADRs

- **ADR-023** — Decision 1 (cosine similarity over static `SIMILARITY_FEATURES`)
  is unaffected; this ADR only extends the *inputs* to a future combined
  similarity vector, documented there as a "future improvement," not a
  present integration.
