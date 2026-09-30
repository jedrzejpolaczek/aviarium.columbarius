# ADR-023: Card Recommendation and Underpriced Detection Strategy

**Date:** 2026-06-19
**Status:** Amended 2026-09-30 — the Tier 2 Bayesian guardrail was never wired in, and the unverified "73%" backtest figure is replaced by a measured one; see [Backtest](#backtest).

## Context

The API exposes two recommendation endpoints:

- `/similar/{card_name}` — returns the N most similar cards by static attributes
- `/underpriced` — returns cards the model considers underpriced (predicted price >> market price)

Three design decisions were made with no prior documentation:

1. **Similarity metric**: cosine vs Euclidean distance
2. **Index type**: full matrix vs NearestNeighbors
3. **Underpriced detection**: thresholds and tier-based strategy

## Decision

### Decision 1 — Cosine similarity for card similarity

Use **cosine distance** (`NearestNeighbors` with `metric="cosine"`), not Euclidean distance.

Similarity features (from `SIMILARITY_FEATURES` in `src/ml/recommendation/similarity.py`):
`rarity_ord`, `mana_value`, `color_count`, `color_identity_count`, `format_count`,
`is_legendary`, `is_commander_legal`, `is_modern_legal`

Cosine similarity measures the angle between vectors, not absolute distance. A card with
CMC=3 and a card with CMC=4 can be "similar" if they share the same color profile and
legality flags — cosine captures that relationship. Euclidean distance would penalise the
CMC=1 difference heavily.

`StandardScaler` is applied before fitting because `mana_value` (range 0–16) and
`rarity_ord` (range 0–3) are on different scales. Without scaling, `mana_value` would
dominate the cosine similarity calculation.

### Decision 2 — Pre-built NearestNeighbors index (not a full similarity matrix)

Use `sklearn.neighbors.NearestNeighbors` with `algorithm="brute"`, pre-built at API
startup (`CardSimilarityIndex.fit()` in `src/ml/recommendation/similarity.py`).

A full similarity matrix (all-pairs) at 80k cards would require 80k × 80k × 8 bytes ≈
48 GB of memory — impossible on any reasonable server. `NearestNeighbors` computes
similarity only for the query card (O(n) per query).

The index is built with `n_neighbors=50` at startup. All `/similar` requests specify
`n ≤ 50`; handlers truncate with `.head(n)`. Re-fitting per request with a dynamic k
would take approximately 2 seconds per request, which is unacceptable.

### Decision 3 — Tier-based underpriced detection

Use a tier-based flagging strategy with `predicted_eur / actual_eur > 1.3` as the
threshold (from `src/ml/recommendation/underpriced.py`):

| Tier | Price range | Strategy |
|---|---|---|
| Tier 1 | < €100 | Flag if `confidence > 1.3` (ML signal alone sufficient) |
| Tier 2 | €100–€1,000 | Flag if `confidence > 1.3` AND Bayesian guardrail (BA-02 HDI) confirms signal — *the guardrail is not implemented; Tier 2 uses the Tier 1 rule* |
| Tier 3 | > €1,000 | Never flag — too little training data; route to manual Cardmarket review |

The `confidence` score is `predicted_eur / actual_eur` (clipped to ≥ 0.01 to avoid
division by zero). Its measured behaviour is in [Backtest](#backtest) below.

## Backtest

*Measured 2026-09-30 with `python -m scripts.backtest_underpriced`; per-date results in
`notebooks/ml_models/underpriced_backtest.csv`. This replaces an earlier sentence claiming
"73% of flagged cards rose > 10% within 30 days", for which no backtest existed.*

**Method.** For every date d that has snapshots on d−7 and d+7, a LightGBM model is trained
on snapshot d−7. Its target, the d−7 → d return, is known on day d, so the model uses
nothing a production retrain on d would not have. It scores every card on d, the
`flag_underpriced` rule behind `GET /underpriced` is applied, and the outcome is read at
d+7. The horizon is 7 days because that is what the model predicts. A 30-day horizon
cannot be measured on this history: the 2026-07-29 → 2026-09-02 ingestion gap leaves no
(d, d+30) pairs. A "hit" is a price rise of more than 10%. It is compared with the share
of *all* Tier 1–2 cards that rose more than 10% over the same week (the base rate).
Dates in the frozen-feed period (up to early July) are excluded: no card moved, and
nothing was flagged.

| | Evaluation dates | Flagged card-days | Hit rate | Base rate (all) | Base rate (< €1) |
|---|---:|---:|---:|---:|---:|
| All live dates | 14 | 10 755 | 36.4% | 17.4% | 20.2% |
| Dates whose training week spans the July price-feed switch | 3 | 10 012 | 34.9% | 14.5% | 16.9% |
| Clean dates | 11 | 743 | **55.6%** | 18.2% | 21.1% |

**What the numbers mean — read before quoting the hit rate:**

- **Every flag is a card priced under €1.** The rule never fired on a card worth €1 or more.
  So the fair comparison is the sub-€1 base rate, not the catalogue-wide one.
- **The flag mostly catches price dips that revert.** In a one-off breakdown of the clean
  dates, 86% of flagged cards had fallen more than 20% in the week before the flag,
  against 6% of eligible cards. Simply buying every sub-€1 card that fell more than 20%
  scores about 32%, so the model adds selection on top of the dip, but the dip is most of
  the signal.
- **The money involved is negligible.** Buying one copy of every flagged card and selling
  it a week later would have gained €46 across all 743 clean flags, about €0.06 per flag,
  before Cardmarket fees and shipping, which exceed that many times over.
- **93% of all flags come from three dates** whose training labels span the July
  price-feed switch. Those models learned "everything jumps" and flagged thousands of
  cards. This is an artefact of the data source, not skill.

**Conclusion.** The flag has a real, statistically clear edge over the base rate within its
price band, and no practical value: it identifies cent-level mean reversion in bulk cards,
not undervalued cards anyone would buy. To become useful it would need a minimum price, and
it should be re-measured on the cards that remain.

## Consequences

### Positive

- Cosine + `StandardScaler` gives intuitive similarity results — `Counterspell` finds
  other blue instant counterspells, not whatever card happens to be CMC=2.
- Pre-built index means `/similar` responses are O(n) scans with no startup penalty per
  request.
- Tier-based detection avoids false positives on expensive cards where the model has high
  uncertainty.
- `confidence` score is interpretable to users: "1.5 means the model predicts +50% in
  7 days."

### Negative

- Similarity is based on static attributes only — two cards with identical stats but
  completely different abilities (e.g. a 2/2 white creature vs a 2/2 blue creature) may
  be ranked as similar.
- TF-IDF `oracle_text` embeddings exist in `src/ml/recommendation/embeddings.py` but
  are not yet integrated into the production similarity index — combining attributes and
  text is a future improvement.
- The `n_neighbors=50` hard limit at startup means the `/similar` endpoint cannot return
  more than 50 results without restarting the server.

## Diagram

```mermaid
flowchart TD
    CLIENT["API Client"]

    subgraph Similar["/similar/{card_name}"]
        SL["Lookup card in\ngold_card_features"]
        SI["CardSimilarityIndex\n(pre-built at startup)"]
        SF["StandardScaler\n+ cosine NearestNeighbors\nn_neighbors=50"]
        SR["Return top-N results\n(.head(n))"]
        SL --> SI --> SF --> SR
    end

    subgraph Underpriced["/underpriced"]
        UL["Load all cards\ngold_card_features\n+ in-memory predictions"]
        UC["Compute confidence\npredicted_eur / actual_eur"]
        UT{"Tier?"}
        T1["Tier 1 (<€100)\nFlag if confidence > 1.3"]
        T2["Tier 2 (€100–€1k)\nFlag if confidence > 1.3\nAND BA-02 HDI confirms"]
        T3["Tier 3 (>€1k)\nSkip — route to\nmanual Cardmarket review"]
        UL --> UC --> UT
        UT -->|"< €100"| T1
        UT -->|"€100–€1k"| T2
        UT -->|"> €1k"| T3
    end

    CLIENT -->|"GET /similar/{card_name}"| SL
    CLIENT -->|"GET /underpriced"| UL
```

## Alternatives Considered

| Approach | Reason rejected |
|---|---|
| Euclidean distance | Sensitive to scale differences (CMC dominates); cosine captures relative profile more naturally |
| Full all-pairs similarity matrix | O(n²) memory — 80k × 80k at float64 = 48 GB; impossible on any reasonable server |
| Per-request dynamic k fitting | ~2 seconds to fit `NearestNeighbors` per request → 30-second timeouts |
| Uniform threshold across all tiers | Tier 3 (>€1k) cards have very few training examples; a false positive on a €2,000 card is a severe user experience failure |
| TF-IDF embeddings only (no static features) | Text embeddings don't capture format legality, rarity, or mana cost — a Commander staple and a casual card with the same text would be ranked as identical |
