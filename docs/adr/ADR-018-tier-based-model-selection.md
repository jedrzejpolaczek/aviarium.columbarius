# ADR-018: Tier-Based Model Selection Strategy

## Context

MTG card prices span six orders of magnitude — from €0.01 bulk commons to €50,000
Black Lotus. A single global model trained on this distribution has two structural problems:

1. **Statistical**: the Pareto distribution (α = 1.303, confirmed in statistical_properties/01)
   means the top 0.85% of cards by price account for a disproportionate share of absolute
   error. A global MAE hides poor performance exactly where financial exposure is highest.

2. **Data scarcity at the tail**: Model Preparation 03 power analysis showed that Tier 3
   (>€1,000) has approximately 70 training rows. At n = 70, statistical power is ~60% —
   below the 80% threshold needed for reliable ML generalisation. Training LightGBM on
   70 rows of high-variance Pareto data produces an overfit, unreliable model.

Three price segments were identified in EDA 02:

| Tier | Price range  | Card count | % of catalogue |
|------|-------------|------------|----------------|
| 1    | < €100      | ~82,000    | 99.15%         |
| 2    | €100–€1,000 | ~560       | 0.68%          |
| 3    | > €1,000    | ~140       | 0.17%          |

## Decision

Use a different prediction strategy for each tier:

- **Tier 1 (<€100)**: LightGBM trained on log_return_7d with walk-forward CV.
  Sufficient data volume; gradient boosting handles the feature interactions well.

- **Tier 2 (€100–€1,000)**: Bayesian hierarchical model with partial pooling across
  the rarity × set_type group. Partial pooling borrows strength from the group mean
  when a specific card has few observations — addressing the moderate data scarcity
  without discarding the available signal.

- **Tier 3 (>€1,000)**: Direct price lookup on Cardmarket at prediction time.
  ~140 cards, all on the Reserved List or Power Nine — price drivers are speculation
  and scarcity, not the tabular features the model was trained on. No ML model is
  justified here given the power analysis result.

## Consequences

### Positive
- Metrics are always reported per tier (`evaluate_per_tier` in metrics.py), making
  model quality transparent where it matters financially.
- Each tier uses the approach most appropriate for its data volume and error cost.
- Tier 3 lookup is cheap to implement and impossible to overfit.
- The Bayesian model for Tier 2 produces calibrated uncertainty intervals, not just
  point predictions — useful for high-value cards where the cost of being wrong is high.

### Negative
- Three separate prediction paths increase operational complexity.
- Tier boundaries are fixed thresholds; a card reprinted from Tier 2 into Tier 1
  switches strategy discontinuously. This is acceptable given the rarity of such events.

### Neutral
- `evaluate_per_tier()` in metrics.py enforces this separation at evaluation time —
  there is no code path that computes a global aggregate metric.
- The tier assignment function (`_assign_tier` in metrics.py) is the single source
  of truth for boundaries; changing a threshold requires only one edit.

## Why Not a Single Global Model

A global LightGBM trained on all tiers would:
- Be dominated by Tier 1 volume (99.15% of rows), producing effectively a Tier 1 model
  with degraded performance on Tier 2 and Tier 3.
- Achieve misleadingly low global MAE while potentially having Tier 3 MAE an order
  of magnitude higher — a metric that would not surface in standard evaluation.
- Require the model to simultaneously learn low-variance bulk card dynamics and the
  speculation-driven dynamics of Power Nine and Reserved List cards — distributions
  that have nothing in common.

## Why Not Two Tiers (ML / No-ML)

Splitting only at €1,000 would leave Tier 2 (€100–€1,000) inside the ML model.
The power analysis shows Tier 2 has ~560 rows — enough for LightGBM, but cards in
this range have meaningfully different feature relationships than Tier 1 (e.g.
`is_reserved` drives nearly all variance). A Bayesian hierarchical model with
an explicit `is_reserved` prior is more appropriate than a gradient boosting model
that must infer this structure from 560 examples.
