# ADR-028: SHAP (TreeSHAP) for Model Interpretability

**Date:** 2026-07-08
**Status:** Accepted

## Context

`src/ml/evaluation/shap_analysis.py` explains individual `LightGBMPriceModel`
predictions — not just *that* a card's price is predicted to rise, but *why*
(e.g. "edhrec_saltiness is high (+0.8 log-return units), is_reserved (+2.2)").
This capability and its algorithm choice (`shap.TreeExplainer`, not a
model-agnostic SHAP variant or a different interpretability library) were
never written down outside the module's own docstring.

`shap_analysis.py` also uses SHAP output as a confound check: per Bayesian
Analysis 02's finding that `print_count` confounds `edhrec_saltiness`
(popular cards are reprinted more *and* tend to be saltier), the module
verifies `print_count` doesn't dominate SHAP importance when `edhrec_saltiness`
is in the model — if it does, that signals saltiness was dropped or
mis-encoded upstream.

## Decision

Use `shap.TreeExplainer` against the trained LightGBM `Booster`.

`TreeExplainer` implements the exact TreeSHAP algorithm: it walks every
decision tree in the ensemble and computes each feature's exact marginal
contribution (not a sampled/approximate Shapley estimate) in polynomial time,
because LightGBM's tree structure makes the exact computation tractable —
unlike `KernelExplainer` (SHAP's model-agnostic variant), which would need to
approximate Shapley values via sampling and is orders of magnitude slower for
no accuracy gain on a tree model. ADR-017's choice of LightGBM (over XGBoost,
scikit-learn's ensembles, and TensorFlow) is a precondition for this ADR:
TreeExplainer's exactness and speed depend specifically on the tree-ensemble
structure ADR-017 committed to.

## Consequences

### Positive

- Exact (not sampled) attribution — reproducible, no approximation noise
  between runs on the same model/data.
- Runs in polynomial time for tree ensembles specifically, fast enough for
  per-card waterfall plots at inference time, not just offline analysis.
- Doubles as a confound-detection tool via the `print_count`/`edhrec_saltiness`
  check, catching a specific class of feature-encoding regression that a
  plain feature-importance ranking (e.g. LightGBM's built-in gain importance)
  would not surface as clearly.

### Negative

- `TreeExplainer` is LightGBM/tree-model-specific — if the project ever
  swaps to a non-tree model (ADR-017 chose LightGBM over XGBoost/TensorFlow;
  a future change to e.g. a neural model would need `DeepExplainer` or
  `KernelExplainer` instead, with different performance characteristics).
- SHAP is a real dependency with its own pinning constraints — ADR-011
  (uv package manager) documents a real case where `shap`'s transitive
  dependencies resolve to different versions on different platforms
  (Intel macOS vs. others), constraining the lower bound `uv lock` can
  satisfy. That is an operational cost of depending on this library at all.

## Alternatives Considered

| Approach | Reason rejected |
|---|---|
| `shap.KernelExplainer` (model-agnostic) | Approximates Shapley values via sampling — slower and less exact than TreeExplainer for a tree model, with no offsetting benefit since LightGBM is exactly the case TreeExplainer is built for. |
| LightGBM's built-in `feature_importance(importance_type="gain")` (already used elsewhere, e.g. `LightGBMPriceModel.feature_importance` in `lightgbm_model.py`) only, no SHAP | Global importance only — cannot explain a single card's individual prediction ("why does the model think *this* card will rise"), which is the actual use case `shap_analysis.py` serves. Gain importance is kept as a coarser, cheaper companion metric, not a replacement. |

## Affected ADRs

- **ADR-017** — LightGBM's choice over XGBoost/TensorFlow is a precondition
  for TreeExplainer's exactness/speed; this ADR is downstream of that one.
- **ADR-011** — Already documents the `shap` dependency's platform-specific
  resolution constraint (Intel macOS); this ADR adds the *why SHAP at all*
  context that constraint was missing.
