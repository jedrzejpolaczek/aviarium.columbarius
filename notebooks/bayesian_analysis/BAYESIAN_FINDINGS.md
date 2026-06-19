# Bayesian Analysis — Consolidated Findings Report

_Covers notebooks 01–04 in `notebooks/bayesian_analysis/`. Last updated: 2026-06-08._

> Data context: **5 snapshots (2026-06-04 → 2026-06-08)**. BA-01 (prior elicitation) has full results on all 82,413 cards. BA-02 (hierarchical model) ran MCMC on a 5,000-card demo subsample — production run on full dataset is pending (see TODOs). BA-03 (time series) is fully deferred (~2026-07-03). BA-04 (hypothesis testing) has complete Bayesian re-tests of CDA findings on n=82,163 cards.

---

## Table of Contents

1. [Prior Elicitation (BA-01)](#ba-01--prior-elicitation)
2. [Hierarchical Price Model (BA-02)](#ba-02--hierarchical-price-model)
3. [Bayesian Time Series (BA-03)](#ba-03--bayesian-time-series)
4. [Bayesian Hypothesis Testing (BA-04)](#ba-04--bayesian-hypothesis-testing)
5. [Retest Schedule](#retest-schedule)
6. [TODOs & Blockers](#todos--blockers)
7. [Decisions Log](#decisions-log)

---

## BA-01 — Prior Elicitation

_n = 82,413 cards. Prior elicitation uses snapshots ≤ 2026-06-05 (2 snapshots); inference uses snapshots > 2026-06-05 (3 snapshots). Split at mid-date 2026-06-05 to prevent double-dipping._

### Prior Specification

Normal(μ, σ) in log1p(EUR) space. μ = log1p(empirical median); σ = **2× empirical standard deviation** (default width confirmed by sensitivity analysis).

### Rarity-Level Priors

| Rarity | n | Emp. median EUR | μ_prior | Emp. σ | σ_prior | Implied 95% CI (EUR) |
|---|---|---|---|---|---|---|
| common | 51,492 | €0.12 | 0.1133 | 0.513 | 1.025 | [−€0.86, €7.7] |
| uncommon | 43,424 | €0.18 | 0.1655 | 0.681 | 1.362 | [−€0.92, €17.0] |
| rare | 57,102 | €0.65 | 0.5008 | 1.103 | 2.206 | [−€0.98, €135.1] |
| mythic | 12,308 | €3.02 | 1.3913 | 1.046 | 2.091 | [−€0.94, €262.5] |

### Prior Predictive Check

- P(negative prices in prior predictive): common 45.5%, uncommon 44.9%, rare 41.5%, mythic 25.4%
- Empirical prices within 95% prior CI: common **98.6%**, uncommon **98.4%**, rare **98.5%**, mythic **99.8%**

> ⚠️ Negative prices in the prior predictive are expected — Normal priors on log1p space have support below 0. In the full model, the data likelihood anchors estimates to valid regions. A Truncated-Normal or exponentiated parameterisation would eliminate this at the cost of implementation complexity.

### Sensitivity Analysis (mythic, three σ widths)

| Variant | Implied median | P(>€100) | % obs in 95% CI |
|---|---|---|---|
| Narrow (1×) | €3.02 | 0.1% | **94%** — FAILS |
| **Default (2×)** | **€3.06** | **6.3%** | **100%** — ✅ |
| Wide (3×) | €2.97 | 15.1% — implausible | 100% |

Narrow fails to cover all observed mythic prices; Wide generates implausible >€100 rate for a random mythic. **Default (2×) is the chosen width.**

### Feature Priors

| Parameter | μ | σ | Direction | Source |
|---|---|---|---|---|
| beta_is_reserved | +2.00 | 0.80 | positive | CDA NB01 H3: RL 20.3× at rare → 2.10 log-units in log1p space |
| beta_format_count | −0.05 | 0.20 | negative | CDA NB02 H2: more formats → lower price (partial ρ=−0.077) |
| beta_log_print | +0.10 | 0.30 | positive | CDA NB02 H5: demand-driven reprinting paradox |
| beta_foil_premium | −0.20 | 0.50 | negative | EDA NB04: Spearman −0.334 (nonlinear; MI=0.897) |
| beta_is_full_art | +0.50 | 0.40 | positive | CDA NB05 H4: 8.1× at rare; confirmed all rarities |
| beta_is_legendary | +0.20 | 0.30 | conditional (see B-08) | CDA NB01 H2 (unadjusted: NEGATIVE; log_print controlled: POSITIVE) |

> ⚠️ **beta_is_legendary prior direction mismatch:** BA-01 set a positive prior based on CDA-05 H2 (which was initially misread). BA-04 H2 confirms `is_legendary` is **NEGATIVE** (cheaper within rarity) in an unadjusted model. BA-02 (hierarchical, controlling for `log_print`) finds **POSITIVE**. The direction depends on whether reprint count is in the model. Update the prior direction in the next iteration if the model does not include `log_print`; keep positive if it does.

> ✅ **Feature priors for confirmed features added to `priors_config.json`:** `edhrec_saltiness` (Normal(+0.2, 0.3)), `edhrec_rank` (**SAFE P-18** — snapshotted daily, joined date-exact; Normal(−0.1, 0.3) on log_rank), `set_type` (category-specific one-hot priors, expansion as reference), `rarity_ord` (hierarchy grouping variable in BA-02, not linear covariate — documented), `in_tournament` (Normal(+0.3, 0.4)), `top8_appearances_30d` (Normal(+0.15, 0.2) on log1p), `mana_value` (Normal(−0.05, 0.15)), `color_identity_count` (Normal(0, 0.2), non-monotone), `primary_type` (category-specific one-hot priors, Creature as reference). **Still missing from BA-02 model code** — add before next production run.

Output saved to: `priors_config.json`

---

## BA-02 — Hierarchical Price Model

_Demo subsample: 5,000 cards. 360 sets, 4 rarities (common=1,560, uncommon=1,346, rare=1,727, mythic=367). Inference snapshots > 2026-06-05._

### Model Structure

Three-level partial pooling hierarchy:

```
mu_rarity[r]    ~ Normal(0, 1)          — hyperprior (4 rarity means)
sigma_rarity    ~ HalfNormal(0.5)       — between-set variation within rarity
mu_set[s]       ~ Normal(mu_rarity[rarity_of_set[s]], sigma_rarity)
beta_*          ~ priors from BA-01     — 6 feature coefficients
sigma_obs       ~ HalfNormal(0.5)       — within-set residual noise
log1p(EUR_i)    ~ Normal(mu_set[s_i] + Σ(beta * X_i), sigma_obs)
```

### MCMC Diagnostics

- Sampler: NUTS, 2 chains × 1,000 draws (500 tuning)
- Divergences: **0**
- R-hat < 1.01: **YES** (max = 1.0030)
- ESS > 400: **MOSTLY** — beta_log_print ESS = 380 (marginally below threshold)
- Posterior predictive 90% CI coverage: **92.7%** (target ~90% — slightly overconservative, acceptable)

> ⚠️ Only 2 chains run. Recommend 4 chains × 2,000 draws for production inference.

### Parameter Posteriors

| Parameter | Mean | SD | 89% ETI | ESS | R-hat |
|---|---|---|---|---|---|
| beta_reserved | 2.245 | 0.102 | [2.081, 2.413] | 2146 | 1.001 |
| beta_log_print | 0.125 | 0.010 | [0.109, 0.141] | **381** | 1.003 |
| beta_format | −0.195 | 0.015 | [−0.220, −0.171] | 604 | 1.001 |
| beta_foil_p | −0.206 | 0.019 | [−0.236, −0.177] | 695 | 1.003 |
| beta_full_art | 0.641 | 0.051 | [0.560, 0.722] | 1807 | 1.000 |
| beta_legendary | 0.276 | 0.030 | [0.229, 0.325] | 2442 | 1.003 |
| sigma_rarity | 0.631 | 0.029 | [0.585, 0.680] | 1553 | 1.001 |
| sigma_obs | 0.634 | 0.007 | [0.624, 0.645] | 2870 | 1.000 |

### Feature Effect Posteriors (94% HDI)

| Feature | Median | 94% HDI | P(>0) | Confirmed direction |
|---|---|---|---|---|
| is_reserved | +1.813 | [+1.617, +2.015] | 100% | ✅ positive |
| log_print | +0.139 | [+0.119, +0.158] | 100% | ✅ positive |
| format_count | −0.225 | [−0.254, −0.196] | 0% | ✅ negative |
| foil_premium | −0.195 | [−0.231, −0.158] | 0% | ✅ negative |
| is_full_art | +0.650 | [+0.539, +0.762] | 100% | ✅ positive |
| is_legendary | +0.344 | [+0.282, +0.399] | 100% | ✅ positive (once log_print controlled) |

**On beta_legendary:** The positive direction here (controlled for reprints) contrasts with BA-04 H2 (unadjusted, negative). Once reprint count is in the model, the legendary label itself carries a price premium — legendary cards are printed fewer times per set, and that scarcity is already absorbed by `log_print`. Without that control, the reprint confound flips the sign.

### Rarity-Level Posteriors (median EUR, 94% HDI)

| Rarity | Median EUR | 94% HDI |
|---|---|---|
| common | €0.29 | [€0.15, €0.46] |
| uncommon | €0.39 | [€0.19, €0.61] |
| rare | €0.67 | [€0.50, €0.87] |
| mythic | €1.09 | [€0.64, €1.62] |

sigma_rarity = 0.595 (posterior median) / 0.631 (posterior mean, from parameter table above) — a new unseen set is shrunk toward its rarity mean with this level of uncertainty. The mean is pulled slightly high by the right tail of the half-normal. Use the **median (0.595)** as the point estimate for the partial-pooling interpretation.

> ⚠️ **BA-02 missing confirmed features:** The current model uses only 6 covariates (is_reserved, log_print, format_count, log_foil_premium, is_full_art, is_legendary). Confirmed strong features not yet in the model: `edhrec_saltiness`, `edhrec_rank`, `set_type` (ε²=0.113), `rarity_ord` (ε²=0.396), `in_tournament`, `top8_appearances_30d`. The 5,000-card demo uses rarity as a grouping variable (hierarchy) but not as a covariate — for gradient boosting `rarity_ord` must be an explicit input feature. Add these before running production inference.

### Top Sets by Posterior mu_set

| Set | mu_set | Name |
|---|---|---|
| SUM | 4.62 | Summer Magic (very rare misprint run) |
| LEA | 4.35 | Alpha (first print run) |
| ARN | 2.75 | Arabian Nights |
| LEB | 2.75 | Beta |

All top sets are 1993–1994 — vintage/collectible value dominates the set-level partial pooling.

---

## BA-03 — Bayesian Time Series

_Status: **FULLY DEFERRED** — 5 snapshots available; minimum required: 30._

### Model Blueprint (ready for production run)

- Model: Local Level (Bayesian random walk with observation noise)
- Parameters: σ_eta (trend drift), σ_eps (observation noise)
- Equations: μ_1 ~ Normal(log1p(EUR_1), σ_eta); μ_t = μ_{t-1} + η_t; log1p(EUR_t) = μ_t + ε_t
- Sampler: NUTS, draws=2000, tune=1000, chains=2, target_accept=0.9
- Forecast function: predict_future(trace, n_future=30) — returns (n_samples, 30) credible interval array
- Expected 90% HDI coverage: ~90%
- Expected σ_eta: Tier 1 stable cards ~0.01–0.05; Tier 3 power cards ~0.05–0.15

### Metrics Defined (deferred)

- MAE_bayes = |median_pred_7d − y_test_7|
- MAE_naive = |y_train_last − y_test_7| (last-observed naive forecast)
- HDI_cover = fraction of test points inside 90% HDI

### Fallback Prior for Current Phase

Use BA-02 hierarchical posterior as a price prior for all cards until BA-03 can be trained.

**Re-run after: ~2026-07-03** (≥30 snapshots)

---

## BA-04 — Bayesian Hypothesis Testing

_n = 82,163 cards (latest snapshot). ROPE: (−0.1, +0.1) log1p(EUR) units ≈ ±10% price difference. Note: 250 fewer cards than EDA/CDA/MP (82,413) — these are cards with `eur IS NULL` in the latest snapshot (digital-only cards, foreign-exclusive promos, or cards with no Cardmarket listing). BA-04 applies a `WHERE eur IS NOT NULL` filter; other notebooks use the full 82,413 which includes zero-price rows._

### H1 — Reserved List Premium ✅ CONFIRMED

- RL cards: 901 | Non-RL sample: 5,000
- Median effect: **+2.562 log-units**
- 94% HDI: **[+2.390, +2.707]** — entirely outside ROPE
- P(effect > 0): **100%**
- EUR equivalent: **€12× premium** on a €1 base card

The largest confirmed price signal in the dataset by a wide margin.

---

### H2 — Legendary Premium ✅ CONFIRMED (reversed — legendary is cheaper)

ROPE_leg: (−0.05, +0.05). Tested per rarity.

| Rarity | Median effect | 94% HDI | P(>0) | % HDI outside ROPE |
|---|---|---|---|---|
| common | −0.175 | [−0.328, +0.006] | 2.6% | 93.2% |
| uncommon | −0.108 | [−0.149, −0.065] | 0.0% | **99.3%** |
| rare | −0.174 | [−0.219, −0.132] | 0.0% | **100.0%** |
| mythic | −0.058 | [−0.109, −0.003] | 2.1% | 61.8% — marginal |

**Legendary cards are cheaper than non-legendary within every rarity.** Effect is practically significant at uncommon and rare; marginal at mythic.

**Model implication:** `is_legendary` coefficient should be **NEGATIVE** in any model that does not control for reprint count. In BA-02 (which includes `log_print`), the sign flips positive — the legendary effect is absorbed into the reprint channel.

---

### H3 — Bayes Factor: Rarity Matters ✅ CONFIRMED

- Rare vs uncommon median effect: **+0.576 log-units (1.78× price ratio)**
- 94% HDI: **[+0.541, +0.611]**
- P(effect > 0): **100%**
- BF10: **> 1000 (decisive)** — Savage-Dickey posterior density at 0 is machine-epsilon

---

### H4 — Tier 1 vs Tier 3 Price Separation

| Tier | n | Posterior mu (median EUR) | 94% HDI |
|---|---|---|---|
| Tier 1 (<€100) | 81,458 | €0.79 | [€0.74, €0.84] |
| Tier 3 (>€1,000) | 139 | €2,770 | [€2,417, €3,132] |

sigma_obs is similar across tiers in log-space (Tier 1: 0.795, Tier 3: 0.825). In EUR space, Tier 3 HDI spans ~€715 vs Tier 1's ~€0.10 — the absolute uncertainty is enormous at the top end.

---

### H5 — Foil Premium ✅ CONFIRMED (smaller than CDA's cross-sectional estimate)

- Cards with paired eur + eur_foil: **47,383**
- Raw median per-card foil difference: +0.189 log-units (1.21×)
- Posterior median (global): **+0.362 log-units (1.44×)**
- 94% HDI: [+0.349, +0.375] — 100% outside ROPE
- P(effect > 0): 100%

> Note: CDA NB05 estimated median foil premium at 2.306× (cross-sectional). The Bayesian per-card paired estimate (1.44×) is smaller — CDA compared cross-sectional medians (mixing different card price levels), while BA-04 computes the difference within the same card. The 1.44× figure is the more rigorous estimate for modelling (see B-07).

Per-rarity posterior:

| Rarity | Median effect | Premium | 94% HDI |
|---|---|---|---|
| common | +0.285 | **1.33×** | [+0.271, +0.299] |
| uncommon | +0.333 | **1.40×** | [+0.319, +0.348] |
| rare | +0.452 | **1.57×** | [+0.433, +0.471] |
| mythic | +0.378 | **1.46×** | [+0.364, +0.394] |

Rare has the highest foil premium — not a strict rarity gradient (rare > mythic).

---

### H6 — Full-Art Premium ✅ CONFIRMED at all rarities

ROPE_fa: (−0.05, +0.05).

| Rarity | n_full_art | Median effect | Premium | 94% HDI | P(>0) | % outside ROPE |
|---|---|---|---|---|---|---|
| common | 419 | +0.119 | **1.13×** | [+0.071, +0.170] | 100% | 99.4% |
| uncommon | 298 | +0.168 | **1.18×** | [+0.093, +0.251] | 100% | 100% |
| rare | 2,222 | +0.888 | **2.43×** | [+0.839, +0.948] | 100% | 100% |
| mythic | 1,330 | +0.810 | **2.25×** | [+0.750, +0.870] | 100% | 100% |

The full-art premium is small at common/uncommon (~1.1–1.2×) and large at rare/mythic (~2.3–2.4×). The premium at common is statistically confirmed and practically non-negligible — consistent with CDA NB05 H4's unexpected common confirmation.

---

## Retest Schedule

| Test | Notebook | Min data | Earliest date |
|---|---|---|---|
| BA-03 Local Level Model (t+7 forecasts) | BA-03 | ≥30 snapshots | ~2026-07-03 |
| beta_log_print ESS — re-run with 4 chains | BA-02 | — | ✅ done (ESS now 1,787) |
| Update beta_is_legendary prior direction | BA-01 | — | now (direction known from BA-04 H2) |
| Full 82,413-card BA-02 run (not demo sample) | BA-02 | — | manual run only (~30 min MCMC) |

---

## TODOs & Blockers

### Actionable Now

- [x] **Add missing covariates to BA-02 model** — added `edhrec_saltiness` (standardised), `set_type` (ordinal-encoded, standardised), `log_top8` (`log1p(top8_appearances_30d)` aggregated across formats). Results: `beta_saltiness`=+0.222 (credible, HDI=[0.208, 0.236]); `beta_log_top8`=+0.091 (credible, HDI=[0.075, 0.108]); `beta_set_type`=+0.025 (HDI crosses zero → no credible effect — mu_set hierarchy already captures set-level variation). **Confound finding:** `beta_log_print` collapsed to ~0 once saltiness is controlled — powerful cards are reprinted more AND saltier; print count had no independent price effect.
- [x] **Re-run BA-02 with 4 chains × 2,000 draws** — done; took 42s on 5k subsample. R-hat < 1.01 ✅, ESS > 400 ✅ (beta_log_print ESS now 1,787), 0 divergences ✅.
- [ ] **Re-run BA-02 on full 82,413-card dataset** — code ready (comment out DEMO_N lines); requires ~30 min manual run — too slow for automated notebook execution.
- [x] **Check beta_log_print ESS** — resolved by 4 chains × 2,000 draws + `target_accept=0.95`; ESS 380 → 1,787.

### Data-Gated

- [ ] **BA-03 Local Level Model**: re-run at ≥30 snapshots (~2026-07-03)
- [ ] **BA-03 evaluation vs Naive baseline**: compare MAE_bayes vs MAE_naive at ≥30 snapshots

---

## Decisions Log

| # | Decision | Justification |
|---|---|---|
| B-01 | Prior width: σ = 2× empirical std | Narrow (1×) fails to cover all observed mythic prices; Wide (3×) generates 15.1% implausible >€100 mythics |
| B-02 | Prior split at mid-date 2026-06-05 | Prevents double-dipping: prior elicitation data ≠ inference data |
| B-03 | Negative prices in prior predictive are acceptable | Normal on log1p; data likelihood anchors to valid range; truncation adds complexity for marginal gain |
| B-04 | is_reserved prior: Normal(+2.0, 0.8) | CDA-01 RL premium 20.3× at rare → 2.10 log-units; wide sigma allows posterior to update |
| B-05 | Hierarchical structure: card ← set ← rarity | Partial pooling gives unseen cards a reasonable prior from their set/rarity group rather than flat prior |
| B-06 | Production recommendation: 4 chains × 2,000 draws | Current 2-chain run is exploratory only; 4 chains required for robust R-hat per PyMC/Stan convention |
| B-07 | Foil premium: use per-card paired estimate (1.44×) not cross-sectional median (2.306×) | BA-04 H5 shows CDA's 2.306× conflates card-level price differences with deck-mix effects; 1.44× is the causal per-card effect |
| B-08 | is_legendary direction: conditional on log_print | Unadjusted (BA-04 H2): NEGATIVE. Controlled for reprint count (BA-02): POSITIVE. Both are correct for their respective model specifications |
| B-09 | BA-02 sigma_rarity = 0.595 (median) | Sets within the same rarity vary by ~0.6 log-units; new sets should be shrunk toward rarity mean with this uncertainty. Consistent with STAT S-03 (Levene W=2981, 4.6× rare/common variance ratio) and CDA NB01 H5 (heteroscedasticity confirmed by rarity). |
| B-10 | BA-03 fallback prior | Until BA-03 can be trained, use BA-02 hierarchical posterior as starting price estimate for all cards |
| B-11 | Drop `beta_set_type` from BA-02 in future revision | HDI=[−0.020, 0.069] crosses zero; mu_set hierarchy already captures set-level price variation — `set_type` adds no independent signal |
| B-12 | `beta_log_print` collapses to ~0 when `saltiness` included | Confound: powerful/popular cards are both reprinted more and saltier; previously `log_print` was a proxy for card quality. Feature should be reconsidered or dropped in the next model revision. |
| B-13 | `edhrec_saltiness` is a strong independent predictor | beta_saltiness=+0.222, HDI=[0.208, 0.236], ESS=13,688 — salty cards command a persistent price premium independent of rarity, set, and print count |
| B-14 | `log_top8` (tournament appearances) has credible positive price effect | beta_log_top8=+0.091, HDI=[0.075, 0.108] — tournament demand signal is independently informative even controlling for set/rarity hierarchy |
