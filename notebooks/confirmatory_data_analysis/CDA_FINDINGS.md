# Confirmatory Data Analysis — Consolidated Findings Report

_Covers notebooks 01–07 in `notebooks/confirmatory_data_analysis/`. Last updated: 2026-06-08._

> All tests run at **α = 0.05** with Bonferroni correction within hypothesis families. All prices are log1p-transformed. Effect sizes: ε² for Kruskal-Wallis, rank-biserial r for Mann-Whitney U, ρ for Spearman. Dataset: 82,413 cards, **5 snapshots (2026-06-04 → 2026-06-08)**, unless noted.

---

## Table of Contents

1. [Rarity & Card Attributes (NB01)](#nb01--rarity--card-attributes)
2. [Format Legality & Demand (NB02)](#nb02--format-legality--demand)
3. [Tournament Signals (NB03)](#nb03--tournament-signals)
4. [Temporal Stability & Autocorrelation (NB04)](#nb04--temporal-stability--autocorrelation)
5. [Card Taxonomy (NB05)](#nb05--card-taxonomy)
6. [Seasonality (NB06)](#nb06--seasonality)
7. [Market Cointegration EUR/USD (NB07)](#nb07--market-cointegration-eurusd)
8. [Retest Schedule](#retest-schedule)
9. [TODOs & Blockers](#todos--blockers)
10. [Feature Engineering Summary](#feature-engineering-summary)
11. [Decisions Log](#decisions-log)

---

## NB01 — Rarity & Card Attributes

### H1 — Rarity Premium ✅ CONFIRMED

Kruskal-Wallis: **H = 32,565.4, ε² = 0.3963** (LARGE; threshold 0.14).  
Rarity explains ~40% of variance in ranked log-prices — the single strongest feature in the dataset.

EUR medians: common €0.12 < uncommon €0.18 < rare €0.65 < mythic €3.02 — monotonic gradient confirmed.

Post-hoc pairwise (Bonferroni k=6): all 6 pairs significant (p_bonf ≈ 0).  
Rank-biserial r: 0.27 (common vs uncommon) → 0.90 (common vs mythic).

**Decision:** Ordinal encoding `[common=0, uncommon=1, rare=2, mythic=3]`.

---

### H2 — Legendary Premium ❌ NOT CONFIRMED (reversed direction)

Mann-Whitney per rarity: rank-biserial r is **negative** in every tier.

| Rarity | r | p_bonf |
|---|---|---|
| common | −0.40 | 1.0 |
| uncommon | −0.06 | 1.0 |
| rare | −0.05 | 1.0 |
| mythic | −0.02 | 1.0 |

**Legendary cards are systematically CHEAPER than non-Legendary cards across all rarities.**

Explanation: Non-legendary mythics are Planeswalkers and powerful spells — typically more expensive than the mass of cheap Commander-fodder legends. The EDA-02 boxplots (`showfliers=False`) hid cheap Legendary bulk and created a false visual premium. This is exactly why confirmatory analysis is necessary.

> ⚠️ **`is_legendary` as a standalone feature is REMOVED from the feature list.** Only valid as an interaction with rarity, and even then with caution.

---

### H3 — Reserved List Premium ✅ CONFIRMED

| | ρ | p |
|---|---|---|
| Unconditional Spearman | 0.144 | ≈ 0 |
| Partial Spearman (controlling for rarity) | 0.109 | ≈ 10⁻²¹⁶ |

Per-rarity breakdown (RL cards exist only at uncommon and rare):

| Rarity | n_RL | Median RL | Median other | r | p_bonf |
|---|---|---|---|---|---|
| uncommon | 16 | €33.74 | €0.18 | 0.964 | <0.001 |
| rare | 885 | €12.41 | €0.61 | 0.679 | ≈ 0 |

Within the same rarity, Reserved List cards cost **20–90× more**. Global ε² is small (~1%) because only 901 cards are Reserved, but the signal for those cards is overwhelming.

**Decision:** `is_reserved` is a STRONG feature. Keep regardless of low global MI.

---

### H4 — Reprints Have Lower Price Volatility ⏭ SKIPPED

`price_volatility_30d` = 0.0 for all 82,413 cards. Pipeline has only 5 snapshots; 30-day rolling stddev of identical prices = 0. Retest at ≥30 snapshots (~2026-07-03).

---

### H5 — Heteroscedasticity Between Rarities ✅ CONFIRMED

Levene W = 2,981.2, p ≈ 0.

| Rarity | std(log1p EUR) |
|---|---|
| common | 0.51 |
| uncommon | 0.68 |
| rare | **1.10** |
| mythic | 1.05 |

Variance ratio rare/common: **4.6×**.

**Decision:** A single linear model with homogeneous residuals is structurally inappropriate. Tiered models per rarity OR weighted regression (WLS) required.

---

### NB01 Scorecard: 3 confirmed, 1 rejected (reversed), 1 skipped

---

## NB02 — Format Legality & Demand

### H1 — Commander-Legal Premium ❌ NOT CONFIRMED (reversed direction)

97.3% of all cards are Commander-legal. Mann-Whitney (legal > illegal) fails in every rarity (p_bonf = 1.0).

**Actual finding:** Commander-ILLEGAL cards are more expensive. The ban list includes Black Lotus, Moxes, Power Nine — the most valuable cards in the game. Being banned = proxy for "too powerful for any fair format."

**Decision:** `is_commander_legal = False` is a HIGH-PRICE SIGNAL. Expect a negative coefficient in the model (False = premium).

---

### H2 — format_count and Price ✅ CONFIRMED (reversed direction)

| | ρ |
|---|---|
| Unconditional Spearman | −0.202 |
| Partial Spearman (controlling for rarity) | −0.077, p = 3×10⁻¹⁰⁹ |

More legal formats → lower price after controlling for rarity. Old cards legal only in Legacy/Vintage (format_count = 3–4) include Reserved List staples. New cards legal in many formats (format_count ≥ 10) have larger supply and lower prices.

**Decision:** Include `format_count` as a feature with expected negative coefficient. Effect is counterintuitive — dominated by the RL/Power Nine confound. Model will learn the correct direction.

---

### H3 — Price Differences Between Set Types ✅ CONFIRMED

Kruskal-Wallis: **H = 9,295, ε² = 0.113** (LARGE).

Median EUR descending: masterpiece €2.21 > box €1.67 > memorabilia €1.40 > promo €0.78 > eternal €0.74 > starter €0.52 > commander €0.35 > core €0.26 > **masters €0.24** > expansion €0.18 > duel_deck €0.15

Notable: **masters < expansion** — reprint products demonstrably depress prices by increasing supply.

**Decision:** `set_type` one-hot encoding; masterpiece/box/memorabilia carry clear premium. Groups with <500 cards (spellbook=24, arsenal=17) → merge into "other".

---

### H4 — EDHREC Rank and Price ✅ CONFIRMED

| | ρ |
|---|---|
| Unconditional Spearman | −0.540 |
| Partial Spearman (controlling for rarity) | −0.434, p ≈ 0 |

EDHREC rank retains 80% of its signal after controlling for rarity — a genuinely independent feature explaining ~19% of log_eur variance beyond rarity alone.

**Decision:** Strong feature. NULL imputation: sentinel = MAX+1 (31,059).

---

### H5 — print_count (Supply Effect) ✅ CONFIRMED, paradoxical direction

| Condition | ρ |
|---|---|
| Spearman including RL | +0.309 |
| Spearman excluding RL | +0.321, p ≈ 0 |

**Supply paradox:** More reprints → higher price (up to ~15 printings).

Median EUR by print_count bucket (excl. RL): 1 print → €0.12, 2–3 → €0.20, 4–7 → €0.35, 8–15 → **€0.57**, 16–50 → €0.43, 50+ → €0.24.

Explanation: heavily reprinted cards (Lightning Bolt, Counterspell) are format staples with high demand. Demand effect dominates supply effect at low-to-moderate print counts. Only at 50+ printings (Basic Lands) do prices return to bulk level.

**Decision:** Non-linear relationship. Use `log1p(print_count)` or bucketed encoding.

---

### H6 — EDHREC Saltiness and Price ✅ CONFIRMED

| | ρ |
|---|---|
| Unconditional Spearman | +0.293 |
| Partial Spearman (controlling for rarity) | +0.220, p ≈ 0 |

99.2% of cards (81,753/82,413) have a saltiness value (range 0.0–3.060). The partial signal (ρ=+0.220) exceeds the pre-specified inclusion threshold of |ρ_partial| > 0.1 and is independent from `edhrec_rank`.

Median EUR by decile: D1 €0.160 → D9 €1.590 — near-monotone increase. Saltier cards are controversial format staples typically priced higher.

**Decision:** Include `edhrec_saltiness` as a feature. NULL imputation: 0 (no EDHREC entry = no controversy, as per EDA D-03).

---

### NB02 Scorecard: 5 confirmed (2 reversed direction), 1 rejected (reversed)

---

## NB03 — Tournament Signals

Dataset: 428 unique tournament cards, 5 formats, 5-snapshot price history. Only 638 tournament records in sample.

> ⚠️ **Selection bias:** The scraper samples the most-played (= most expensive) cards per format. The tournament vs non-tournament premium reflects scraper design, not causality.

---

### H1 — Tournament Cards vs Non-tournament Price Premium ✅ CONFIRMED (selection bias caveat)

Mann-Whitney: **r = +0.508, p ≈ 0**. Median: tournament €1.80 vs non-tournament €0.26 (6.9× premium).

**Within-tournament gradient (no selection bias):**  
Spearman(top8_appearances_30d, eur) = **+0.279, p < 0.001**.

| Appearances bucket | Median EUR |
|---|---|
| 0 (tracked but no top-8 in 90d) | €0.96 |
| 1–5 | €1.77 |
| 6–15 | €2.18 |
| 16+ | **€7.99** |

Jump from occasional → highly played: **~4.5×**. This gradient is free of selection bias and is genuine signal.

**Decision:** Include `top8_appearances_30d` as sparse feature (impute 0 for non-tournament cards). Only 428 of 82,413 cards (0.5%) are covered.

---

### H2 — Appearances vs Price Volatility ⛔ NOT TESTABLE

98.9% of tournament cards have `price_volatility_30d` = 0 (5-snapshot pipeline). Spearman ρ = −0.036 is meaningless (zero variance in the dependent variable). Retest at ≥30 snapshots (~2026-07-03).

---

### H3 — Price Movement Before/After First Tournament Appearance ⛔ NOT TESTABLE

0 eligible cards — all `last_top8_date` values are on or before the price history start date.

> ⚠️ **Leakage risk noted:** If pricing sources update within 24h of tournament results, the day-of snapshot already incorporates the outcome. Requires investigation of T−1, T=0, T+1 snapshot timing relative to tournament date. Retest at ≥60 days of price history (~2026-08-06).

---

### H4 — Granger Causality (Tournaments → Prices) ⛔ NOT TESTABLE

1 weekly observation per format; minimum 30 required. Expected lag if significant: 1–2 weeks. If significant only at lag > 4 → signal too slow to be a useful feature. Retest at ≥30 weekly snapshots (~2026-12-01).

---

### NB03 Scorecard: 1 confirmed (with caveat), 3 not testable

---

## NB04 — Temporal Stability & Autocorrelation

Dataset: 5 snapshots (2026-06-04 → 2026-06-08), market median EUR = 0.27 on all 5 days (zero variance). All log-returns = 0.0.

---

### H1 — Distribution Stability Over Time ⛔ NOT TESTABLE

Need ≥15 days per half-window. KS test methodology fully documented; will auto-execute.  
Threshold: D < 0.05 = practically stable; D > 0.10 = meaningful drift.

> **Regardless of outcome:** temporal train/test split is REQUIRED — training on future data introduces target leakage even with a stable distribution. Retest at ≥30 snapshots (~2026-07-03).

---

### H2 — Log-Return Autocorrelation (Ljung-Box) ⛔ NOT TESTABLE

n = 2 log-return observations; minimum 50 required. ACF(1) undefined at n=2.

**Critical rule embedded in notebook:** `max_lag = n // 5` (never test lags above this — spurious autocorrelation at long lags).

Expected result at ≥50 snapshots: if ACF(1) > 0.3 → include lag-1 price-change as model feature; if ACF(7) > 0.3 → include lag-7 feature. Retest at ≥50 snapshots (~2026-07-26).

---

### H3 — ACF(1) Differs Between Tier 1 and Tier 3 ⛔ NOT TESTABLE

All tiers have 2 return observations; minimum 5 required for even a preliminary estimate.

Expected: Tier 3 higher |ACF(1)| — speculative momentum in expensive cards; Tier 1 lower — more liquid, efficient bulk market. Retest at ≥15 snapshots (~2026-06-21).

---

### Blocking Dependency

> ⚠️ **All NB04 hypotheses test log-return properties. Their relevance depends on whether the ML target is log-returns (I(1)) or log-levels (I(0)) — a question that cannot be answered until Stat 02 ADF/KPSS runs at ≥30 snapshots (~2026-07-03).** If Stat 02 confirms I(0), these hypotheses are moot for feature engineering. **Provisional working assumption: log-returns adopted** (MODEL_PREP P-03).

---

### NB04 Scorecard: 0 confirmed, 3 not testable

---

## NB05 — Card Taxonomy

_Cross-sectional, n=82,413 cards, 5 snapshots. All Kruskal-Wallis/Mann-Whitney with ε² effect sizes._

### H1 — Color Identity Count vs Price ✅ CONFIRMED

Kruskal-Wallis: **H = 1,644.3, ε² = 0.0199** (small effect, exceeds 0.01 inclusion threshold).

| Group | n | Median EUR |
|---|---|---|
| Colorless (0) | 8,403 | €0.460 |
| Multi (3+) | 2,370 | €0.430 |
| Two-color (2) | 12,348 | €0.300 |
| Mono (1) | 59,292 | €0.240 |

Non-monotone as expected — Colorless ranks highest due to the Power Nine, artifacts, and Eldrazi. The ordering is not purely by color complexity.

**Decision:** Include `color_identity_count` as a bucketed feature (0 / 1 / 2 / 3+). ε² = 0.0199 exceeds the 0.01 threshold.

---

### H2 — Primary Card Type vs Price ✅ CONFIRMED (below pre-specified effect threshold)

Kruskal-Wallis: **H = 2,529.0, ε² = 0.0306** (statistically significant; p ≈ 0; but below the 0.05 inclusion threshold).

| Primary Type | Median EUR |
|---|---|
| Planeswalker | €1.940 |
| Artifact | €0.380 |
| Enchantment | €0.370 |
| Land | €0.300 |
| Sorcery | €0.280 |
| Kindred | €0.250 |
| Creature | €0.220 |
| Instant | €0.210 |

Planeswalker is the most expensive type by far. Land is NOT at the top — Basic Lands (€0.02) dominate the distribution and pull the median down. The pre-specified ordering (Land > Planeswalker) was wrong.

**Decision:** BORDERLINE — ε² = 0.0306 is below the 0.05 threshold. The effect is real but small. Consider including `primary_type` one-hot as a lower-priority feature; finalize in model_preparation/02.

---

### H3 — Foil Premium vs Non-Foil ✅ CONFIRMED

Wilcoxon signed-rank one-sided (eur_foil > eur): **p ≈ 0**. 47,627/82,413 cards (57.8%) have both EUR and EUR_foil prices.

Overall median foil premium: **2.306×** (93.1% of foil cards are more expensive than non-foil).

| Rarity | Foil premium (cross-sectional ratio) |
|---|---|
| common | 3.143× |
| uncommon | 2.812× |
| rare | 1.854× |
| mythic | 1.453× |

Cross-sectional ratios decrease as rarity rises. `foil_premium` is already a confirmed strong feature.

> ⚠️ **These are cross-sectional ratios of group medians, not causal per-card effects.** BA-04 H5 gives the per-card paired estimate: **1.44× globally** (common 1.33×, uncommon 1.40×, **rare 1.57× — highest**, mythic 1.46×). The per-card gradient is the **opposite** at rare: rare has the highest causal premium, not the lowest. The cross-sectional gradient (decreasing with rarity) is a composition artifact. Use BA-04 figures for modelling (BAYESIAN B-07). The 2.306× cross-sectional figure is retained here as a description of the dataset distribution.

---

### H4 — Full-Art Premium per Rarity ✅ CONFIRMED for ALL rarities (unexpected)

Mann-Whitney one-sided (full-art > standard-art) per rarity, Bonferroni k=4. All four rarities confirmed — including common, which was expected NOT to be confirmed.

| Rarity | Median full-art | Median standard | Premium | r | p_bonf |
|---|---|---|---|---|---|
| common | €0.280 | €0.120 | 2.3× | 0.537 | < 0.001 |
| uncommon | €0.360 | €0.180 | 2.0× | 0.357 | < 0.001 |
| rare | €4.710 | €0.580 | 8.1× | 0.569 | < 0.001 |
| mythic | €7.880 | €2.210 | 3.6× | 0.455 | < 0.001 |

Common full-art basics are collectible (foil full-art basics trade at €0.10–0.50, not €0.01). The original expectation (only rare/mythic significant) was wrong.

**Decision:** Include `is_full_art` as a feature. Confirmed premium in all rarities.

---

### NB05 Scorecard: 4/4 confirmed (H2 borderline on effect size)

---

## NB06 — Seasonality

Confirmatory extension of Stat 03. All tests deferred — only 5 snapshots of data.

### H1 — STL Seasonal Pattern ⏳ DEFERRED

Threshold: seasonal_fraction > 0.05 = CONFIRMED; < 0.01 = REJECTED.  
If CONFIRMED → add seasonal features (day-of-quarter, set-release-offset).  
If trend significant → model levels with trend term, not just returns.

### H2 — Tier 1 vs Tier 3 Seasonal Amplitude ⏳ DEFERRED

Hypothesis: new set releases depress bulk prices more than premium prices. Tier 1 seasonal fraction expected HIGHER than Tier 3 (Reserved List / Power Nine are not reprinted — event-driven, not release-driven).  
If CONFIRMED → use tier-specific seasonal features (bulk needs them; Tier 3 doesn't).

**Technical notes:** Uses SQL median (`PERCENTILE_CONT(0.5)`), not mean — Pareto distribution makes mean unreliable. STL `robust=True` to down-weight ban/unban spikes.

---

## NB07 — Market Cointegration EUR/USD

Confirmatory extension of Stat 04. All tests deferred — only 5 snapshots of data. Additionally blocked on I(1) confirmation from Stat 02.

### H1 — EUR Scryfall ~ Cardmarket EUR Cointegration ⏳ DEFERRED

Prior evidence: EDA-02 Spearman r = 0.948, median ratio = 1.000 — cointegration likely.  
If CONFIRMED → spread `log(EUR) − β·log(CM_EUR)` is stationary and mean-reverting → add as ECM feature.

### H2 — EUR ~ USD Cross-Currency Cointegration ⏳ DEFERRED

Expected β₁ ≈ 0.93 (EUR/USD FX rate). Spread represents exchange-rate-adjusted pricing divergence.  
If CONFIRMED → USD predictions cross-validate EUR; use FX-adjusted spread.  
If NOT CONFIRMED → markets are independent; no cross-market feature is useful.

**Notes:** Minimum n = 20 for Engle-Granger (below this, ADF on residuals is unreliable and spurious cointegration is almost guaranteed). Half-life of mean reversion requires stable ADF at ≥30 snapshots.

---

## Retest Schedule

| Hypothesis | Notebook | Min snapshots | Earliest date |
|---|---|---|---|
| H3 ACF tier differences | NB04 | 6 | ~2026-06-09 |
| H1/H2 seasonality STL (weekly) | NB06 | 15 | ~2026-06-18 |
| H1 EUR~Cardmarket cointegration | NB07 | 20 | ~2026-06-23 |
| H2 EUR~USD cointegration | NB07 | 20 | ~2026-06-23 |
| H4 reprint volatility | NB01 | 30 | ~2026-07-03 |
| H1 distribution stability (KS) | NB04 | 30 | ~2026-07-03 |
| Stat 02 blocker (I(0)/I(1)) | NB04 | 30 | ~2026-07-03 |
| Half-life of EUR spread | NB07 | 30 | ~2026-07-03 |
| H2 appearances vs volatility | NB03 | 30 | ~2026-07-03 |
| H2 Ljung-Box autocorrelation | NB04 | 50 | ~2026-07-23 |
| H3 before/after tournament | NB03 | 60 | ~2026-08-06 |
| H1/H2 seasonality STL (quarterly) | NB06 | 180 | ~2026-12-01 |
| H4 Granger causality | NB03 | 210 (30 weekly) | ~2026-12-01 |

---

## TODOs & Blockers

### Blocking Dependency

- [ ] **Resolve Stat 02 I(0)/I(1) question (~2026-07-03)** — determines whether ML target is log-returns or log-levels. If I(1): re-run EDA-04 with return-based MI/correlation before finalising feature list in model_preparation/02. **Provisional working assumption: log-returns adopted** (MODEL_PREP P-03; consistent with provisional I(1) expectation from STAT S-05).

### Data-Gated Re-runs (see schedule above)

- [ ] NB04 H3 tier ACF — 2026-06-21
- [ ] NB01 H4 reprint volatility, NB04 H1 distribution stability — 2026-07-03
- [ ] NB03 H2 appearances vs volatility — 2026-07-03
- [ ] NB06 STL weekly seasonality — 2026-06-19
- [ ] NB07 cointegration tests — 2026-06-24
- [ ] NB04 H2 Ljung-Box — 2026-07-26
- [ ] NB03 H3 before/after tournament (+ leakage investigation) — 2026-08-06
- [ ] NB03 H4 Granger causality — 2026-12-01
- [ ] NB06 STL quarterly seasonality — 2026-12-01

### Leakage Risk to Investigate

- [ ] **Tournament result leakage (NB03 H3):** Confirm whether pricing sources update within 24h of tournament results. Compare T−1, T=0, T+1 snapshots relative to tournament date before using tournament features in training. **Query implemented and runnable** — 3 tournament dates (2026-06-05/06/07) found within price window with 54–251 card overlap, but all changes are 0.0 (prices flat across all 5 snapshots, not yet meaningful). Re-run once `avg_change_day_of` or `avg_change_day_after` is non-zero.

---

## Feature Engineering Summary

### Confirmed Strong Features

| Feature | Evidence | Notes |
|---|---|---|
| `rarity_ord` | ε² = 0.396, Spearman r = 0.619 | Ordinal [0,1,2,3]; single strongest feature |
| `edhrec_rank` | Partial ρ = −0.434, MI = 0.299 | NULL → sentinel MAX+1; use log(rank) to linearise |
| `is_reserved` | Partial ρ = 0.109; 20–90× premium within rarity | Keep despite low global MI; 901 cards |
| `set_type` | ε² = 0.113 | One-hot; merge tiny categories into "other" |
| `top8_appearances_30d` | r = 0.279 within tournament set, 4.5× gradient | Sparse feature; impute 0 for non-tournament cards |
| `foil_premium` | MI = 0.897 (dominant) | Highly nonlinear; Pearson r misleadingly low. ⚠️ For modelling use BA-04 H5 per-card estimate (1.44× global; rare 1.57× highest) not cross-sectional ratios (2.306×) — see B-07 |
| `edhrec_saltiness` | Partial ρ = +0.220, MI = 0.161 | NULL → impute 0 (no controversy); confirmed independent from edhrec_rank |

### Confirmed but Counterintuitive Direction

| Feature | Raw signal | Correct interpretation |
|---|---|---|
| `is_commander_legal` | Legal cards are CHEAPER | False = proxy for Power Nine / banned power cards |
| `format_count` | More formats → lower price | RL/Power Nine confound; weak feature after controlling |
| `print_count` | More reprints → higher price (up to 15×) | Demand dominates supply; use log1p or buckets |

### Removed / Demoted

| Feature | Reason |
|---|---|
| `is_legendary` (standalone) | **REMOVED** — reverses direction across all rarities; EDA-02 visual was misleading. Only valid as `is_legendary × rarity` interaction |
| `lang_eur_premium` | Not usable (see EDA_FINDINGS.md) |
| `has_etched_finish`, `is_textless`, `is_standard_legal` | MI < 0.015; DROP candidates |

### Confirmed from NB05

| Feature | Decision | Evidence |
|---|---|---|
| `color_identity_count` (bucketed 0/1/2/3+) | **INCLUDE** | ε² = 0.0199 > 0.01 threshold; colorless highest (Power Nine) |
| `is_full_art` | **INCLUDE** | Confirmed premium in ALL rarities (2–8×); common was unexpected |
| `primary_type` one-hot | **INCLUDE** (low-priority) | ε² = 0.0306 < 0.05 threshold but statistically significant; Planeswalker clearly distinct (€1.94 median); included in Tier 1 as low-priority feature (MP-02 P-21) |

### Pending (data-gated)

| Feature | Gate |
|---|---|
| `price_volatility_30d` (reprints H4) | ≥30 snapshots |
| `price_change_1d_pct`, `price_change_7d_pct` | ≥50 snapshots + Stat 02 I(1) confirmation |
| EUR spread (ECM feature) | Cointegration confirmed, ≥20 snapshots |
| Seasonal features (day-of-quarter, set-release-offset) | STL confirmed, ≥15 snapshots |
| `is_dual_land`, `is_power_nine` group features | Rolling comovement ρ > 0.6, ≥30 snapshots |

---

## Decisions Log

| # | Decision | Justification |
|---|---|---|
| C-01 | `rarity_ord` ordinal [0,1,2,3] | ε² = 0.396, monotonic gradient confirmed; strongest single feature |
| C-02 | `is_legendary` standalone REMOVED | Reversed direction in all 4 rarities; EDA-02 visual was misleading due to showfliers=False |
| C-03 | `is_reserved` KEEP despite low MI | 901 cards; 20–90× premium within rarity; class imbalance suppresses MI score |
| C-04 | `is_commander_legal = False` is a high-price signal | Ban list = Power Nine proxy; negative coefficient expected |
| C-05 | `format_count` included with negative coefficient | Confirmed weak negative partial ρ; counterintuitive direction is real |
| C-06 | `print_count` → log1p or bucketed | Supply paradox (demand dominates at <50 prints); non-linear relationship |
| C-07 | `set_type` one-hot, merge tiny categories | ε² = 0.113; masterpiece/box/memorabilia carry measurable premium |
| C-08 | `top8_appearances_30d` sparse feature (impute 0) | 4.5× gradient within tournament set; 0.5% coverage acceptable for a sparse feature |
| C-09 | Tiered/segmented models required | Levene 4.6× variance ratio (confirmed H5 CDA-01 + Stat 01); pooled OLS structurally wrong |
| C-10 | Temporal train/test split REQUIRED | Target leakage risk regardless of distribution stability test outcome |
| C-11 | ML target (levels vs returns) SUSPENDED | Blocked on Stat 02 ADF/KPSS (~2026-07-03); provisional = log-returns if I(1) confirmed |
| C-12 | `edhrec_saltiness` INCLUDED | Partial ρ=+0.220 > 0.1 threshold; independent from edhrec_rank; NULL → impute 0 |
| C-13 | `color_identity_count` bucketed INCLUDED | ε²=0.0199 > 0.01 threshold; non-monotone (colorless > multi > two > mono) |
| C-14 | `is_full_art` INCLUDED | Premium confirmed in all rarities including common (full-art basics are collectible); common 2.3×, rare 8.1× |
| C-15 | `primary_type` included as low-priority one-hot (Tier 1) | ε²=0.0306 < 0.05 threshold but statistically significant; Planeswalker clearly distinct (€1.94 median vs €0.22 Creature); decided in MP-02 P-21 |
