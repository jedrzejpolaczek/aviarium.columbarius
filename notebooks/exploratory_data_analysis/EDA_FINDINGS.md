# EDA Findings — Consolidated Report

_Covers notebooks 01–04 in `notebooks/exploratory_data_analysis/`. Last updated: 2026-06-08._

---

## Table of Contents

1. [Data Structure & Quality (EDA-01)](#eda-01--data-structure--quality)
2. [Variable Distributions (EDA-02)](#eda-02--variable-distributions)
3. [Price Time Series (EDA-03)](#eda-03--price-time-series)
4. [Feature Relationships (EDA-04)](#eda-04--feature-relationships)
5. [TODOs & Blockers](#todos--blockers)
6. [Decisions Log](#decisions-log)

---

## EDA-01 — Data Structure & Quality

### Table Sizes

| Layer | Notable counts |
|---|---|
| Bronze | ~530k Scryfall cards, ~110k MTGJson cards, ~9.7M MTGJson prices |
| Silver | ~521k `silver_cards`, ~2.8M `silver_prices_history`, ~430k `silver_language_prices_history` |
| Gold | ~98k `gold_card_features`, ~3.3M `gold_price_features`, ~430k `gold_language_premiums` |

Row reduction Bronze→Silver and Silver→Gold is intentional and expected.  
`silver_cards` gained column `canonical_uuid` after rebuild — links non-English Scryfall cards to English MTGJson UUID via `(set_code, collector_number)`. Digital exclusives have `canonical_uuid = NULL`.

### NULL Regimes (Bronze)

Three distinct patterns — only the third is a data quality concern:

1. **Structural source gap (~81%):** `mana_value`, `set_code`, `supertypes` — MTGJson columns, absent for all Scryfall-only cards. **NOT errors.**
2. **Domain-structural (~20–99%):** `loyalty`, `power`, `toughness` — NULL means "not applicable". **NOT errors.**
3. **Genuine incompleteness (<10%):** `edhrec_rank` at ~6% — cards not listed on EDHREC.

EUR coverage: 16.8% of card-date rows in bronze; 83.7% of unique cards have ≥1 real EUR price in silver.

### Join Coverage

- Both sources: ~108k cards (~20.3%)
- Scryfall-only: ~422k cards (~79.7%) — language variants, digital-only, promos
- MTGJson-only: ~0

The 80/20 split is structural, not a pipeline issue.

### Language Premiums — Blocked

`lang_eur_premium` is **not usable**. Scryfall does not provide per-language Cardmarket prices — `prices.eur` for non-English cards equals the English canonical price in ~99.9% of rows. Only 11 Spanish cards showed any difference (likely noise). Japanese/Korean/Russian premiums show as 0.

> ⚠️ **Do NOT use `lang_eur_premium` as a model feature.** The infrastructure (canonical_uuid resolution, language tagging, gold join) is correct and kept intact for when a dedicated Cardmarket per-language scraper is added.

### Date Spine & History Window

- Pipeline age at time of EDA-03 run: **3 snapshots** (2026-06-04 → 2026-06-06). Current state as of 2026-06-08: **5 snapshots**. Daily cadence intact.
- 17% of cards (16,914) have no real EUR price in any snapshot.
- `price_7d_avg` is computable for all cards; `price_30d_avg` is **not viable yet**.
- t+30 and t+90 prediction targets are **infeasible** until ≥60 days of history exists.

### Oracle ID / Split Card Check

- Oracle ID name conflicts: **0** at time of analysis (previously 11 split cards were affected).

---

## EDA-02 — Variable Distributions

### EUR Price Distribution

- Raw EUR skewness: **62.00** — extreme.
- Mean (€11.55) >> Median (€0.27). 74.5% of cards cost <€1; 94.2% cost <€10.
- p99 = €68.80; max = €30,975 (Reserved List / Power Nine).

**Target transformation: `log1p(eur)` is the project-wide standard.** Inverse: `expm1()`. Use consistently in all notebooks and pipeline code.

### Numerical Feature Decisions

| Feature | Finding | Decision |
|---|---|---|
| `mana_value` | Gleemax at 1,000,000 (2 rows) | Cap at natural max (16) in pipeline |
| `edhrec_rank` | ~4.9% NULL (4,016 cards) | Sentinel = MAX+1 (31,059); preserves monotonicity |
| `print_count` | p50=4, max=848 (Basic Lands) | Use `log1p(print_count)` — capping destroys signal |
| `format_count` | Bounded 0–21 naturally | No transformation needed |
| `edhrec_saltiness` | ~0.5% NULL | Impute 0 (no entry = no controversy) |

### Rarity

Clear price gradient: common < uncommon < rare < mythic.  
Ordinal encoding: `{common: 0, uncommon: 1, rare: 2, mythic: 3, special: 2}`.  
Special (250 cards) maps to 2 — too small for its own category.

### Reserved List

Median €12 vs €0.26 for others — **~46× premium**.  
`is_reserved` is the single strongest binary feature.

### Foil Premium

Median foil premium by rarity: common 3.11×, uncommon 2.80×, rare 1.85×, mythic 1.45×, special 9.04×.  
Premium decreases as rarity rises. 3.3% of cards have `foil_premium < 1.0` (foil cheaper than non-foil) — pre-8th Edition sets. **This is a real market signal; do not remove these rows.**

> ⚠️ These are **cross-sectional ratios of group medians** (`eur_foil / eur` averaged per rarity) — not causal per-card effects. CDA NB05 H3 gives the overall cross-sectional median as 2.306×. BA-04 H5 gives the preferred per-card paired estimate: **1.44× globally** (common 1.33×, uncommon 1.40×, rare 1.57×, mythic 1.46×). The per-card gradient reverses direction at rare (rare is highest, not lowest). Use the BA-04 figures for modelling (BAYESIAN B-07).

### Print Count

Monotonic inverse relationship: `print_count=1` → highest median price; `print_count≥10` → near-bulk (~€0.10–0.20).

### EUR vs USD Market Divergence

Implied EUR/USD from card prices is **0.76 vs real rate ~0.92** — USD prices systematically ~18% higher due to independent Cardmarket (EU) / TCGPlayer (US) liquidity pools. Do not use USD as a EUR proxy directly.

### Price Tiers

| Tier | Count | % | Median EUR | Modeling approach |
|---|---|---|---|---|
| <€100 | 81,711 | 99.15% | €0.26 | Gradient boosting ML model |
| €100–€1,000 | 563 | 0.68% | €328 | ML model + guardrail floor |
| >€1,000 | 139 | 0.17% | €2,411 | Direct Cardmarket lookup |

Tier 3 has only 417 training rows — insufficient for ML. Direct lookup is the correct approach.

---

## EDA-03 — Price Time Series

> ⚠️ **This notebook was run with only 3 snapshots (2026-06-04 → 2026-06-06). Nearly all analyses are uninformative. The conclusions below describe expected behavior — not confirmed findings. Current pipeline state: 5 snapshots (2026-06-04 → 2026-06-08) — still below all retest thresholds.**

### Current Status of Time-Series Features

| Feature | Status | Decision |
|---|---|---|
| `price_7d_avg` / `price_30d_avg` | Identical to spot price | Exclude from training until ≥30 snapshots |
| `price_change_1d/7d/30d_pct` | 100% zero (uuid-linked) / NULL | Exclude from training until ≥30 snapshots |
| `price_volatility_30d` | 0.0 for ALL cards | Exclude from training until ≥30 snapshots |
| `is_price_spike` | 100% in `uuid=NULL` records | **Blocked** — UUID linkage fix required first |
| `LAG(7d)` row-based correctness | Untestable at 3 days | Re-verify at ≥8 snapshots |

### is_price_spike — UUID Linkage Bug

All 88 spike records have `uuid=NULL` and cannot be joined to card features. These are real market events for well-established cards (median `days_with_price` ≈ 30 years). Example magnitudes: +362%, +220%, +178%.

> ⚠️ **`is_price_spike` is blocked pending UUID fix. Do not use as a model feature until resolved.** Note: at 5 snapshots (2026-06-08) `is_price_spike = 0` for all 412,065 rows in `gold_price_features` — the 88 spike records are from an earlier pipeline state and are not currently propagating to the gold layer (STAT NB06).

### Expected Behavior Once Data Accumulates

- Change distribution: leptokurtic (narrow peak + fat tails). Most days: 0% change. Rare events: ±200–1000%.
- Loss function: **MAE or Huber, NOT MSE** — **finalised** by STAT NB01 (STAT S-01: Pareto α=1.303 < 2 → infinite variance → MSE gradient unbounded). No longer deferred.
- Tier 3 (>€1k) will show significantly higher volatility than Tier 1.

### LAG Feature Risk

Gold computes `LAG(7 rows)` not `LAG(7 calendar days)`. If the pipeline skips a day, row-based LAG silently shifts the window. This is the most critical pipeline correctness check for time-series features.

---

## EDA-04 — Feature Relationships

### Feature Ranking (24 features, by MI)

| Feature | Pearson r | Spearman r | MI | Assessment |
|---|---|---|---|---|
| `foil_premium` | −0.086 | −0.334 | **0.897** | STRONG — dominant MI, highly nonlinear |
| `edhrec_rank` | −0.314 | −0.540 | 0.299 | STRONG — all 3 metrics |
| `rarity_ord` | +0.415 | +0.619 | 0.280 | STRONG — top Spearman |
| `edhrec_saltiness` | +0.373 | +0.293 | 0.161 | STRONG |
| `set_type_ord` | +0.231 | +0.313 | 0.146 | STRONG — confirmed CDA NB02 H3 (ε²=0.113); masters < expansion (reprinting depresses prices) |
| `format_count` | −0.151 | −0.201 | 0.147 | WEAK partial (rarity proxy after control) |
| `print_count` | −0.029 | +0.309 | 0.124 | STRONG — use `log1p`; nonlinear |
| `variation_count` | +0.051 | +0.232 | 0.066 | MODERATE |
| `is_full_art` | +0.265 | +0.253 | 0.053 | MODERATE |
| `is_legendary` | +0.148 | +0.230 | 0.041 | MODERATE — ⚠️ **direction REVERSED**: CDA NB01 H2 confirms legendary cards are cheaper within every rarity; standalone feature **REMOVED** (CDA C-02) |
| `is_modern_legal` | −0.197 | −0.236 | 0.040 | MODERATE — negative direction |
| `mana_value` | −0.009 | −0.006 | 0.040 | MODERATE — hidden via partial Spearman |
| `finish_count` | −0.064 | −0.116 | 0.040 | MODERATE — hidden via partial Spearman |
| `color_count` | −0.054 | −0.050 | 0.036 | MODERATE — hidden via partial Spearman |
| `color_identity_count` | −0.042 | −0.014 | 0.028 | MODERATE — hidden via partial Spearman |
| `is_legacy_legal` | −0.156 | −0.155 | 0.026 | WEAK, collinear |
| `is_reserved` | +0.286 | +0.143 | 0.026 | **KEEP** — domain: 190× premium at uncommon |
| `is_commander_legal` | −0.132 | −0.146 | 0.021 | KEEP — negative direction (see below) |
| `is_promo` | +0.064 | +0.118 | 0.020 | MARGINAL |
| `is_reprint` | +0.063 | +0.101 | 0.020 | MARGINAL |
| `is_colorless` | +0.088 | +0.103 | 0.018 | DROP candidate |
| `is_standard_legal` | −0.045 | −0.053 | 0.014 | DROP |
| `has_etched_finish` | +0.039 | +0.040 | 0.009 | DROP |
| `is_textless` | +0.029 | +0.036 | 0.008 | DROP |

### Key Nonlinear / Counterintuitive Findings

**`foil_premium` dominant MI (0.897 vs 0.299 for #2):**  
Pearson r = −0.086 is completely misleading. MI reveals a rich, non-monotonic dependency — bulk cards have high foil premium (collector rarity); competitive staples have low foil premium (both formats are liquid). Tree models will capture this naturally; linear models will miss most of the signal.

**`mana_value` hidden signal:**  
Raw Spearman ≈ 0. Partial Spearman (controlling for rarity) = **−0.259**. Signal fully suppressed by confound: higher-CMC cards are disproportionately rare/mythic. Within the same rarity, lower-CMC = more expensive (efficiency premium). Include `mana_value`.

**`print_count` sign flip:**  
Raw Pearson −0.029 but Spearman +0.309. Causality is reversed — powerful cards get reprinted more, pushing `print_count` up. Log1p transformation and tree-based models handle this correctly.

**Format legality — inverted direction:**  
Cards NOT commander-legal have a **higher** median price (€1.33 vs €0.26) because the ban list includes Black Lotus, Power Nine, etc. Being banned = proxy for "too powerful for any fair format." Same inversion for `format_count` (0–4 formats → highest median; 20+ formats → lowest median). Models will learn the correct direction; do not manually flip.

### Reserved List × Rarity Interaction

| Rarity | n Reserved | Median Reserved | Median Other | Premium |
|---|---|---|---|---|
| uncommon | 16 | €34.21 | €0.18 | **190×** |
| rare | 885 | €12.41 | €0.61 | **20×** |

Reserved List cards exist only at uncommon and rare. The uncommon premium is the most extreme price distortion in the dataset.

### EDHREC Rank Shape

Spearman: −0.540 raw, −0.434 after controlling for rarity (80% survives — genuinely independent signal).  
LOWESS shape: steep decline ranks #1–#500; plateau #500–#5,000; near-flat #5,000+. Use `log(edhrec_rank)` to linearize the steep top region.

### Tournament & Format Staple Signals

- `in_tournament` binary: median €1.80 vs €0.26 non-tournament (6.9× premium). Spearman r = 0.279. Include.
- `deck_pct` (MTGGoldfish): Vintage r = +0.491 (strong), Commander r = −0.232 (inverted), Modern/Legacy r ≈ 0. Use as **format-specific features** (`deck_pct_vintage`, `deck_pct_commander`, etc.), not a global signal.

### Multicollinearity Candidates (to address in model_preparation/02)

- `rarity_ord` ↔ `set_type_ord`
- `is_commander_legal` ↔ `format_count`
- `is_modern_legal` ↔ `is_legacy_legal` ↔ `format_count`

---

## TODOs & Blockers

### Re-run When Data Accumulates

- [ ] Re-run entire EDA-03 (time series) at ≥30 snapshots (~2026-07-03)
- [ ] Re-run forward-fill analysis (EDA-01 §6) at ≥30 snapshots
- [ ] Verify `LAG(7d)` row-based vs date-based correctness at ≥8 snapshots
- [ ] Re-run ban/unban price impact analysis (EDA-04 §7) after the first ban event is captured (Wizards announces quarterly: March, June, September, December)
- [ ] Evaluate t+30 / t+90 prediction targets at ≥60 days of history (earliest ~2026-08-03, counting from pipeline start 2026-06-04)

---

## Decisions Log

| # | Decision | Rationale |
|---|---|---|
| D-01 | `log1p(eur)` is the project-wide price **transformation standard** | Skewness 62; mean >> median; `expm1()` is exact inverse. The ML **prediction target** is the log-return `log1p(EUR[t+7]) − log1p(EUR[t])` (defined in MP-01) — D-01 describes how raw EUR is represented, not what the model predicts. |
| D-02 | `edhrec_rank` NULL → sentinel = MAX+1 (31,059) | Preserves monotonic ordering; missingness = obscurity |
| D-03 | `edhrec_saltiness` NULL → impute 0 | No EDHREC entry = no controversy |
| D-04 | Rarity ordinal: common=0, uncommon=1, rare=2, mythic=3, special=2 | Special (250 cards) too small for own category |
| D-05 | Tier 3 (>€1k) → direct Cardmarket lookup | 417 training rows — insufficient for ML |
| D-06 | Drop `cardmarket_eur` | Duplicate of `eur`; adds only multicollinearity |
| D-07 | Exclude `days_since_last_real_price` from current model | Zero variance in latest snapshot |
| D-08 | Do not cap `print_count`; use `log1p` | Capping destroys reprinting-signal gradient |
| D-09 | Cap `mana_value` at 16 | Gleemax (1,000,000) is 2 rows; would dominate any normalisation |
| D-10 | Keep `foil_premium < 1.0` rows | Real market signal (pre-8th Edition) — not noise |
| D-11 | Keep `is_reserved` regardless of low MI (0.026) | Class imbalance (~1.1%; 901/82,413) suppresses MI; 190× uncommon premium (€34.21/€0.18) is domain truth; confirmed BA-04 H1 (+2.562 log-units, P>0=100%) |
| D-12 | Include `mana_value` in model | Partial Spearman −0.259 after controlling for rarity — genuine signal |
| D-13 | `deck_pct` → per-format features only | Global signal r=0.127 is misleading (Vintage +0.49, Commander −0.23) |
| D-14 | Add `in_tournament` binary feature | 6.9× price premium; Spearman r=0.279 |
| D-15 | Defer all time-series features | Pipeline too young (5 snapshots as of 2026-06-08); revisit at ≥30 snapshots (~2026-07-03) |
| D-16 | `lang_eur_premium` deferred | Scryfall price feed does not differentiate by language |
