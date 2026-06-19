# Statistical Properties — Consolidated Findings Report

_Covers notebooks 01–07 in `notebooks/statistical_properties/`. Last updated: 2026-06-08._

> ⚠️ **Data context:** The pipeline has **5 daily snapshots** (2026-06-04 → 2026-06-08). Notebooks 02–07 remain almost entirely deferred — prices are constant across all 5 days (zero variance in market median), so time-series tests are still uninformative. The only notebook with actionable, confirmed findings is **NB01** (cross-sectional distribution testing).

---

## Table of Contents

1. [Distribution Testing (NB01)](#nb01--distribution-testing)
2. [Stationarity (NB02)](#nb02--stationarity)
3. [Seasonal Decomposition (NB03)](#nb03--seasonal-decomposition)
4. [Cointegration (NB04)](#nb04--cointegration)
5. [Per-Card Stationarity (NB05)](#nb05--per-card-stationarity)
6. [Structural Breaks (NB06)](#nb06--structural-breaks)
7. [Price Comovement (NB07)](#nb07--price-comovement)
8. [Retest Schedule](#retest-schedule)
9. [TODOs & Blockers](#todos--blockers)
10. [Decisions Log](#decisions-log)

---

## NB01 — Distribution Testing

_Cross-sectional, n=82,413 cards. Results are confirmed and actionable._

### Transformation Comparison

| Transformation | D (KS) | Verdict |
|---|---|---|
| Raw EUR | 0.4824 | Highly non-normal |
| log1p(EUR) | 0.2575 | Best available; right tail remains |
| log10(EUR+0.01) | 0.1164 | Slightly better D, but undefined at 0 |

**log1p(EUR) confirmed as project standard.** log10 looks better on D but is undefined at €0 and loses sub-cent precision. Inverse: `expm1()`.

Anderson-Darling: all rarities reject H₀ (p≈0 at n>25k — expected). Anderson-Darling is more sensitive in tails than KS; both confirm non-normality.

### Heteroscedasticity (Levene's Test)

W=2,981.17, p≈0 — **heteroscedasticity confirmed.**

| Rarity | std(log1p EUR) |
|---|---|
| common | 0.513 |
| uncommon | 0.681 |
| rare | **1.103** (highest) |
| mythic | 1.046 |

Variance ratio rare/common: **4.6×** — exceeds the threshold where a pooled OLS model produces correct inference. A single global model with homogeneous residuals is structurally inappropriate.

### Power-Law Tail (Pareto)

Zipf plot is linear for the top 5% of cards (R²=0.9776).  
Estimated Pareto alpha: **α = 1.303**

This means **1 < α < 2**: finite mean, but **infinite variance**. Consequences:
- MSE is dominated by the top 0.1% of prices (Power Nine). The gradient of MSE is theoretically unbounded for this distribution — gradient descent on MSE is structurally unstable.
- The sample variance grows unboundedly with more data — it does not converge.
- Log-normal gives a biased-low point estimate for Tier 3 (>€1k). Correct prior: Pareto(α=1.3).

### Confirmed Decisions from NB01

| Decision | Justification |
|---|---|
| **Target: log1p(EUR)** | D drops from 0.48 to 0.26; `expm1()` is exact inverse |
| **Loss: MAE or Huber, NOT MSE** | Pareto α=1.303 < 2 → infinite variance → unbounded MSE gradient |
| **Tiered models required** | 4.6× variance ratio exceeds WLS threshold |
| **Tier 3 prior: Pareto(α=1.3)** | Log-normal underestimates the tail; Bayesian/Pareto model needed |

Tier model design:
- **Tier 1 (<€100):** OLS or Huber on log1p(EUR) — bulk residuals quasi-normal
- **Tier 2 (€100–€1k):** OLS/Huber with rarity interaction terms
- **Tier 3 (>€1k):** Pareto-based or Bayesian model; log-normal point estimate is systematically biased low

---

## NB02 — Stationarity

_Market aggregate series. 5 snapshots → all tests still skipped (prices constant, zero variance across all 5 days)._

### Tests and Minimum Data Requirements

| Test | Min n required | Status |
|---|---|---|
| ACF / PACF | 10 | Skipped (n=5) |
| ADF (Augmented Dickey-Fuller) | 20 | Skipped (n=5) |
| KPSS | 10 | Skipped (n=5) |
| Ljung-Box autocorrelation | 50 | Skipped (n=5) |

### Expected Results (theory, unconfirmed)

- Log-price levels → **non-stationary I(1)** (unit root; random walk)
- Log-returns (first differences) → **stationary I(0)**
- If ACF(1) > 0.3 after ≥50 snapshots → include lag-1 price as model feature

### Critical Blocking Dependency

EDA-04 ranks features by correlation with **log1p(EUR) price levels**.  
If ADF/KPSS confirms I(1) → the correct ML target becomes **log-returns** (`log1p(EUR[t+7]) − log1p(EUR[t])`), not levels.  
Static attributes (rarity, edhrec_rank) may explain price levels well but explain little of short-term changes.

> ⚠️ **EDA-04's feature ranking is NOT directly transferable to a log-return model.** EDA-04 must be re-run with return-based correlation/MI before feature selection is finalised if the target changes to returns. **MODEL_PREP_FINDINGS.md P-03 has provisionally adopted log-returns as the working target** pending this confirmation — see STAT S-05.

---

## NB03 — Seasonal Decomposition

_5 snapshots, market median EUR = 0.2310 on all 5 days (zero variance). All sections deferred. Mann-Kendall technically runnable (≥4 snapshots) but returns NaN while prices are constant._

### Minimum Data Requirements

| Analysis | Min n | Expected date |
|---|---|---|
| Mann-Kendall trend test | 4 | now (5 snapshots; returns NaN while prices constant) |
| STL decomposition (period=7, weekly proxy) | 15 | ~2026-06-18 |
| STL (period=90, quarterly MTG set cycle) | 180 | ~2026-12-01 |

### Design Decisions (embedded in notebook, for future runs)

- **STL period=90** is the target for proper quarterly MTG seasonality (sets release every ~3 months).
- **robust=True** — to down-weight ban/unban price spikes in the decomposition.
- Separate STL per tier (Tier 1 and Tier 3 may have different seasonal patterns).
- If STL residuals show autocorrelation → add ARIMA or SARIMA on residuals.

---

## NB04 — Cointegration

_4 price series: EUR Scryfall, Cardmarket EUR, USD TCGPlayer, USD Scryfall. All tests need ≥20 observations → all skipped._

### Current Indicative Spread (n=5, not reliable)

Spread log(EUR_Scryfall) − log(Cardmarket_EUR):
- Mean = −0.0079 (~0.8% Cardmarket higher)
- Std = 0.0000 (constant over 5 days — no variation)

> ⚠️ OLS at n=3 produced β₁≈0.05 — an artifact of near-constant series. Do not interpret.

### Expected Results (from EDA-02 cross-section)

- EUR ~ Cardmarket EUR: cointegration **likely** (Spearman r=0.948, median ratio=1.000)
- EUR ~ USD: cointegration **likely** (r=0.948, with FX offset)

### Model Implications (all deferred)

- Spread (EUR − Cardmarket_EUR) as a model feature: deferred until cointegration confirmed
- Error Correction Model: deferred (requires I(1) confirmation + Engle-Granger test)

---

## NB05 — Per-Card Stationarity

_Extends NB02 to individual card level. All tests deferred — cards with ≥20 observations: 0._

### Expected Results by Tier

| Tier | Expected I(1) rate | Implication |
|---|---|---|
| Tier 1 (<€100) | ~80% | Bulk cards follow random walk; spike on reprint/ban |
| Tier 2 (€100–€1k) | ~85% | Staples tend non-stationary |
| Tier 3 (>€1k) | ~90% | Reserved List/Power Nine strongly non-stationary |

### Model Target Decision (pending)

| I(1) prevalence | Model target decision |
|---|---|
| >70% I(1) | Log-return as primary target (confirms provisional NB02 finding) |
| 20–70% I(1) | Mixed model: level for I(0) cards, return for I(1) cards |
| Tier 3 likely | Pure I(1) target regardless of global result |

---

## NB06 — Structural Breaks

_Bai-Perron and Chow tests. All deferred (need ≥30 observations per card series)._

### Current Status (5 snapshots)

`is_price_spike = 0` for all 412,065 rows across the 5-snapshot history. No price spikes detected yet — expected given the short window and constant prices.

### Methods Planned

1. **Bai-Perron** (multiple breaks via `ruptures.Binseg`, BIC to select k, max k=5; `n_bkps = min(5, len(series)//10)`)
2. **Chow test** on known event dates (Mann-Whitney U on 30-day before/after windows)
3. **CUSUM** (parameter instability in rolling regression)

### Additional Blocker

The Chow test requires a `gold_events` table (ban/unban event calendar). **This table does not exist in the pipeline yet.**

### Known Event Anchors for Future Chow Test

| Date | Event |
|---|---|
| 2019-11-18 | Oko, Thief of Crowns ban (Standard, Pioneer, Modern) |
| 2020-06-01 | Companion mechanic nerf (rule errata) |
| 2020-09-28 | Uro, Titan of Nature's Wrath ban |

### Model Implications (pending confirmation)

If structural breaks are confirmed:
- Add `is_price_spike` as a model feature (already computed in `gold_price_features`)
- Add `days_since_last_ban` / `days_since_last_unban` features (requires `gold_events` table — pipeline TODO)
- Consider separate training windows: pre-break vs post-break
- Tier 3 is especially sensitive — ban events dominate price behavior at that level

---

## NB07 — Price Comovement

_Tests whether cards within categories (Dual Lands, Power Nine, Commander staples) move together. Rolling/time-series tests deferred; cross-sectional level analysis is possible now._

### Methods Planned

1. **Cross-sectional Spearman correlation** within categories — possible with 3 snapshots (measures shared price levels, not dynamics)
2. **Rolling pairwise correlation of log-returns** — needs ≥30 snapshots
3. **Hierarchical clustering** by price movement pattern — needs ≥30 snapshots

### Dual Land Identification Note

Must be matched **by card name, not `set_code`** — all 10 original duals were reprinted across Revised, 4th, 5th Edition. Explicit list: Tundra, Underground Sea, Bayou, Savannah, Taiga, Scrubland, Volcanic Island, Badlands, Plateau, Tropical Island. Correlation computed on log-returns (`.diff()`), not levels.

### Expected Results

| Category | Expected mean ρ | Interpretation |
|---|---|---|
| Dual Lands | 0.7–0.9 | Move as a collector class |
| Power Nine | 0.6–0.8 | Shared speculation/collector market |
| Commander staples | 0.3–0.5 | Moderate; diverse demand drivers |

Threshold for confirming category-level signal: **mean ρ > 0.6**.

### Model Implications (pending)

- If Dual Lands ρ > 0.7: add `is_dual_land`, `is_power_nine` as group-level categorical features — expected to outperform individual card attributes for Tier 3 predictions
- If ρ < 0.4: each card is independent → rely on card-level features only

---

## Retest Schedule

| Test | Notebook | Min snapshots | Earliest date |
|---|---|---|---|
| ACF / PACF | NB02 | 10 | ~2026-06-13 |
| Mann-Kendall trend test | NB03 | 4 | now (5 snapshots; returns NaN while prices constant) |
| STL decomposition (weekly proxy) | NB03 | 15 | ~2026-06-18 |
| Per-card ADF / KPSS | NB05 | 20 per card | ~2026-06-23 |
| Engle-Granger cointegration | NB04 | 20 | ~2026-06-23 |
| ADF on EUR-Cardmarket spread | NB04 | 20 | ~2026-06-23 |
| Market ADF / KPSS | NB02 | 30 | ~2026-07-03 |
| Per-tier I(0)/I(1) breakdown | NB05 | 30 | ~2026-07-03 |
| Bai-Perron structural breaks | NB06 | 30 | ~2026-07-03 |
| Rolling price comovement | NB07 | 30 | ~2026-07-03 |
| Spread half-life of mean reversion | NB04 | 30 | ~2026-07-03 |
| Ljung-Box autocorrelation | NB02 | 50 | ~2026-07-23 |
| STL decomposition (quarterly) | NB03 | 180 | ~2026-12-01 |

---

## TODOs & Blockers

### Re-run When Data Accumulates (see schedule above)

- [ ] Re-run NB02 stationarity tests at ≥20 snapshots (ACF/PACF first at ≥10)
- [ ] Re-run NB03 seasonal decomposition — STL period=7 at ≥15 snapshots, period=90 at ≥180
- [ ] Re-run NB04 cointegration at ≥20 snapshots
- [ ] Re-run NB05 per-card stationarity at ≥20 snapshots per card
- [ ] Re-run NB06 structural breaks at ≥30 snapshots
- [ ] Re-run NB07 rolling comovement at ≥30 snapshots

### Model Design (carry to model_preparation)

- [ ] **Determine ML target:** once ADF/KPSS runs (~2026-07-03), confirm whether target is log-price levels or log-returns. If I(1) → re-run EDA-04 feature ranking with return-based correlation/MI before feature selection. **Provisional working assumption: log-returns adopted** (MODEL_PREP P-03; STAT S-05).
- [ ] **Evaluate `is_dual_land` / `is_power_nine` group features** once NB07 rolling comovement is confirmed (ρ > 0.6 threshold)
- [ ] **Compute spread (EUR − Cardmarket_EUR)** as candidate feature once cointegration is confirmed

---

## Decisions Log

| # | Decision | Justification |
|---|---|---|
| S-01 | Loss function: **MAE or Huber, NOT MSE** | Pareto α=1.303 < 2 → infinite variance → MSE gradient unbounded; confirmed both empirically and theoretically |
| S-02 | Target transformation: **log1p(EUR) confirmed** | D reduces 0.48→0.26; `expm1()` exact inverse; defined at €0 |
| S-03 | **Segmented/tiered models required** | Levene's W=2,981, p≈0; 4.6× variance ratio rare/common — pooled OLS structurally wrong |
| S-04 | Tier 3 prior: **Pareto(α=1.3)** | Zipf R²=0.977, estimated α=1.303; log-normal biased low for expensive tail |
| S-05 | ML target (levels vs returns): **suspended** | Requires I(0)/I(1) confirmation from ADF/KPSS; available ~2026-07-03. **Provisional working assumption:** MODEL_PREP_FINDINGS.md P-03 adopted log-returns pending this confirmation. |
| S-06 | EDA-04 feature ranking validity: **conditional** | Valid for levels model; must be re-run with return-based MI/correlation if target switches to log-returns |
| S-07 | STL period: **90 days** (quarterly set cycle) | MTG releases every ~3 months; period=7 is a proxy only for early-data phase |
| S-08 | STL robust=True | Protects decomposition from ban/unban spike contamination |
| S-09 | Cointegration spread feature: **deferred** | Plausible (r=0.948) but blocked on ≥20 snapshots for EG test |
