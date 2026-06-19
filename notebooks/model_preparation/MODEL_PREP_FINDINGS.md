# Model Preparation — Consolidated Findings Report

_Covers notebooks 01–04 in `notebooks/model_preparation/`. Last updated: 2026-06-08._

> Data context: **5 snapshots (2026-06-04 → 2026-06-08).** MP-01 (leakage audit) and MP-02 (feature selection) have complete cross-sectional results. MP-03 (validation strategy) and MP-04 (baseline models) are structurally complete but produce no numeric results — no t+7 forward-looking targets exist yet (need ≥8 snapshots).

---

## Table of Contents

1. [Leakage Audit & Target Definition (MP-01)](#mp-01--leakage-audit--target-definition)
2. [Feature Selection (MP-02)](#mp-02--feature-selection)
3. [Validation Strategy (MP-03)](#mp-03--validation-strategy)
4. [Baseline Models (MP-04)](#mp-04--baseline-models)
5. [Retest Schedule](#retest-schedule)
6. [TODOs & Blockers](#todos--blockers)
7. [Decisions Log](#decisions-log)

---

## MP-01 — Leakage Audit & Target Definition

### Leakage Classification

| Status | Count | Columns |
|---|---|---|
| LEAKAGE | 3 | `price_ath`, `price_atl`, `days_with_price` |
| RISKY | 4 | `price_change_7d_pct`, `price_change_7d_abs`, `price_change_30d_pct`, `price_change_30d_abs` |
| SAFE | 16 | (see table below) |

### Leakage Columns (drop unconditionally)

| Column | Reason |
|---|---|
| `price_ath` | PARTITION BY over entire card history — uses future rows |
| `price_atl` | PARTITION BY over entire card history — uses future rows |
| `days_with_price` | COUNT OVER entire card partition — knows future snapshot count |

### Risky Columns (target-dependent)

| Column | Risk | When safe |
|---|---|---|
| `price_change_7d_pct/abs` | Safe for t+7 target; leaks for t+14 target | Verify target horizon before use |
| `price_change_30d_pct/abs` | Safe for t+30 target; leaks for t+60 target | Verify target horizon before use |

### Safe Columns

| Column | Rationale |
|---|---|
| `eur`, `eur_foil`, `cardmarket_eur`, `cardmarket_eur_foil`, `tcgplayer_usd`, `tcgplayer_usd_foil` | Current prices at snapshot_date (t=0 input) |
| `price_7d_avg`, `price_30d_avg` | Rolling window looking backward only |
| `price_change_1d_abs`, `price_change_1d_pct` | LAG(1) — previous day only |
| `price_volatility_30d` | Rolling std over 30-day backward window |
| `foil_premium` | eur_foil / eur at same snapshot_date |
| `price_rank_global` | RANK WITHIN snapshot_date — cross-sectional, no future info. ⚠️ Near-circular with price level target (MI=5.33); less circular against log-return target — re-evaluate at ≥8 snapshots (see P-17) |
| `days_since_last_real_price` | Backward-looking only |
| `edhrec_rank` | Snapshotted daily in `silver_meta_history`; joined date-exact (`p.snapshot_date = m.snapshot_date`) — confirmed in `gold/features.py:build_price_features()`. P-18. |
| `is_price_spike` | `ABS((eur − LAG(eur,1)) / LAG(eur,1)) > 3.0` — backward LAG(1) only, no future prices. Leakage: none. ⚠️ Currently 0 for all rows due to UUID linkage bug (`uuid=NULL` spike records filtered by `WHERE p.uuid IS NOT NULL`). Add to feature sets once UUID bug is fixed. |
| `deck_pct_vintage`, `deck_pct_commander` | Scraped from MTGGoldfish format-staples pages. Stored per `snapshot_date` in `silver_format_staples_history` via `_snapshot()` (deduplicates on `(id, snapshot_date)`); promoted to `gold_format_staples`; joined date-exact in `gold_ml_dataset.py`: `cf.name = fs.card_name AND wc.snapshot_date = fs.snapshot_date AND fs.format = 'vintage'/'commander'`. P-22. ⚠️ Selection bias: MTGGoldfish only tracks expensive cards — non-null values pre-select for high price; treat as ordinal signal, not causal predictor. |

Saved to: `leakage_config.json`

### Target Definition

- **M1 target (primary):** `log1p(EUR[t+7]) − log1p(EUR[t])` — 7-day log-return
- **M2 target (secondary):** `log1p(EUR[t+30]) − log1p(EUR[t])` — 30-day log-return
- Join method: date-exact (`INTERVAL '7 DAY'` / `'30 DAY'`), **not** row-shift LAG — LAG silently shifts the window if a day is skipped
- SQL implementation: `LOG(p7.eur + 0.01) - LOG(p.eur + 0.01)`

> **Clarification vs EDA D-01:** EDA D-01 states "`log1p(eur)` is the project-wide price transformation standard" — this describes how raw EUR is represented, not what the model predicts. The ML prediction target is the log-return above (difference of two log1p-transformed prices). STAT S-05 formally suspends levels-vs-returns pending ADF/KPSS (~2026-07-03), but the return formulation is used here as the working assumption consistent with provisional I(1). All EDA/CDA feature rankings were computed against price **levels** — re-validate against returns once t+7 data exists (≥8 snapshots).

### Current Training Availability

| Target | Rows available | Status |
|---|---|---|
| t+7 (M1) | **0** | Need ≥8 snapshots (~2026-06-12) |
| t+30 (M2) | **0** | Need ≥31 snapshots (~2026-07-04) |

### Binary Spike Target (deferred)

Positive rate threshold rules (to apply once t+7 data exists):
- Positive rate < 5% → weighted loss or oversampling required
- Positive rate < 1% → threshold may be impractical for training

---

## MP-02 — Feature Selection

_24 candidate features. Dataset: 82,413 cards (latest snapshot). MI proxy target: log1p(EUR) price level — note this is not the final log-return target; re-run at ≥8 snapshots._

### NULL Rates

| Feature | NULL % | Imputation |
|---|---|---|
| `foil_premium` | **42.21%** (47,627 cards have foil prices) | fill 1.0 |
| `edhrec_saltiness` | **0.50%** | fill 0 (no controversy) |
| all boolean features | — | fill False |
| `mana_value` | — | median per rarity |

Note: `has_mtgjson_data = 1` for 100% of cards in the latest snapshot — the binary is zero-variance and excluded automatically.

### VIF Analysis — Exclusions

Features with VIF > 10 excluded due to multicollinearity:

| Feature | VIF | Action |
|---|---|---|
| `price_30d_avg` | ∞ | EXCLUDE — near-identical to spot price (5-snapshot history; re-evaluate at ≥30 snapshots) |
| `price_7d_avg` | ∞ | EXCLUDE — same as above |
| `is_commander_legal` | 156.3 | EXCLUDE |
| `is_legacy_legal` | 148.6 | EXCLUDE |
| `format_count` | 35.4 | EXCLUDE |
| `finish_count` | 12.4 | EXCLUDE |

Note: `is_modern_legal` (VIF=7.36), `color_identity_count` (VIF=7.90), `color_count` (VIF=7.51) are all below the VIF threshold of 10 — they are excluded by MI, not VIF (see MI table below). ⚠️ `color_identity_count` exclusion is disputed: CDA C-13 pre-specified ε²>0.01 as the inclusion criterion; CDA NB05 confirmed ε²=0.0199 which passes this threshold. The VIF (7.9 < 10) is also acceptable. This conflict must be resolved before finalising the feature set.

### Mutual Information Results (full ranking)

_MI computed against log1p(EUR) price **level**. Will shift when re-run against log-return target at ≥8 snapshots — re-run is mandatory before finalising feature set._

| Feature | MI | VIF | Decision | Notes |
|---|---|---|---|---|
| `price_30d_avg` | 5.356 | ∞ | EXCLUDE | VIF |
| `price_7d_avg` | 5.356 | ∞ | EXCLUDE | VIF |
| `price_rank_global` | 5.329 | 5.55 | **INCLUDE** ⚠️ | Near-circular with price level — see warning below |
| `foil_premium` | 0.896 | 1.25 | **INCLUDE** | |
| `edhrec_saltiness` | 0.158 | 2.54 | **INCLUDE** | |
| `format_count` | 0.144 | 35.4 | EXCLUDE | VIF |
| `print_count` | 0.123 | 2.09 | **INCLUDE** | Apply **log1p transform** at training time (EDA D-08) |
| `variation_count` | 0.066 | 1.34 | **INCLUDE** (caution) | |
| `is_full_art` | 0.056 | 1.24 | **INCLUDE** | CDA confirmed all rarities |
| `is_modern_legal` | 0.045 | 7.36 | EXCLUDE | MI threshold; collinear with format_count |
| `mana_value` | 0.043 | 4.47 | EXCLUDE ⚠️ | MI threshold — **conflicts with EDA D-12** (see disputes below) |
| `finish_count` | 0.043 | 12.4 | EXCLUDE | VIF |
| `color_identity_count` | 0.034 | 7.90 | EXCLUDE ⚠️ | MI threshold — **conflicts with CDA C-13** (ε²=0.0199 > 0.01) |
| `color_count` | 0.031 | 7.51 | EXCLUDE | MI threshold |
| `is_reserved` | 0.026 | 1.10 | EXCLUDE ⚠️ | MI threshold — **algorithmically wrong; must be force-added** |
| `is_legacy_legal` | 0.023 | 148.6 | EXCLUDE | VIF |
| `is_commander_legal` | 0.022 | 156.3 | EXCLUDE | VIF |
| `is_reprint` | 0.021 | 2.57 | EXCLUDE | MI |
| `is_promo` | 0.020 | 1.16 | EXCLUDE | MI |
| `is_standard_legal` | 0.013 | 2.98 | EXCLUDE | MI |
| `has_etched_finish` | 0.008 | 1.04 | EXCLUDE | MI |
| `is_textless` | 0.007 | 1.04 | EXCLUDE | MI |

**Features not in the VIF/MI analysis at all (dropped before or never added):**
- `rarity_ord` — the single strongest cross-sectional feature (CDA ε²=0.396). Absent from analysis; must be in ML feature set explicitly for tree models.
- `set_type` — 3rd strongest feature (CDA ε²=0.113). Absent entirely; must be one-hot encoded and added.
- `edhrec_rank` — 2nd strongest by EDA MI (0.299). Intentionally excluded from this analysis pending MP-01 P-02 leakage resolution — snapshot-at-t=0 join safety unverified. Once cleared, add with `log(edhrec_rank)` transform to both tier feature sets.
- `in_tournament` — EDA D-14 explicitly decided to include (6.9× premium). Absent.
- `top8_appearances_30d` — CDA C-08 decided to include (4.5× gradient). Absent.

> ⚠️ **`is_reserved` exclusion is wrong.** MI = 0.026 is suppressed by 1.1% class imbalance. BA-04 H1: +2.562 log-units, P(>0)=100%, HDI entirely outside ROPE. BA-02 beta_reserved=2.245 (ESS=2,146). Force-add to both tier feature sets regardless of MI.

> ⚠️ **`price_rank_global` circularity warning.** MI=5.33 against a price-level proxy is expected — rank within snapshot is a near-monotonic transformation of price. For a log-return target this is less circular (current rank predicts future change). But for any level-prediction model it is circular. Re-evaluate MI against the return target before retaining this feature.

### Disputes with Prior Notebook Decisions

| Feature | Prior decision | MP-02 result | Conflict |
|---|---|---|---|
| `is_reserved` | EDA D-11 + CDA C-03: KEEP | MI=0.026 → EXCLUDED | ✅ Force-added (P-07) — MI suppressed by 1.1% class imbalance |
| `mana_value` | EDA D-12: INCLUDE (partial Spearman −0.259) | MI=0.043 → EXCLUDED | ✅ Included in Tier 1 (P-19) — EDA partial Spearman is more reliable |
| `color_identity_count` | CDA C-13: INCLUDE (ε²=0.0199 > 0.01) | MI=0.034 → EXCLUDED | ✅ Included in Tier 1 (P-20) — CDA pre-specified criterion passed |
| `set_type` | CDA C-07: INCLUDE (ε²=0.113, one-hot) | Absent from analysis | ✅ Added (P-08) |
| `rarity_ord` | EDA/CDA: strongest feature | Absent from analysis | ✅ Added (P-08) |
| `in_tournament` | EDA D-14: ADD | Absent from analysis | ✅ Added to Tier 1 (P-09) |
| `top8_appearances_30d` | CDA C-08: ADD (sparse, impute 0) | Absent from analysis | ✅ Added to Tier 1 (P-09) |
| `print_count` | EDA D-08 + CDA C-06: use **log1p** | Listed as raw `print_count` | ✅ Transform annotated (P-10) |
| `edhrec_rank` | EDA D-02: INCLUDE (log transform) | Absent from analysis | ✅ SAFE confirmed in code (P-18) — added to both tiers |
| `primary_type` | CDA C-15: low-priority candidate | Absent from analysis | ✅ Included as low-priority one-hot (P-21) |

### Final Feature Sets (corrected)

_Updated to apply all prior decisions. Saved to `feature_sets.json`._

| Tier | Features |
|---|---|
| **Tier 1 (<€100)** | `rarity_ord`, `set_type`, `edhrec_rank`², `edhrec_saltiness`, `foil_premium`, `is_full_art`, `is_reserved`, `in_tournament`, `top8_appearances_30d`, `price_rank_global`, `print_count`¹, `variation_count`, `mana_value`³, `color_identity_count`⁴, `primary_type`⁵, `deck_pct_vintage`⁶, `deck_pct_commander`⁷ |
| **Tier 2 (€100–€1,000)** | Same as Tier 1 + rarity interaction terms (STAT S-03: 4.6× variance ratio; Levene W=2981). Train separate model instance — do not pool with Tier 1. |
| **Tier 3 (>€1,000)** | `rarity_ord`, `set_type`, `edhrec_rank`², `foil_premium`, `is_full_art`, `is_reserved`, `price_rank_global`, `print_count`¹, `variation_count`, `mana_value`³, `deck_pct_vintage`⁶ |

¹ Apply `log1p(print_count)` at training time.  
² Apply `log(edhrec_rank)` after sentinel fill (NULL → MAX+1 = 31,059).  
³ Cap at 16; impute median within rarity for NULL (Scryfall-only cards).  
⁴ Bucket as 0 / 1 / 2 / 3+. Tier 1 only — Tier 3 is dominated by colorless/artifacts.  
⁵ One-hot. Low-priority feature (ε²=0.0306 < 0.05 threshold). Tier 1 only.  
⁶ Fill 0 for non-vintage-tracked cards. Tier 1 + Tier 3 (r=+0.491; 4th strongest EDA signal).  
⁷ Fill 0 for non-commander-tracked cards. Tier 1 only — r=−0.232 (inverted; ambiguous for Tier 3).

### Notes on Transforms and Encoding

| Feature | Required transform | Reason |
|---|---|---|
| `print_count` | `log1p(print_count)` | EDA D-08; nonlinear relationship; raw value has sign-flip artifact |
| `set_type` | One-hot (merge tiny categories into "other") | CDA C-07; 15 set types; spellbook=24 + arsenal=17 → "other" |
| `edhrec_rank` | Sentinel fill (NULL → 31,059) then `log(edhrec_rank)` | EDA D-02; LOWESS shows steep decline ranks 1–500; log linearises. Sentinel preserves monotonicity (obscurity = high rank number = low demand). |
| `top8_appearances_30d` | Fill 0 for non-tournament cards | CDA C-08; 0.5% coverage sparse feature |
| `foil_premium` | None (already a ratio) | Keep rows where foil_premium < 1.0 — real signal (EDA D-10) |
| `rarity_ord` | Ordinal [0,1,2,3] | CDA C-01; common=0, uncommon=1, rare=2, mythic=3, special=2 |
| `mana_value` | Cap at 16; impute median per rarity | EDA D-09; Gleemax (1,000,000) is corrupted data; NULL = Scryfall-only card |
| `color_identity_count` | Bucket: 0 / 1 / 2 / 3+ | CDA C-13; non-monotone gradient — colorless highest (Power Nine effect) |
| `primary_type` | One-hot (8 types) | CDA C-15; Planeswalker is the most differentiating type (€1.94 median) |
| `deck_pct_vintage` | Fill 0 | Sparse (~0.5% coverage); 0 = not in any vintage list |
| `deck_pct_commander` | Fill 0 | Sparse (~0.5% coverage); 0 = not in any commander list |

---

## MP-03 — Validation Strategy

_5 snapshots, 82,413 cards/snapshot, 412,065 total rows._

### Walk-Forward Cross-Validation Design

| Parameter | Value |
|---|---|
| Strategy | Walk-forward expanding window |
| min_train_days | 30 |
| val_window | 7 days |
| step | 7 days |
| Folds generated now | **0** |
| Folds possible at | ≥37 snapshots (~2026-07-11) |

Temporal split is mandatory — random splits on time-series data introduce target leakage regardless of distribution stability.

### Hold-Out Test Set

- Status: **DEFERRED**
- Design: last 14 days of history (covers 2 full t+7 prediction horizons)
- Protocol: evaluated **once only**, after all modelling decisions are final
- Available at: ≥51 snapshots (~**2026-07-25**)
- Until then: the entire history is used for development only

### Minimum Data Requirements Per Card

| Minimum history | n cards meeting threshold | Feature use |
|---|---|---|
| 1 day | 82,413 (100%) | any static feature |
| 7 days | **0 (0%)** | `price_7d_avg` |
| 14 days | **0 (0%)** | M1 training rows (t+7 target) |
| 30 days | **0 (0%)** | `price_30d_avg` |
| 60 days | **0 (0%)** | M2 training rows (t+30 target) |
| 90 days | **0 (0%)** | full feature set |

### Statistical Power By Tier (cross-sectional proxy, all 5 snapshots)

| Tier | n rows | Power (d=0.5, α=0.05) |
|---|---|---|
| Tier 1 (<€100) | 408,540 | 100% |
| Tier 2 (€100–€1,000) | 2,830 | 100% |
| Tier 3 (>€1,000) | 695 | 100% |

> Note: per-CV-fold training windows will have fewer rows once slicing is applied. Power at fold level should be verified at ~2026-07-11.

Saved to: `validation_config.json` (status=DEFERRED)

---

## MP-04 — Baseline Models

_Status: **ALL BASELINES DEFERRED** — 0 rows with t+7 target (need ≥8 snapshots)._

### Baseline Suite

Four models defined and ready for execution:

| Model | Description | Signal it captures |
|---|---|---|
| **Naive** | Predict 0 log-return (price unchanged) | MA7d < Naive → mean reversion dominates |
| **MA7d** | Predict 7-day moving average return | MA7d > Naive → momentum dominates |
| **AR(1)** | Autoregression on lag-1 log-return | AR(1) β > 0.1 → include lag-1 as feature |
| **Ridge** | Ridge regression on confirmed feature set | Ridge > MA7d → domain features add signal beyond price history |

### Metric Definitions

- `MAE_logr` = mean(|y_true − y_pred|) on log-return scale
- `RMSE_logr` = sqrt(mean((y_true − y_pred)²)) on log-return scale
- `MAE_EUR` = mean(|EUR_actual − EUR_pred|) in EUR space

The **Naive MAE on first run** is the official benchmark that XGBoost/LightGBM must beat.

### AR(1) Decision Rules (for when data is available)

| AR(1) beta | Interpretation |
|---|---|
| > 0.1 | Positive lag-1 autocorrelation → include lag-1 price-change as a model feature |
| < −0.1 | Negative autocorrelation → mean-reverting signal; consider mean-reversion features |
| −0.1 to +0.1 | No useful autocorrelation signal |

**Re-run after: ~2026-06-12** (≥8 snapshots → first t+7 target rows)

---

## Retest Schedule

| Task | Min data | Earliest date |
|---|---|---|
| t+7 target rows available; re-run MP-01/04 | ≥8 snapshots | ~2026-06-12 |
| Re-run MI against log-return target | ≥8 snapshots | ~2026-06-12 |
| Re-run VIF for `price_7d_avg` / `price_30d_avg` | ≥30 snapshots | ~2026-07-03 |
| Binary spike target analysis | ≥14 snapshots | ~2026-06-18 |
| First CV folds executable | ≥37 snapshots | ~2026-07-11 |
| Hold-out test set definable | ≥51 snapshots | ~2026-07-25 |
| M2 (t+30) target rows available | ≥31 snapshots | ~2026-07-04 |
| Full AR(1) / Ljung-Box signal | ≥50 snapshots | ~2026-07-23 |

---

## TODOs & Blockers

### Re-run When Data Accumulates

- [ ] **Re-run MP-01 baseline metrics** at ≥8 snapshots (~2026-06-12) — first t+7 target rows
- [ ] **Re-run MI in MP-02 against log-return target** at ≥8 snapshots — feature ranking against price levels may not transfer to returns
- [ ] **Re-run MP-04 all baselines** at ≥8 snapshots — record Naive MAE as the official benchmark
- [ ] **Re-run VIF for `price_7d_avg` / `price_30d_avg`** at ≥30 snapshots — ∞ VIF is an artifact of 5 identical snapshots
- [ ] **Run first CV folds and log per-fold metrics** at ≥37 snapshots (~2026-07-11)
- [ ] **Define and freeze hold-out test set** at ≥51 snapshots (~2026-07-25) — evaluate once only

---

## Decisions Log

| # | Decision | Justification |
|---|---|---|
| P-01 | Leakage: drop `price_ath`, `price_atl`, `days_with_price` | PARTITION BY without ORDER BY ROWS BETWEEN → future data included by construction |
| P-02 | `edhrec_rank` classified as RISKY | Rank stored in historical rows may reflect state as of scrape time, not snapshot time |
| P-03 | ML target: log-return (not log-level) | Log-return model is stationary-safe; avoids predicting diverging price levels; aligns with Stat 02 provisional I(1) finding |
| P-04 | Target join: date-exact (`INTERVAL '7 DAY'`) not row LAG | LAG silently mis-aligns if pipeline skips a day; date join is always correct |
| P-05 | VIF threshold: 10 | Standard threshold; captures is_commander_legal (156), is_legacy_legal (149), format_count (35) |
| P-06 | `price_7d_avg` / `price_30d_avg` excluded despite high MI | VIF = ∞ at 5 snapshots (identical values); re-evaluate at ≥30 snapshots |
| P-07 | `is_reserved` force-added despite MI exclusion | MI = 0.026 suppressed by 1.1% class imbalance; BA-04 +2.562 log-units P>0=100%; BA-02 beta=2.25, ESS=2,146 |
| P-08 | `rarity_ord` and `set_type` force-added to feature sets | Absent from VIF/MI analysis; rarity is the single strongest cross-sectional feature (CDA ε²=0.396); set_type 3rd strongest (ε²=0.113); tree models need explicit inputs |
| P-09 | `in_tournament` + `top8_appearances_30d` added to Tier 1 | EDA D-14 and CDA C-08 explicitly decided to include; fell through the gap to MP-02 |
| P-10 | `print_count` stored as raw column; log1p applied at training time | EDA D-08 / CDA C-06; transform documented in feature_sets.json transforms block |
| P-11 | Walk-forward CV: min_train=30, val=7, step=7 | No data leakage; rolling windows match prediction horizon |
| P-12 | Hold-out test set: 14 days, evaluated once | 2 full t+7 horizons; single evaluation prevents over-fitting to the test set |
| P-13 | Naive MAE = official XGBoost/LightGBM benchmark | Any ML model that cannot beat a "predict no change" baseline adds no value |
| P-14 | Tier 3 uses Bayesian/Cardmarket lookup, not ML baselines | 695 Tier 3 rows across 5 snapshots; insufficient for gradient boosting |
| P-15 | MI rankings against price levels are provisional | All EDA/CDA MI and Spearman computed on price levels not returns; re-run at ≥8 snapshots against log-return target before finalising |
| P-16 | Loss function: MAE or Huber, NOT MSE | Pareto α=1.303 < 2 → infinite variance → MSE gradient unbounded; confirmed by STAT S-01. Applies to all tiers. |
| P-17 | `price_rank_global` provisionally included in both tier feature sets | Highest MI (5.33) but near-circular with price level target. Safe from temporal leakage (cross-sectional rank at t=0). Re-evaluate MI against log-return target at ≥8 snapshots before final decision. |
| P-18 | `edhrec_rank` reclassified RISKY → SAFE; added to both tiers | Code verified: `silver_meta_history` stores rank per `snapshot_date`; `gold/features.py:build_price_features()` joins date-exact (`p.snapshot_date = m.snapshot_date`). Transform: `log(edhrec_rank)` after sentinel fill. |
| P-19 | `mana_value` included in Tier 1 | EDA partial Spearman −0.259 within rarity (genuine signal after rarity control); MI=0.043 suppressed by rarity confound. Cap at 16; impute median per rarity. |
| P-20 | `color_identity_count` included in Tier 1 | CDA C-13 pre-specified ε²>0.01 threshold passed (ε²=0.0199); VIF=7.9 < 10. Non-monotone gradient — bucket as 0/1/2/3+. Tier 1 only (Tier 3 dominated by colorless artifacts). |
| P-21 | `primary_type` included as low-priority one-hot in Tier 1 | CDA ε²=0.0306 below 0.05 threshold but statistically significant; Planeswalker clearly distinct (€1.94 median). Tier 1 only. |
| P-22 | `deck_pct_vintage` + `deck_pct_commander` added to feature sets | Leakage SAFE — `silver_format_staples_history` stores value per `snapshot_date`; `gold_ml_dataset.py` joins date-exact on `(card_name, snapshot_date, format)`. `deck_pct_vintage` r=+0.491 (4th strongest EDA signal) → Tier 1 + Tier 3. `deck_pct_commander` r=−0.232 (inverted; ambiguous for expensive cards) → Tier 1 only. Also fixed `gold_ml_dataset.py` join bug: was `cf.oracle_id = fs.id`, now `cf.name = fs.card_name`. |
| P-23 | Formal multicollinearity assessment skipped | Tree-based models (GBM/XGBoost) are collinearity-immune. VIF in MP-02 already excluded the worst cases (`is_commander_legal` VIF=156, `is_legacy_legal` VIF=149, `format_count` VIF=35). `rarity_ord`/`set_type_ord` have distinct semantics — not problematic collinearity. Revisit only if a linear model tier is added. |
| P-24 | `has_mtgjson_data` not added as a model feature | Zero-variance: gold pipeline gates all joins on `uuid IS NOT NULL`, so every row in `gold_ml_dataset` has MTGJson data. Feature = constant 1 across all training rows. Already noted in MP-02 NULL Rates section. |
| P-25 | `is_power_restricted` (Commander=False AND Modern=False) not added | Signal already captured by `is_reserved` + `rarity_ord`. Group median €1.33 is only marginally above bulk; Reserved List median is €12. Adding would reintroduce collinearity with legality flags excluded by VIF. 2,143 cards insufficient for its own model segment. |
