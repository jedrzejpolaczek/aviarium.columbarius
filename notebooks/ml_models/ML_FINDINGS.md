# ML Models — Findings

Results from the `notebooks/ml_models/` notebooks. Every number below comes from the
notebook's saved output or from the MLflow run it names.

**Read this first:** T6 and T8 are a *single chronological split* at
`SNAPSHOT_DATE = 2026-07-02` (36 snapshots). The cross-validated result, in T7 (2026-09-30),
is less flattering: LightGBM does not beat the naive baseline.

---

## T5 — Feature engineering

**Notebook:** `01_feature_engineering.ipynb` · run at `SNAPSHOT_DATE = 2026-07-02`

- **Training matrix:** 93 386 rows × 39 columns. After dropping rows with a NULL
  target (cards with no Cardmarket EUR price on either end of the 7-day window),
  78 692 rows and the 17 features the pipeline uses remain.
- **Lag coverage:** `lag_7d` complete for 80 757 / 96 261 cards (83.9%), `lag_30d` for
  79 908 (83.0%). The missing ~16% are cards with no EUR price on the snapshot date;
  history length is no longer the constraint.
- **Target shape:** `log_return_7d` is sharply peaked at zero with a thin tail to about
  ±1 and rare outliers up to +5. About 84% of cards have an identical price after
  7 days. `momentum_7d` shows the same zero spike.
- **Implication:** aggregate error metrics are dominated by cards whose price did not
  move. Evaluations should also report the non-zero subset.

---

## T6 — Baselines vs LightGBM

**Notebook:** `02_baseline_lightgbm.ipynb` · run 2026-07-10 · single chronological split

MAE on the `log1p` return scale, per price tier:

| Tier | Test cards | Naive | MA7d | LightGBM |
|---|---:|---:|---:|---:|
| 1 (< €100) | 15 606 | 0.056128 | 0.056128 | **0.054001** |
| 2 (€100–1000) | 111 | **0.031356** | 0.031356 | 0.033488 |
| 3 (> €1000) | 22 | 0.053028 | 0.053028 | **0.052530** |

AR(1), overall only: 0.0569 — the weakest baseline.

- **LightGBM wins Tier 1, loses Tier 2 by ~7%, and ties Tier 3 within noise.** Seven-day
  price moves are close to a random walk. Tier 2 has the lowest naive error of any tier
  and the fewest training examples relative to the feature count.
- **Naive and MA7d are identical to six decimals** in every tier. MA7d predicts
  `rolling_mean_7d - log_eur`, and the price feed was effectively frozen until early
  July: comparing each snapshot with the one 7 days later gives a price-level error of
  exactly 0.00% for every snapshot up to 2026-06-29. The 7-day mean before the split date
  therefore equals the current price, and MA7d predicts a zero return, exactly like
  Naive. The comparison between them becomes meaningful only on post-July data.
- **An earlier run was degenerate.** At 32 snapshots (2026-06-22) the target was almost
  entirely zero and every model scored MAE ≈ 0. That was a price-feed artefact that
  disappeared by 36 snapshots, not a result.

---

## T7 — Walk-forward cross-validation

**Measured 2026-09-30** via `scripts/train_model.py` → `retrain()` → `walk_forward_cv`,
MLflow run `351ad6ef3b664b38885fa3965bd3a8a8`. 3 usable folds, the project's minimum.
MAE on the log1p return scale:

| Fold | Validation snapshot | Tier 1 LightGBM | Tier 1 naive | Tier 2 LightGBM | Tier 3 LightGBM |
|---:|---|---:|---:|---:|---:|
| 0 | 2026-07-08 | 0.025339 | 0.021252 | 0.012602 | 0.004750 |
| 1 | 2026-07-15 | 0.024447 | 0.024424 | 0.011353 | 0.010953 |
| 2 | 2026-09-23 | 0.025813 | 0.026280 | 0.011563 | 0.007121 |
| **Mean** | | **0.025200** | **0.023985** | 0.011839 (naive 0.010363) | 0.007608 (naive 0.006905) |

- **LightGBM does not beat naive under CV:** 5% worse on Tier 1, 14% on Tier 2 and 10% on Tier 3.
- **Fold 0 is the outlier.** Its training labels span the July price-feed switch. The only
  fold on clean data both ends (fold 2) has LightGBM 1.8% better on Tier 1, which three
  folds cannot establish.
- **Early-stopping leak, fixed the same day.** Before the fix, CV early-stopped on the
  validation fold. That leak alone showed Tier 1 at 0.02385, 0.6% better than naive.
  The fix is in `trainer.py`, with a regression test in `tests/ml/training/test_trainer.py`.
- **Card metadata leak, checked and immaterial.** `gold_card_features` holds current-state
  card metadata (e.g. `print_count` including later reprints). Rebuilding it as of each
  fold's date changed Tier 1 MAE by less than 0.1%.
- **Next re-measurements:** 6 folds (≈2026-10-21) and 14 folds (≈2026-12-16).
- The optional Prophet comparison has not been run.

---

## T8 — Optuna and SHAP

**Notebook:** `04_shap_optuna.ipynb` · run 2026-07-10 · MLflow run `tuned_lightgbm_nb04`
(`9c1ec7de65c7476aa75a199b7189d6f2`)

### Optuna

| Parameter | Best value | Search range |
|---|---:|---|
| `num_leaves` | 203 | 32–256 |
| `learning_rate` | 0.0308 | 0.01–0.3 (log) |
| `min_child_samples` | 37 | 20–200 |
| `subsample` | 0.7539 | 0.6–1.0 |
| `colsample_bytree` | 0.8 | fixed |
| `n_estimators` | 1000 | fixed (early stopping) |

- **Tuning gain is marginal:** validation MAE 0.053116 with default parameters versus
  0.053038 for the best of 20 trials — a 0.15% improvement. The model is close to what
  this feature set allows.
- The run logged training metrics only (`train_mae = 0.0500`, `train_mape = 85.23%`), so
  it cannot be compared with the per-tier test MAE in T6. The 85% MAPE is expected:
  the target is a log-return near zero, so the MAPE denominator is tiny even after the
  `MAPE_CLIP_MIN = 0.01` clip. MAE is the comparison metric.

### SHAP

- **Most important feature:** `edhrec_rank`, followed by `foil_premium`, `rarity_ord`,
  `lag_1d`, `lag_30d`, `print_count`, `format_count` and `edhrec_saltiness` (8th of ~20).
- **BA-02 not confirmed:** the Bayesian analysis suggested `edhrec_saltiness` would absorb
  the importance of `print_count`. In this model `print_count` ranks higher (6th) than
  `edhrec_saltiness`, and both keep a visible spread.
- **`is_reserved` has almost no effect** on predictions (near the bottom of the ranking).
- **Open question:** the waterfall section explained the first three rows of the matrix
  rather than hand-picked examples, so the effect of `is_reserved` on an actual Reserved
  List card was not examined.

---

## T9 — Underpriced-card backtest

See [ADR-023](../../docs/adr/ADR-023-card-recommendation-strategy.md#backtest) for the
method and the measured hit rate of the `/underpriced` flag.
