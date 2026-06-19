# ADR-017: LightGBM as the Primary Gradient Boosting Library

## Context

The ML module requires a regression model for predicting MTG card prices.
Three candidates were considered: scikit-learn's ensemble methods, XGBoost, and LightGBM.
TensorFlow was also evaluated as an option.

The data has a documented statistical property that directly constrains the choice:
ceny kart MTG follow a Pareto distribution with α = 1.303 (confirmed in Statistical Properties 01).
Because 1 < α < 2, the distribution has a **finite mean but infinite theoretical variance**.
This rules out any loss function that squares errors (MSE) — a single €2000 outlier would
dominate the entire gradient signal. MAE and Huber are robust to this; both are first-class
objectives in LightGBM and XGBoost, but not in scikit-learn's `GradientBoostingRegressor`.

A secondary requirement is compatibility with Optuna for hyperparameter tuning (T8)
and MLflow for experiment tracking — both libraries have mature integrations with
LightGBM and XGBoost.

## Decision

Use **LightGBM** (`lightgbm` package) as the primary gradient boosting library.

## Consequences

### Positive
- Native `objective='mae'` and `objective='huber'` — correct choice given infinite-variance data.
- Built-in early stopping (`lgb.early_stopping`) with a validation set — no manual epoch tuning.
- Leaf-wise tree growth is 2–10× faster than level-wise (XGBoost default) on datasets
  of this size.
- Native categorical feature support — no manual encoding required for `rarity`, `set_type`, etc.
- `lgb.train()` returns a `Booster` with `best_iteration` — trivial to restore best checkpoint.

### Negative
- Leaf-wise growth can overfit more aggressively than level-wise on small datasets;
  `min_child_samples` must be tuned carefully (Optuna T8 handles this).
- Slightly less community documentation than XGBoost, though both are mature.

### Neutral
- `LightGBMParams` is a `dataclass` so `vars(params)` converts directly to the dict
  that `lgb.train()` expects and `mlflow.log_params()` accepts.

## Why Not XGBoost

XGBoost would also have been a valid choice. The decision margin is narrow:

| Criterion | LightGBM | XGBoost |
|---|---|---|
| MAE/Huber objective | yes | yes |
| Early stopping | yes | yes |
| Tree growth strategy | leaf-wise (faster) | level-wise (more conservative) |
| Categorical features | native | requires encoding |
| Memory usage | lower | higher |
| Optuna integration | mature | mature |

LightGBM was chosen because it is the current industry default for new tabular projects
and offers a minor practical advantage in speed and memory. If benchmark results (T8)
show XGBoost achieving materially better MAE on walk-forward CV, migrating is low-effort —
the `LightGBMPriceModel` interface (`fit` / `predict` / `feature_importance`) is compatible
with a drop-in XGBoost wrapper.

## Why Not TensorFlow / Keras

TensorFlow is optimised for unstructured data (images, text, sequences) and requires
substantially more engineering effort for tabular regression: manual feature preprocessing,
architecture search, and GPU infrastructure. For a dataset with ~25 tabular features,
gradient boosting consistently matches or outperforms neural networks at a fraction of
the operational complexity. There is no scenario in this project where TensorFlow would
be justified over LightGBM.

## Why Not scikit-learn

`sklearn.ensemble.GradientBoostingRegressor` and `RandomForestRegressor` lack:
- First-class MAE objective (only available via `criterion='absolute_error'` with significant
  performance penalty in `GradientBoostingRegressor`).
- Native early stopping tied to a validation set.
- GPU or distributed training paths for future scaling.

`HistGradientBoostingRegressor` is closer but still lacks the MLflow / Optuna ecosystem
depth that LightGBM and XGBoost provide.

## Mechanism: Why Trees Beat Networks on Tabular Price Data

Neural networks learn by multiplying matrices and passing values through activation functions.
To learn that `edition = Alpha` correlates with a high price, the network must encode this
signal across combinations of weights in potentially hundreds of neurons. This works well
when training examples are dense — but MTG has ~25,000 distinct cards, many printed only once
30 years ago. A network either overfits those rare cases or never learns them at all.

A decision tree asks binary questions directly:

```
Is edition == Alpha?
├─ YES → Is condition == NM?
│           ├─ YES → predict: €2,400
│           └─ NO  → predict: €1,200
└─ NO  → Is card a tournament staple?
             └─ ...
```

LightGBM builds 1,000 such trees sequentially, each correcting errors from the previous
(gradient boosting). Three concrete advantages for MTG pricing:

1. **Sparse, high-cardinality categoricals**: "edition = Alpha" is a single tree branch.
   A network must embed "Alpha" as a vector trained on enough Alpha examples — infeasible
   for cards with limited market history.

2. **Pareto-distributed targets with MAE**: Each tree leaf minimises MAE within its region.
   A single €2,000 outlier affects only the leaf where it lands, not the entire gradient.
   With MSE in a neural network, the same outlier propagates squared error through every weight.

3. **Interaction detection without feature engineering**: The interaction
   `(edition = Alpha) AND (condition = NM)` is two tree levels. A network must learn this
   interaction implicitly, or the engineer must create an explicit cross-feature.

## Alternatives Considered

| Library | Reason not chosen |
|---|---|
| XGBoost | Functionally equivalent; LightGBM preferred for speed and native categoricals |
| TensorFlow/Keras | Wrong tool for tabular data; excessive engineering overhead |
| scikit-learn GBM | No first-class MAE objective; no built-in early stopping on val set |
| scikit-learn RF | No sequential tree building; cannot use gradient-based early stopping |
| CatBoost | Also valid; less common in Optuna tutorials, smaller community relative to LightGBM |
