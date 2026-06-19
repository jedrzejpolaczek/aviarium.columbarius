# Data Analysis Notebooks — aviarium.columbarius

Pre-ML analysis for MTG card price prediction. These notebooks cover the full analytical
pipeline before any model is trained: data quality, statistical properties, hypothesis
testing, Bayesian modelling, and baseline benchmarking.

The goal is not to produce a model — it is to understand the data well enough to
make every modelling decision on solid ground. Skipping this work does not save time;
it moves the debugging cost into the model training phase where it is far harder to
diagnose.

---

## Prerequisites

Before running any notebook, the data pipeline must have completed through the **Gold tier**:

```
scripts/run_pipeline.py  →  Bronze  →  Silver  →  Gold
```

Verify with:
```python
import duckdb
gold = duckdb.connect("../../data/gold/cards.duckdb", read_only=True)
gold.execute("SELECT COUNT(*) FROM gold_card_features").fetchone()  # should be ~521k
gold.execute("SELECT COUNT(*) FROM gold_price_features").fetchone() # should be ~3.3M
```

**Additional library for Bayesian section only:**
```
uv add pymc arviz
```
Install only when you reach `bayesian_analysis/`. The first four sections work with
`duckdb`, `pandas`, `numpy`, `scipy`, `statsmodels`, and `sklearn` — all already
in the project.

**Minimum price history required:**
- `exploratory_data_analysis/` and `confirmatory_data_analysis/` — works with any history
- `statistical_properties/02_stationarity` — needs ≥ 30 Scryfall snapshots
- `statistical_properties/03_seasonal_decomposition` — needs ≥ 60 snapshots for useful results
- `model_preparation/04_baseline_models` — needs ≥ 14 snapshots per card for t+7 targets
- `model_preparation/` full target t+30 — needs ≥ 60 snapshots (not available yet)

---

## Execution Order and Dependencies

The sections are designed to be read in order. Each section builds on findings from
the previous one. The diagram below shows hard dependencies (files passed between
notebooks) and soft dependencies (findings that inform decisions).

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  PHASE 1 — exploratory_data_analysis/          (no external dependencies)   │
│                                                                               │
│  01 → 02 → 03 → 04   (sequential within the section, but 02/03/04 can be   │
│                         run independently once 01 passes quality checks)     │
└───────────────────────────────┬─────────────────────────────────────────────┘
                                │ findings: distribution shape, outlier caps,
                                │ forward-fill quality, feature–price signals
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  PHASE 2 — statistical_properties/             (informed by EDA findings)   │
│                                                                               │
│  01 → 02 → 03 → 04   (01 and 02 should be run before 03 and 04)            │
│                                                                               │
│  Key output: is log1p(EUR) the right transform? Are prices I(0) or I(1)?   │
└───────────┬───────────────────────────────────────────────────────────────--┘
            │ findings: stationarity result, distribution test result,
            │ autocorrelation at lag-1 and lag-7
            ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  PHASE 3 — confirmatory_data_analysis/         (informed by phases 1 + 2)  │
│                                                                               │
│  01 → 02 → 03 → 04   (independent within the section)                      │
│                                                                               │
│  Key output: which card attributes have confirmed statistical signal        │
└───────────┬───────────────────────────────────────────────────────────────--┘
            │ findings: confirmed hypotheses, effect sizes, ROPE results
            ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  PHASE 4 — bayesian_analysis/                  (informed by phases 1–3)    │
│                                                                               │
│  01 → 02 → 03 → 04   (strictly sequential — 01 produces priors_config.json)│
│                                                                               │
│  Key output: posterior credible intervals, hierarchical price estimates     │
└───────────┬─────────────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  PHASE 5 — model_preparation/                  (consolidates all phases)   │
│                                                                               │
│  01 → 02 → 03 → 04   (strictly sequential — each produces a config file)   │
│                                                                               │
│  01 → leakage_config.json                                                   │
│  02 → feature_sets.json          (reads leakage_config.json)                │
│  03 → validation_config.json                                                │
│  04 → baseline_benchmark.csv     (reads all three config files)             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Section-by-Section Guide

---

### `exploratory_data_analysis/` — Start here

EDA is hypothesis-generating, not hypothesis-testing. The goal is to build an accurate
mental model of the data before any formal claims are made. Every visualisation here
is asking "does this look like what I expected, and if not, why?" Formal tests come
later — rushing to statistics without understanding the data's shape leads to applying
the wrong tests and misinterpreting their results.

No p-values appear in this section. Only descriptive statistics and plots.

---

#### `01_data_structure_and_quality.ipynb`

Before any analysis is meaningful, the data must be structurally sound: the right
tables exist, primary keys are unique, join logic worked correctly, and the price
history has no silent gaps. This notebook is a pre-flight check. Problems found here
— a pipeline bug, a broken join, unexpected NULL patterns — are cheap to fix now and
catastrophic to discover inside a trained model.

We check NULL rates per column and per source because 81% of cards lack MTGJson data
(Scryfall stores all language/promo variants; MTGJson covers only canonical English
printings). This is not a data error — it is structural. But it means any feature
derived from MTGJson will be missing for ~421k cards, and the model will need a
separate imputation path for those cards. Knowing this now shapes every downstream
decision about features and model architecture.

We also verify forward-fill run lengths in Silver. The pipeline fills price gaps by
carrying the last known price forward. If a card's price goes missing for two weeks
and gets forward-filled, the 7-day rolling average becomes meaningless — it is just
the same stale price repeated seven times. Understanding how long forward-fills
typically run sets the boundary for which rolling features are safe to use.

**What to look for:**
- All Gold tables present with expected row counts (~521k cards, ~3.3M price rows)
- NULL rate 81.1% for MTGJson-sourced columns — expected and structural
- `mana_value` NULL 81.1% in Silver — this is a **pipeline bug** (missing fallback to
  Scryfall `cmc`). Flag it and fix the Silver pipeline before modelling
- 11 `oracle_id` name conflicts — all split cards. Exclude these 11 from training
- Forward-fill run-length p95 ≤ 7 days — if higher, a staleness flag is needed in Gold
- Date spine max gap ≤ 2 days — expected (bi-daily MTGJson cadence)

**Decision gate:** If Gold tables are missing or row counts are off by >10%, stop
and re-run the pipeline before continuing.

---

#### `02_distributions.ipynb`

A regression model trained on raw card prices would be dominated by the handful of
cards costing thousands of euros, while giving almost no gradient signal to the 99%
of cards below €5. The solution is a price transformation that compresses the scale.
This notebook establishes which transformation is correct and confirms it visually
before we commit to it project-wide.

We use `log1p(EUR)` rather than `log10(EUR + 0.01)` because `log1p` is gentler near
zero (a card at €0.01 does not become −2 on the scale), and its inverse `expm1` is
numerically stable. We also examine the foil premium, the multi-market comparison
(EUR vs Cardmarket vs USD), and the price tier breakdown — all of which shape later
modelling decisions. For example, discovering that Tier 3 contains only ~139 cards
is the first signal that a standard ML model will overfit there; that conversation
gets picked up properly in the Bayesian section.

The mana value outlier (max = 1,000,000) is a concrete example of why this step
cannot be skipped. Without inspecting distributions, that value would silently
distort any model using mana value as a feature.

**What to look for:**
- EUR price distribution is power-law shaped — `log1p(eur)` should produce a roughly
  symmetric histogram; confirm visually before accepting it as the standard
- `mana_value` max = 1,000,000 (Infinity tokens) — list all cards with `mana_value > 20`
  to confirm a hard cap at 10 is appropriate
- Tier breakdown: ~99.2% Tier 1 (<€100), ~0.6% Tier 2, ~0.2% Tier 3 (~139 cards)
- Foil premium below 1.0 (foil cheaper than non-foil) is real in older sets — keep these
- Cardmarket / Scryfall EUR ratio should cluster near 1.0; systematic offset = join mismatch

**Key decision made here:** `log1p` confirmed as the project-wide price transformation.
All subsequent notebooks, model targets, and evaluation metrics use this scale.

---

#### `03_time_series.ipynb`

Price prediction is fundamentally a time series problem, and this notebook is the
first place we look at the time dimension directly. Before running any test of
stationarity or autocorrelation, we should have visual intuition for what the
data looks like: are prices stable day-to-day? Do cheap and expensive cards behave
differently? Does the rolling average track prices smoothly or does it create long
flat plateaus where the data is actually forward-filled?

We also inspect the `is_price_spike` feature here. The pipeline flags any 1-day
price change over 300% as a spike. In theory this captures genuine market events
(ban announcements, new deck discoveries). In practice, many "spikes" turn out to
be artefacts of the first day a card enters the price history — the lag feature
computes the change from a forward-filled zero to the actual price, producing a
14,000% spike that has no market meaning. Understanding how many spikes are real
vs artefactual determines whether this feature is usable or needs redefinition.

The kurtosis of log-returns is measured here as an input to a practical modelling
decision: fat tails (kurtosis > 3) mean large errors are more common than a normal
distribution would suggest. MAE handles this better than MSE because it does not
square the error — a model that occasionally misses by €500 on a €2,000 card is not
penalised 10,000× harder than a model that misses by €50. This choice gets carried
into `model_preparation/04`.

**What to look for:**
- `price_7d_avg` smooths the raw line without visible lag — flat plateaus = forward-fill artefacts
- `is_price_spike` top-30 inspect: if >80% are first-day artefacts, the feature needs
  redefinition before use in training
- Log-return kurtosis > 3 confirms fat tails → use MAE rather than MSE as training loss
- Verify empirically that LAG(7) rows back = 7 calendar days (holds by construction
  with a complete date spine, but must be confirmed once before trusting)

---

#### `04_feature_relationships.ipynb`

Having established that the data is clean and that we understand its shape, we can
now ask which card attributes actually correlate with price. This is the EDA version
of feature importance — exploratory, not definitive.

Before computing any correlations, this notebook first validates that the chosen
correlation measures are appropriate for this dataset. Prices follow a Pareto
distribution (α ≈ 1.303 — confirmed in `statistical_properties/01`), which means
theoretically infinite variance. Several candidate features have discrete value ranges
with many tied ranks. Using Pearson or Spearman without checking these conditions
risks interpreting measurement artefacts as real signal. Four checks are performed
upfront:

- **Tied ranks:** discrete features like `finish_count` (values 1–3) and `format_count`
  inflate Spearman rank distances. We compare Spearman ρ against Kendall's τ per feature —
  large divergence (|Spearman − Kendall| > 0.05) means Spearman is biased and Kendall's τ
  should be preferred for that feature in the CDA section.
- **Heavy tails:** Winsorizing at the 1st/99th percentile before computing Pearson reveals
  how much the raw Pearson r is distorted by extreme price outliers. Large divergence from
  the winsorized value means Pearson is unreliable for that pair; Spearman or Kendall's τ
  should be used instead.
- **Bivariate outliers:** a card can be within the univariate distribution of both its
  feature and its price, yet be a joint outlier that distorts the correlation estimate.
  Mahalanobis distance (chi² threshold, df = 2, α = 0.001) identifies these pairs.
  Features with a bivariate outlier rate above 1% warrant extra caution when interpreting
  their Pearson correlation.
- **Sampling stability:** with n ≈ 80k cards, any correlation will be statistically
  significant by construction. Bootstrap confidence intervals (1,000 resamples) for the
  top features reveal whether the *magnitude* of Spearman ρ is stable or whether it is
  sensitive to which cards happen to be in the sample.

With assumption checks complete, we compute Pearson, Spearman, and Mutual Information
across all 24 candidate features. We use three complementary measures because each
captures something different. Pearson measures linear relationships and is familiar,
but it systematically underestimates the strength of binary features like `is_reserved`.
Spearman measures monotone relationships and handles the non-normality of price
distributions well. Mutual Information measures any statistical dependence, linear or
not — it is the right tool for binary and categorical features, and for features like
`mana_value` which may have a non-monotone relationship with price (both zero-CMC and
seven-CMC cards can be expensive for different reasons).

We also compute partial Spearman for EDHREC rank, controlling for rarity. This is
important because EDHREC rank and rarity are correlated — mythic rares tend to be
both more played and more expensive. Without controlling for rarity, we cannot tell
whether EDHREC rank carries independent predictive signal or is simply acting as
a proxy for rarity. If the partial correlation drops below 0.1, EDHREC rank is not
worth including as a feature beyond what rarity already captures.

The output of this notebook is a preliminary ranked list of feature candidates. It
is not the final feature set — that requires VIF and a formal MI analysis in
`model_preparation/02` — but it eliminates features that clearly have no signal
before we invest in more rigorous analysis.

**What to look for:**
- Assumption checks: features with |Spearman − Kendall| > 0.05 or bivariate outlier
  rate > 1% should not be reported by Pearson alone — note these before reading the
  correlation matrix
- Bootstrap CI width > 0.05 for any top feature: re-examine that feature's Spearman
  result before trusting it in CDA
- MI ranking approximately: `is_reserved > edhrec_rank > print_count > format_count > mana_value`
  — a very different order warrants investigation
- Large Pearson vs Spearman gap for binary features confirms Pearson is undercounting
  their importance; trust MI for those features
- EDHREC partial ρ after rarity control: if it drops below 0.1, edhrec_rank adds
  nothing beyond rarity
- Tournament data covers ~60 cards (scraper cap) — treat as directional, not definitive

**Output that carries forward:** a preliminary ranked list of feature candidates,
to be formally validated in `model_preparation/02_feature_selection`.

---

### `statistical_properties/` — Verify assumptions before testing

EDA tells us what the data looks like. This section asks whether those visual
impressions hold up to formal scrutiny. The specific questions answered here are
not interesting in isolation — they are prerequisites. Getting them wrong means
applying the wrong tests in the CDA section, building models on non-stationary
series, or discovering that your assumed transformation actually makes the
distribution worse.

---

#### `01_distribution_testing.ipynb`

The choice of `log1p` was made visually in `eda/02`. Here we verify it formally.
We use the Kolmogorov-Smirnov and Anderson-Darling tests to measure how far
`log1p(EUR)` deviates from a normal distribution. Anderson-Darling is the preferred
test because it gives more weight to the tails, which is where MTG prices have the
most structure — the difference between a €300 dual land and a €500 dual land matters
more than the difference between two €0.05 commons.

There is an important subtlety: at n ≈ 82,000, both tests will reject normality for
any real dataset. With this many observations, the test has enough power to detect
deviations that are statistically real but practically irrelevant — a distribution
that deviates from normal by 0.5% still fails at n = 82k. For this reason we focus
on the **KS D statistic** (the maximum absolute distance between the empirical and
theoretical CDFs) rather than the p-value. D < 0.02 means the transformation is
working well enough for modelling purposes.

We also run Levene's test for homoscedasticity across rarity groups. This is not
about whether prices are normally distributed — it is about whether the variance
of log-prices is the same for common, uncommon, rare, and mythic cards. If it is
not (and it almost certainly is not), a single linear model treating all cards
identically will have systematically miscalibrated confidence intervals. The Levene
result directly determines whether `model_preparation/04` should use one pooled model
or tier-segmented models.

Finally, we fit a Pareto / power-law model to the tail. If the top 5% of prices
follow a power-law, that tells us the prior distribution to use for Tier 3 cards
in the Bayesian model — a log-normal prior on a power-law tail is a modelling error.

**What to look for:**
- KS D < 0.02 on `log1p(eur)` = transformation is practically adequate regardless of p-value
- Levene test on rarity groups: almost certainly rejected — confirms the case for
  segmented models or weighted regression
- If the Zipf plot is linear for the top 5% of prices, the tail is power-law distributed —
  use a Pareto prior for Tier 3 in `bayesian_analysis/02`, not a normal prior

**Decision made here:** Levene result determines whether `model_preparation/04` uses
a single model or tier-segmented models, and informs the Bayesian prior family for Tier 3.

---

#### `02_stationarity.ipynb`

This is one of the most consequential notebooks in the entire pipeline. The choice
of what to model as the target — price levels vs price changes — depends entirely on
whether the price series is stationary.

A stationary series has constant mean and variance over time. A non-stationary series
(unit root process) drifts without bound — its mean and variance grow with time.
Fitting a regression on non-stationary data produces what statisticians call spurious
regression: the model appears to explain variance extremely well (R² close to 1) but
is actually just tracking the shared drift of two unrelated trending series. In plain
terms: the model looks good in backtesting but predicts nothing useful.

We use two complementary tests: ADF (Augmented Dickey-Fuller) and KPSS
(Kwiatkowski-Phillips-Schmidt-Shin). They test opposite null hypotheses, so using
them together eliminates ambiguity. ADF's null is "series is non-stationary";
KPSS's null is "series is stationary". If ADF fails to reject and KPSS rejects,
the series is I(1) with high confidence. This is the expected outcome for price
levels — and if confirmed, the model must predict log-returns (differences), not
levels.

We also run Ljung-Box on the log-returns themselves to test whether those differences
have autocorrelation. This is separate from the stationarity question: a series can
be stationary but still have memory (today's return correlated with yesterday's).
If ACF(1) > 0.3 and Ljung-Box at lag 1 is significant, the lag-1 return carries
predictive signal and belongs in the feature set. Note the max-lag constraint: with
n ≈ 90 days, the maximum reliable lag to test is 18 (n/5). Testing at lag 30 with
n = 90 produces unreliable results and was an error in the original `cda_01` notebook.

**What to look for:**
- `log1p(EUR)` levels: expect I(1) — ADF not significant, KPSS significant
- Log-returns: expect I(0) — ADF significant, KPSS not significant
- Use the ADF+KPSS combination table in the notebook for the unambiguous conclusion
- Ljung-Box max lag = n // 5; if ACF(1) > 0.3 with significant p, lag-1 return is a feature
- Compare stationarity per tier: Tier 3 (Power Nine, Reserved List) may show stronger
  autocorrelation than Tier 1, warranting a separate model structure

**Decision made here:** If prices are I(1) (expected), the ML target is confirmed
as log-returns. The Ljung-Box result determines whether lag-1 and lag-7 return
features belong in the model.

---

#### `03_seasonal_decomposition.ipynb`

MTG releases new sets approximately every three months. Each release changes the
competitive landscape, adds new cards to the market, and often reprints older cards —
directly affecting supply and demand. This cyclicality is a form of seasonality.
If it is present in the price data, models that ignore it will make systematically
biased predictions around set-release periods.

We use STL (Seasonal and Trend decomposition using Loess) because it is robust to
the price spikes and outliers that are common in MTG data. Classical additive
decomposition would misallocate a spike caused by a ban announcement into the
seasonal component, distorting the seasonality estimate for the rest of the series.
STL handles this by iteratively down-weighting outliers in the seasonal smoother.

We also run a Mann-Kendall trend test before decomposing. Mann-Kendall is
non-parametric and does not assume normality — it simply tests whether the series
has a monotone upward or downward tendency. If the market is steadily inflating,
models trained on older prices will underpredict future prices in a systematic way.
Knowing this informs how the training window is structured in `model_preparation/03`.

The honest caveat: with < 180 days of history, the seasonal period (90 days for
quarterly releases) fits less than twice in the data. STL needs at least two full
cycles to reliably separate seasonal from trend. Results here are therefore
preliminary and should be revisited after ≥ 180 days of accumulation.

**What to look for:**
- Mann-Kendall τ and direction: is the market trending and how strongly?
- STL residuals should look like white noise — if ACF of residuals has structure,
  there is a pattern the decomposition missed
- Amplitudes: if Tier 3 seasonal amplitude is near-zero, expensive cards are driven
  by speculation rather than release cycles — reinforcing the case for a separate model
- Treat all results here as preliminary until ≥ 180 days of history are available

**Re-run trigger:** Repeat when ≥ 180 days of history are available.

---

#### `04_cointegration.ipynb`

The Gold tier stores three independent price sources for the same cards: Scryfall EUR,
Cardmarket EUR, and TCGPlayer USD. If these series move together over the long run —
rising and falling together around a stable ratio — they are cointegrated. Cointegration
matters because a cointegrated pair contains more information than either series alone:
when the spread between them widens, it will tend to revert to its long-run equilibrium.
That spread is a stationary, predictable signal.

The Engle-Granger test formalises this: it fits a regression between two series and
tests whether the residuals are stationary. If they are, the pair is cointegrated and
the residual (the spread) is a valid feature. The half-life of mean reversion tells
us how many days it takes for the spread to halve — if it is 3 days, we have a
short-term arbitrage signal; if it is 60 days, the signal is too slow to be useful
for 7-day predictions.

This analysis only makes sense if the price series are I(1), which is why it runs
after `02_stationarity`. Cointegration of I(0) series has no statistical meaning.

**What to look for:**
- Run only if `02_stationarity` confirmed prices are I(1)
- EUR and Cardmarket EUR should be cointegrated — rejection of the test would suggest
  a pipeline join mismatch, not a genuine market anomaly
- Half-life of the cointegrating spread: < 7 days means the spread is a candidate
  feature for 7-day price prediction models
- If EUR and USD are cointegrated and the implied exchange rate is stable near ~0.92,
  the models can cross-validate predictions across currencies

---

### `confirmatory_data_analysis/` — Test specific hypotheses

EDA showed us patterns and CDA checks them. The key distinction between the two:
in EDA we look at everything and note what is interesting; in CDA we pre-specify
hypotheses derived from domain knowledge, then test them with the appropriate
statistical machinery. Testing a hypothesis you formed *after* looking at the data
is not CDA — it is EDA with extra steps, and the p-values are meaningless.

All tests here are non-parametric because `statistical_properties/01` confirmed that
price distributions have non-normal tails and heterogeneous variance across rarity
groups. Non-parametric tests (Kruskal-Wallis, Mann-Whitney U, Spearman ρ) make
no assumptions about the underlying distribution — they work on ranks rather than
values, so they are robust to the extreme skew and outliers present in MTG prices.

Effect size is always reported alongside the p-value. With n > 80,000 cards, almost
any real difference will be statistically significant — the p-value tells us only
whether an effect is distinguishable from zero, not whether it is large enough to
matter. Effect sizes tell us the latter. We use ε² (epsilon-squared) for Kruskal-Wallis
and rank-biserial r for Mann-Whitney. We use ε² rather than η² because η² is positively
biased at finite sample sizes — ε² is the less biased estimator.

Bonferroni correction is applied within each hypothesis family. When testing six
pairwise rarity comparisons at once, the probability of a false positive inflates;
Bonferroni corrects for this by dividing α by the number of comparisons.

---

#### `01_rarity_and_card_attributes.ipynb`

Rarity, the Legendary supertype, Reserved List status, and reprint history are the
four most domain-obvious predictors of price in MTG. This notebook gives each of
them a rigorous statistical treatment.

The rarity premium is tested with Kruskal-Wallis because we are comparing more than
two groups. The ε² effect size tells us whether rarity is a strong predictor or merely
a weak one that reaches significance because of sample size. A large ε² (> 0.14) means
rarity alone explains a substantial fraction of the variance in log-prices, which
would make it one of the most important features in the model.

The Legendary premium is deliberately tested **for every rarity separately**, not
just mythic. The original `cda_01` notebook made the methodological error of
selecting only the mythic result as "representative" — that is cherry-picking.
A Legendary premium that exists only at mythic rarity is a very different signal
from one that exists across all rarities. The former suggests an interaction term
(`is_legendary × rarity`) in the model; the latter suggests a simple additive feature.

The Reserved List partial Spearman controls for rarity because Reserved List cards
are disproportionately rare and mythic. Without this control, we would be measuring
the confounded effect of "Reserved List cards are expensive" partly because
"rare and mythic cards are expensive." The partial correlation isolates the Reserved
List effect specifically due to the no-reprint guarantee, above and beyond rarity.

Levene's test on rarity groups directly connects to the discussion in
`statistical_properties/01` about heteroscedasticity. Consistent results across
both notebooks increase confidence in the finding.

**What to look for:**
- H1 (rarity premium): ε² effect size — is it large (> 0.14) or just significant due to n?
  The magnitude matters more than whether it reaches 0.05
- H2 (legendary premium): all four rarities separately; if the premium exists only at
  mythic, add `is_legendary × rarity` as an interaction term rather than a plain binary feature
- H3 (Reserved List): compare raw ρ vs partial ρ — a large drop means rarity was
  confounding the Reserved List signal
- H5 (Levene): expected to be confirmed; cross-reference with `statistical_properties/01`

---

#### `02_format_legality_and_demand.ipynb`

If rarity captures supply-side constraints (fewer mythic cards are printed), format
legality captures demand-side breadth — how many different player communities can
legally use this card. A card legal in Commander, Modern, Legacy, and Vintage has
a potential market of millions of players; a card legal only in Commander has a
smaller but still substantial market. This notebook tests whether that breadth
translates into a measurable price premium.

We test Commander legality separately from format_count because Commander is
disproportionately important — it is estimated to be the most-played format in MTG,
and many players buy cards specifically for Commander regardless of competitive viability.
The partial Spearman for format_count then asks: beyond Commander, does access to
additional formats add incremental value?

EDHREC rank is tested here because it measures not just whether a card is legal in
Commander but how often it is actually played. A card that is legal but never played
in Commander does not benefit from Commander demand. The partial correlation controls
for rarity to ask whether EDHREC rank has information beyond what rarity already provides.

An important coverage caveat: 82% of cards have NULL EDHREC rank (Scryfall-only cards
without MTGJson data). Any finding here applies only to the ~18k cards with rank data
and cannot be generalised to Scryfall-only cards.

**What to look for:**
- H1 (Commander legal): check base rate first — if >95% are Commander-legal, the test
  has low statistical power due to extreme group imbalance
- H2 (format_count partial ρ): if partial ρ drops below 0.1 after rarity control,
  format_count is proxying rarity, not measuring independent demand breadth
- H4 (EDHREC rank): results apply only to ~18k cards with MTGJson data; note this clearly
  when interpreting

---

#### `03_tournament_signals.ipynb`

Tournament results are a leading indicator of competitive demand. When a card
suddenly appears in multiple winning decklists, players buy it before supply adjusts —
creating a price spike. If we can detect this signal before or shortly after it
appears in tournament data, we have an edge in predicting price movements.

The critical methodological problem here is that the tournament scraper currently
collects only the top-10 cards per format — approximately 60 cards total. This is
not a random sample; it is a selected sample of the most expensive, most-played
cards. Any price premium we find between "tournament cards" and "non-tournament
cards" is almost certainly a selection artefact. We note this in the notebook
rather than pretending it does not exist.

The most valuable test in this section is the Granger causality test: does knowing
last week's tournament activity help predict next week's price movement, beyond
what last week's price alone tells us? Granger causality is not true causality —
it is a test of whether the tournament time series has predictive information
about the price series. If the answer is yes at lag k, then `top8_appearances` k
weeks ago is a legitimate predictive feature. If no, the feature is correlated with
price but does not lead it — it arrives at the same time or after, and cannot be
used for forward prediction.

**What to look for:**
- Selection bias is unavoidable with the current scraper cap — document it and do not
  over-interpret the Mann-Whitney result
- H4 (Granger): the most actionable result; the lag tells you how far ahead tournament
  data is predictive — this directly determines how to construct the `top8_appearances` feature
- With < 30 weeks of history, skip Granger and note it requires ≥ 6 months of data

**Re-run trigger:** After fixing the scraper to collect all tournament cards (not top-10)
and after ≥ 6 months of history.

---

#### `04_temporal_stability_and_autocorrelation.ipynb`

A model trained on six months of old data and deployed to predict prices today is
making an implicit assumption: that the statistical relationship between features
and price changes has not shifted over those six months. This notebook tests that
assumption directly.

We use a two-sample KS test comparing the price distribution in the first half of
the history against the second half. If the distributions have drifted (KS D > 0.1),
the oldest training data may be more harmful than helpful — it teaches the model
patterns that no longer hold. This directly informs the trade-off between expanding
(uses all history) and sliding (uses only recent history) training windows in
`model_preparation/03`.

The Ljung-Box test revisits the autocorrelation question from `statistical_properties/02`,
this time within the CDA framework — as a formal hypothesis test rather than a
diagnostic. The max-lag constraint (n // 5) is strictly enforced here because
`cda_01` made the error of testing at lag 30 with n = 90, which produces unreliable
results with almost no power. At n ≈ 90, reliable lags are 1, 7, and 14 at most.

Testing autocorrelation separately for Tier 1 and Tier 3 cards is motivated by the
hypothesis that expensive cards behave differently: Reserved List speculation creates
momentum (today's price rise makes tomorrow's rise more likely), while cheap bulk
commons are essentially random noise. If Tier 3 ACF(1) is substantially higher than
Tier 1 ACF(1), the two tiers need structurally different time-series models.

**What to look for:**
- H1 (KS stability): D < 0.05 = distribution is practically stable, standard temporal
  split is adequate; D > 0.1 = meaningful drift, consider a sliding training window
- H2 (Ljung-Box): max lag = n // 5 strictly; ACF(1) > 0.3 with significant Ljung-Box
  means `price_change_1d_pct` belongs in the feature set
- H3 (per-tier ACF): higher Tier 3 ACF supports a separate autoregressive component
  for the Bayesian time-series model in `bayesian_analysis/03`

---

### `bayesian_analysis/` — Uncertainty quantification

Classical ML models produce point predictions. A XGBoost model trained to predict
7-day price changes will output "this card will be worth €2.50 in seven days" with
no indication of how confident that prediction is or what range of prices is plausible.

For most of the ~90,000 Tier 1 cards, that limitation is acceptable — the training
data is plentiful. But for the ~139 cards in Tier 3 (Power Nine, Dual Lands), a
point prediction is nearly meaningless: the model has almost no data for those cards
and is likely extrapolating far outside its training distribution. In this regime,
knowing the uncertainty is more valuable than the point estimate itself.

Bayesian models replace point estimates with posterior distributions — a full
probability distribution over all plausible values of the parameter being estimated.
A Bayesian model does not say "this card will be worth €2.50"; it says "given
everything we know, there is a 90% probability this card will be worth between €1.80
and €3.40." That credible interval is the core value this section adds.

**Install before starting this section:**
```
uv add pymc arviz
```

---

#### `01_prior_elicitation.ipynb`

Every Bayesian model begins with prior distributions — our beliefs about parameter
values before seeing any data. Choosing a prior is the most subjective step in
Bayesian analysis and must be done carefully. Two failure modes exist: priors that
are too narrow force the posterior toward the wrong values regardless of the data;
priors that are too wide let the model wander into physically impossible territory
(negative prices, or prices of €10,000,000 for a common card).

We use weakly informative priors: centred at the empirical median per rarity but
with twice the empirical standard deviation. "Weakly informative" means the prior
is wide enough to allow the data to dominate the posterior, but narrow enough to
exclude implausible values. The 2× factor is a deliberate choice — with only 90
days of price history, we do not want priors so tight that they cannot accommodate
a market shift if a new set changes baseline prices.

The critical methodological point is that priors must be elicited from a held-out
portion of the data — specifically the older half of the price history. Using all
available data for both prior elicitation and inference (likelihood) is a form of
double-dipping: the prior already "knows" about the data, making the posterior
appear more confident than it should be. This data split is enforced throughout
the Bayesian section.

Prior predictive checking validates the prior before any inference: we sample
directly from the prior and check that the generated prices look like plausible
MTG prices. If the prior generates cards costing −€200 or €50,000,000, it needs
tightening regardless of what the data say.

**What to look for:**
- Prior predictive samples must contain 0% negative prices
- The 95% prior predictive interval for each rarity should envelope the observed empirical
  distribution — if not, the prior is too tight and will override the data
- Sensitivity check: wide (3×), default (2×), and narrow (1×) prior variants should
  produce similar predictive intervals; large divergence means the data are too sparse
  to overcome prior assumptions and the choice of prior matters more than the data

**Output file:** `priors_config.json` — required by notebooks 02, 03, and 04.

---

#### `02_hierarchical_price_model.ipynb`

This model addresses a specific problem: how do we estimate the expected price of
a card from a set we have never seen before? A standard regression model has no
way to do this — it averages over all sets and produces the same prediction for
a card from a 2024 Masters set as for a card from a 1993 Alpha printing. A
hierarchical model solves this by learning the distribution *of* set-level effects,
and then placing new sets at that distribution's centre with appropriately wide
uncertainty.

The three-level hierarchy (card → set → rarity) encodes domain knowledge directly
into the model structure. Cards in the same set share information about their set's
"price level" (e.g., Masters sets have suppressed prices due to the reprint purpose).
Sets in the same rarity category share information about the baseline for that rarity.
This sharing — called partial pooling — is the main statistical advantage: the model
uses more information than a fully independent model (which estimates each set
in isolation) but less than a fully pooled model (which ignores set differences).

MCMC diagnostics are non-negotiable before interpreting any result. The sampling
algorithm explores the posterior distribution by running multiple independent chains.
If those chains end up in different parts of the parameter space (high R-hat) or
explore it inefficiently (low ESS), the posterior approximation is unreliable.
Divergences are the most serious indicator — they signal regions of the parameter
space where the sampler cannot go but the posterior has probability mass. A model
with divergences has an incorrect posterior.

**What to look for:**
- R-hat < 1.01, ESS > 400, zero divergences — these are gates, not suggestions.
  Do not interpret any posterior until all three are satisfied
- If R-hat > 1.01: increase `tune` to 2000 or apply non-centred reparametrisation
  for `mu_set` (standard fix for hierarchical models with convergence issues)
- Posterior predictive coverage ~90%: if substantially below, `sigma_obs` prior is too
  narrow; the model is overconfident
- `mu_set` for a new set should equal `mu_rarity[r]` — this is the key advantage:
  sensible defaults for cards in sets the model has never seen

**Output file:** `hierarchical_price_trace.nc`

---

#### `03_bayesian_time_series.ipynb`

The hierarchical model estimates the *level* of a card's price based on its static
attributes. This notebook adds a *temporal* component: how does the price evolve
day to day, and what can we say about its trajectory 7 or 30 days from now?

We use a Local Level Model (also called a random walk with noise). This is the
simplest Bayesian time-series model that can capture price drift: the true underlying
price level follows a random walk, and what we observe is that level plus measurement
noise. Two parameters control the model: `sigma_eta` (how fast the underlying level
drifts) and `sigma_eps` (how noisy the observations are). A card with large `sigma_eta`
is speculatively volatile — its price can shift dramatically from week to week.
A card with small `sigma_eta` is stable — its price moves slowly and predictably.

The Local Level Model is chosen over ARIMA or SARIMA because it is interpretable,
has only two parameters (reducing overfitting risk with short histories), and
naturally produces posterior predictive distributions rather than point estimates.
With n ≈ 30–90 observations per card, a more complex model would be under-identified.

The main output is not better point predictions — the naive baseline often matches
or beats the point estimate. The output is calibrated uncertainty: a 90% credible
interval that actually contains the true price 90% of the time. That calibration
is what justifies the Bayesian approach and is what gets used in production for
communicating price uncertainty to users.

**What to look for:**
- `sigma_eta` should differ substantially between stable bulk commons and volatile
  Reserved List cards — similar values across both groups = model not capturing dynamics
- 90% HDI coverage on test set ≈ 90%: below = intervals too narrow (overconfident);
  above = intervals too wide (uninformative)
- The point estimate (posterior median) is often similar to the naive baseline —
  this is expected and does not mean the model failed; the value is in the uncertainty interval
- Cards with < 30 days history: fall back to the hierarchical model's static estimate

---

#### `04_bayesian_hypothesis_testing.ipynb`

This notebook re-examines the hypotheses from `confirmatory_data_analysis/01` using
Bayesian methods. It is not redundant with CDA — it adds a different kind of answer.

The frequentist CDA tests told us whether each effect is distinguishable from zero
(p < 0.05) and how large it is (effect size). What they cannot tell us is: what is
the probability that the Reserved List premium is greater than €2 per card? How much
more likely is the data if the rarity premium exists compared to if it does not?
Is the Legendary premium large enough to matter for a trading decision, or is it
statistically real but practically negligible?

ROPE (Region of Practical Equivalence) formalises the last question. We define a
range around zero — say, ±10% price difference in log1p scale — within which we
would consider the effect "too small to matter for a buy/sell decision." If the
94% credible interval for an effect falls entirely outside the ROPE, the effect
is both statistically real and practically meaningful. If it overlaps the ROPE,
the effect may exist but is too small to inform trading decisions.

Bayes Factors compare the probability of the observed data under two competing
models: H₁ (effect exists) vs H₀ (no effect). BF₁₀ = 100 means the data are
100 times more likely if the effect exists than if it does not. This is a more
direct answer to "how much evidence do we have?" than a p-value, which only
measures "how surprising would this data be if there were no effect?"

**What to look for:**
- Reserved List ROPE analysis: 94% HDI entirely outside [-0.1, 0.1]? If yes,
  the premium is large enough to matter for trading; if it overlaps, it is real but modest
- `P(effect > 0)` for Legendary per rarity — "98% probability the premium exists for
  mythic" is directly actionable; "p = 0.003" is not
- Bayes Factor for the rarity premium: BF₁₀ > 10 = strong evidence; cross-check
  with the Kruskal-Wallis result from CDA/01
- Wide posteriors for Tier 3 are correct behaviour with ~139 cards — do not treat
  wide credible intervals as a model failure; they are honest uncertainty quantification

---

### `model_preparation/` — Ready the ML training setup

The previous four sections built understanding. This section turns that understanding
into concrete artefacts: a list of features that are safe to use, a validated training
pipeline, and a set of baseline numbers that every future model must beat. These
notebooks are strictly sequential — each produces a config file that the next one
reads. Running them out of order will fail.

---

#### `01_leakage_and_target_definition.ipynb`

Data leakage is the most dangerous silent failure mode in ML. A model with leakage
appears to work perfectly in evaluation — sometimes achieving near-perfect accuracy —
while predicting nothing useful in production. The cause: training features that
contain information about the future, information that would not exist at the moment
of making a real prediction.

The Gold tier contains three confirmed leakage columns. `price_ath` is computed as
`MAX(eur) OVER (PARTITION BY uuid)` — an unbounded window over the card's entire
price history. This means a row from the card's very first day in the database already
contains the all-time-high price that the card will reach months later. From the
model's perspective, this is like being given the answer. We verify this empirically
rather than inferring it from the SQL: we check that `price_ath` on day 1 equals
the maximum price across all future dates — 100% of sampled cards confirm leakage.
`price_atl` and `days_with_price` have the same structural problem.

`edhrec_rank` has a different kind of leakage: temporal. The `gold_card_features`
table is rebuilt daily with the current rank applied to all historical rows. When
training on data from 60 days ago, the model sees the card's current popularity —
not its popularity 60 days ago. For a card that became popular recently, this means
the model is told "this card is very popular" at a time when it was not yet popular,
and it learns a spurious correlation between high demand and high past prices.

The target definition — log1p(EUR[t+7]) − log1p(EUR[t]) — is finalised here.
We use a date-exact join rather than a row-based shift because a row-based LAG(7)
is only equivalent to 7 calendar days when the date spine has no gaps. With a
complete Silver date spine this should hold, but we verify empirically rather
than assuming.

**What to look for:**
- `price_ath` leakage: empirical verification should show 100% leakage rate — any
  value below 100% warrants investigation of the verification query itself
- `edhrec_rank` temporal leakage: note whether daily rank snapshots are now being
  stored in Gold; if yes, this column may become safe to use for historical training
- Target t+7 availability: count the rows with valid targets — if < 5%, training
  for the 7-day model must wait for more history
- Binary spike threshold: the +30% threshold gives ~15% positive rate (manageable
  with class weights); +50% gives ~8.5%; the choice is a business decision

**Output file:** `leakage_config.json`

---

#### `02_feature_selection.ipynb`

Having removed leakage columns, we now reduce the remaining candidate features to
a set that is both informative and non-redundant. Two problems to address: features
that carry no signal (noise that wastes model capacity and slows training), and
features that say the same thing as each other (multicollinearity, which destabilises
linear model coefficients and inflates variance in tree-based models).

We use Mutual Information rather than Pearson correlation as the primary signal
measure because ~40% of our candidate features are binary. Pearson correlation
applied to a binary feature measures only the linear component of its relationship
with the target — it systematically underestimates the strength of binary predictors.
MI measures total statistical dependence regardless of the relationship's shape.
We run MI three times with different random seeds and average the scores to reduce
the variance of the estimator.

For multicollinearity we use VIF (Variance Inflation Factor) rather than pairwise
correlation matrices. A correlation matrix shows pairs; VIF shows whether a feature
is a linear combination of *many* other features simultaneously. The three format
legality flags (`is_legacy_legal`, `is_vintage_legal`, `is_commander_legal`) have
pairwise correlations of 0.90–0.95, which would be visible in a correlation matrix.
But VIF catches the combined problem: even if no single pair were above 0.7, their
collective collinearity could give VIF = 20, indicating that the feature adds almost
no independent information.

The Tier 3 feature set is a strict subset of Tier 1's. With ~139 cards, every
additional feature increases the risk of overfitting. The guiding principle is
parsimony: use only features with strong MI evidence and low VIF, and fewer features
rather than more when the data are sparse.

**What to look for:**
- `has_mtgjson_data` should be added as a feature — it encodes whether 13+ other
  features are present or imputed, and models should know the difference
- VIF > 10 for format legality flags: replace with `format_count` as the aggregate
- MI < 0.02 for any feature: remove regardless of domain intuition — if the data
  do not show the signal, the model will not learn it
- Tier 3 feature set: keep to the highest-MI, lowest-VIF subset; fewer is better

**Output file:** `feature_sets.json`

---

#### `03_validation_strategy.ipynb`

Random cross-validation is one of the most common mistakes in time-series ML.
When a randomly-selected fold includes a row from day 80 in the training set and
a row from day 10 in the validation set, the model is trained on "future" data
to predict "past" data. It learns patterns that reverse the direction of time and
achieves excellent cross-validation scores while predicting nothing useful in
production.

Walk-forward (expanding window) cross-validation prevents this: the training set
always ends before the validation set begins, and the validation set moves forward
in time. This mimics real-world use exactly — we always train on the past to predict
the future, never the other way around.

The power analysis for Tier 3 provides the quantitative argument for why the
Bayesian approach was necessary. With ~139 Tier 3 cards and approximately 1,100
training rows at the most data-rich fold, the statistical power to detect a
medium-sized effect (d = 0.5) is below 60% at α = 0.05. This means we would miss
real effects 40% of the time — an ML model trained on Tier 3 alone is effectively
guessing. The Bayesian hierarchical model solves this by borrowing information from
similar cards in the rarity group.

The hold-out test set (last 14 days) is treated as a sealed envelope. No model
hyperparameter, no feature choice, no architecture decision should look at it.
The only permissible use is computing the final reported performance of the chosen
model after all decisions have been made. Looking at it earlier converts it into
a validation set and invalidates the evaluation.

**What to look for:**
- Walk-forward folds: with ~90 days and 30-day minimum training, expect 8–10 folds;
  fewer than 5 is insufficient for reliable model selection
- Power analysis: Tier 3 power < 60% at d = 0.5 is the quantitative justification
  for using the Bayesian model rather than standard ML for expensive cards
- Hold-out test set is sealed — do not look at it until all modelling decisions are final

**Output file:** `validation_config.json`

---

#### `04_baseline_models.ipynb`

No model should be deployed until its performance is compared to the simplest
possible alternative. This notebook implements four baselines in increasing order
of complexity: the naive forecast (predict no change), a 7-day moving average,
AR(1) autoregression, and Ridge linear regression with static card features.

The naive forecast is the absolute floor. It embodies the claim "I cannot predict
anything about how prices will change in the next 7 days, so I predict zero change."
Any model that does not beat this is not useful — it is adding complexity without
adding value.

The comparison between naive and MA7d has direct interpretive value: if MA7d is
better, prices exhibit mean reversion (they tend to return to their recent average),
which means the rolling price history is informative. If naive is better, prices
exhibit momentum (the most recent price is the single best predictor), and rolling
averages actually hurt by averaging away the signal. This is a domain question with
a real answer in the data.

The comparison between AR(1) and naive cross-validates the Ljung-Box finding from
`statistical_properties/02`. If Ljung-Box found significant lag-1 autocorrelation,
AR(1) should improve over naive. If the two results disagree, investigate why.

Ridge regression with static card features tests whether the domain knowledge
encoded in features — rarity, Reserved List status, print count, format legality —
adds incremental predictive power beyond the time-series signal alone. If MAE does
not improve over naive, the static features carry no predictive power for 7-day
returns, which would be a significant finding about the nature of the MTG market.

Critically, all metrics are computed **per tier**. A model with overall MAE = 0.05
can have Tier 1 MAE = 0.02 and Tier 3 MAE = 0.45. Aggregate performance hides
exactly the failures that matter most.

**What to look for:**
- MA7d vs Naive: mean reversion (MA7d better) vs momentum (naive better) — the result
  has implications for how rolling price features are included in ML models
- AR(1) vs Naive: should agree with Ljung-Box from `statistical_properties/02`;
  inconsistency warrants investigation
- Ridge regression improvement: the MAE improvement from Naive to Ridge, per tier,
  quantifies the marginal value of static card features for 7-day price prediction
- All results per tier — aggregate numbers hide failures that matter

**Output file:** `baseline_benchmark.csv` — the official performance bar that every
subsequent XGBoost / LightGBM / deep learning model must exceed.

---

## File Outputs and Handoffs

| Notebook | Produces | Consumed by |
|---|---|---|
| `bayesian_analysis/01` | `priors_config.json` | `bayesian_analysis/02`, `03`, `04` |
| `bayesian_analysis/02` | `hierarchical_price_trace.nc` | `bayesian_analysis/04` (reference) |
| `model_preparation/01` | `leakage_config.json` | `model_preparation/02`, `04` |
| `model_preparation/02` | `feature_sets.json` | `model_preparation/04` |
| `model_preparation/03` | `validation_config.json` | `model_preparation/04` |
| `model_preparation/04` | `baseline_benchmark.csv` | future ML model notebooks |

Store config files in the same directory as the notebook that reads them, or
update the paths at the top of each notebook if you prefer a central config location.

---

## Known Data Limitations

These are hard constraints that cannot be worked around with better analysis —
only by waiting for more data to accumulate.

| Limitation | Impact | When resolved |
|---|---|---|
| Scryfall prices: 1 snapshot (daily scraping started 2026-05-26) | t+7 targets: 11.6% of rows; t+30 targets: 0% | ≥ 37 Scryfall snapshots for t+7; ≥ 60 for t+30 |
| MTGJson prices: 90 snapshots (short for seasonality) | STL preliminary; Ljung-Box max lag = 18 | ≥ 180 days for reliable STL |
| Tier 3: ~139 cards, ~1 112 training rows | Classical ML overfits; power < 60% | Bayesian hierarchical model is the right tool now |
| Tournament scraper: top-10 per format only | Tournament signal analysis covers ~60 cards | Fix scraper to collect all tournament cards |
| `mana_value` Silver pipeline bug (missing Scryfall `cmc` fallback) | 81.1% NULL for Scryfall-only cards | Fix Silver pipeline, re-run |

---

## Quick Reference — What Each Notebook Decides

| Notebook | Decision Made |
|---|---|
| `eda/02_distributions` | `log1p` as the project-wide price transformation |
| `statistical_properties/01` | Formal confirmation of transformation; Levene → segmented vs single model |
| `statistical_properties/02` | Target = log-returns (confirmed I(1) for levels); LAG-1 feature value |
| `confirmatory_data_analysis/01` | Which card attributes have confirmed signal and practical effect sizes |
| `confirmatory_data_analysis/04` | Whether LAG-1 and LAG-7 features belong in the model |
| `bayesian_analysis/01` | Prior specification — document carefully, it is a subjective choice |
| `model_preparation/01` | Final leakage audit; target horizon confirmed |
| `model_preparation/02` | Final feature set per tier |
| `model_preparation/03` | CV fold structure; Tier 3 model strategy (Bayesian vs classical ML) |
| `model_preparation/04` | **Baseline numbers** — the bar every future model must beat |
