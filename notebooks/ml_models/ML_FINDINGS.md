# ML Models — Findings

Wyniki i obserwacje z notebookow ML (Miesiac 2).
Wypelnij kazda sekcje po uruchomieniu odpowiedniego notebooka.

---

## T5 — Feature Engineering

**Notebook:** `01_feature_engineering.ipynb`
**Status:** [ ] Do uruchomienia

Kluczowe pytania do odpowiedzi:
- Ile kart ma kompletne lag features (nie-NaN dla lag_7d)?
- Jaki jest rozklad momentum_7d — czy symetryczny wokol 0?
- Czy rolling_std_14d dobrze rozroznia stabilne karty od spekulatywnych?

_(wypelnij po uruchomieniu notebooka)_

---

## T6 — Baseline vs LightGBM

**Notebook:** `02_baseline_lightgbm.ipynb`
**Status:** [X] Uruchomiony ponownie (36 snapshotów, SNAPSHOT_DATE = 2026-07-02). Dataset: 78 692 wierszy × 45 kolumn (17 cech użytych przez pipeline, 14 694 wiersze odrzucone z powodu NaN w targecie).

### Wynik — już nie degeneratywny, LightGBM wygrywa w 2/3 tierów

| model | tier | mae |
|---|---|---:|
| Naive | 1 | 0.056128 |
| MA7d | 1 | 0.056128 |
| LightGBM | 1 | **0.054001** |
| Naive | 2 | **0.031356** |
| MA7d | 2 | 0.031356 |
| LightGBM | 2 | 0.033488 |
| Naive | 3 | 0.053028 |
| MA7d | 3 | 0.053028 |
| LightGBM | 3 | **0.052530** |

AR1 (overall, nie per-tier w tym notebooku): 0.0569 — najsłabszy baseline.

**Does LightGBM beat Naive?** Tier 1: **Tak** (0.054001 < 0.056128) | Tier 2: **Nie** (0.033488 > 0.031356, ~7% gorzej) | Tier 3: **Tak, marginalnie** (0.052530 < 0.053028).

To już nie jest degeneratywny remis MAE=0 opisany poniżej (32 snapshoty, SNAPSHOT_DATE = 2026-06-22) — przy 36 snapshotach wszystkie modele raportują niezerowe, nie-degeneratywne MAE/MAPE per tier. Tier 2 (111 kart testowych) to najtrudniejszy tier do pobicia — Naive/MA7d siedzą już na MAE ≈ 0.031 (najniższe ze wszystkich tierów) i LightGBM tego nie dogonił. Tier 1 (15 606 kart) i Tier 3 (22 karty) — LightGBM wygrywa, ale nie jest to jednolite zwycięstwo we wszystkich tierach, warto zweryfikować ponownie gdy przybędzie więcej snapshotów (zwłaszcza Tier 3 przez małą próbkę).

### Poprzedni wynik (32 snapshoty, 2026-06-22) — zachowane jako kontekst historyczny

Przy 32 snapshotach i pierwszym uruchomieniu, `log_return_7d` był idealnie płaski (~84% kart identycznych po 7 dniach) — Naive, MA7d i LightGBM wszystkie osiągały MAE ≈ 0 dla każdego tieru, co nie było zwycięstwem LightGBM tylko degeneratywnym porównaniem (LightGBM's `No further splits with positive gain` na każdej rundzie). Root cause zweryfikowany bezpośrednio na `data/gold/cards.duckdb`: ceny Tier 1 (<€100) były wtedy płaskie w niemal 100% przypadków w oknie 7-dniowym, zgodnie z zamkniętą już analizą `2026-07-06-price-feed-anomalies.md`. Przy 36 snapshotach ten efekt już nie dominuje — patrz wynik powyżej.

### Implication for future work

- **Headline MAE per tier jest teraz wiarygodny** (nie jest już zdominowany przez zerowy target) — ale nadal warto raportować obok niego frakcję wierszy z `log_return_7d != 0` per tier, żeby śledzić kiedy/czy efekt degeneracji wraca.
- Tier 2 pozostaje najtrudniejszy — LightGBM przegrywa z najprostszymi baseline'ami mimo dostępu do 17 cech; warto sprawdzić czy to przeuczenie (zbyt mało przykładów w Tier 2 relative do liczby cech) czy faktyczny brak sygnału.
- AR1 pozostaje najsłabszym baseline'em nawet przy większej ilości danych historycznych (0.0569 vs Naive 0.0561-0.0566) — sugeruje że `lag_1d`-owy return term dodaje szum, nie sygnał, na poziomie globalnym.

---

## T7 — Time Series (data-gated)

**Notebook:** `03_time_series.ipynb`
**Status:** [ ] Data-gated (wymaga >= 20 snapshotow ~2026-07-03)

UWAGA: Zamien kolejnosc z T8 — zrob T8 najpierw na dostepnych danych,
wróc do T7 gdy beda >= 20 snapshoty.

Kluczowe pytania:
- Prophet vs LightGBM: ktory model jest lepszy dla plynnych kart?
- Czy jest widoczna sezonowosc tygodniowa (FNM)?

_(wypelnij po uruchomieniu notebooka)_

---

## T8 — Optuna + SHAP

**Notebook:** `04_shap_optuna.ipynb`
**Status:** [ ] Do uruchomienia (nie wymaga pelnego CV, mozna na crosssectional)

Kluczowe pytania:
- Najlepsze parametry z Optuna (num_leaves, learning_rate, min_child_samples)?
- Kolejnosc waznosci SHAP: czy edhrec_saltiness > is_reserved?
- SHAP print_count: czy spada do zera gdy saltiness jest w modelu? (weryfikacja BA-02)
- Waterfall dla 3 kart: taniej common, Reserved List, tournament staple

_(wypelnij po uruchomieniu notebooka)_
