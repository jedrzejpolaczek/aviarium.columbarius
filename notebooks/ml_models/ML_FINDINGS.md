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
**Status:** [ ] Odblokowany — próg 20 snapshotów minął (jest 36, stan na 2026-07-02), do uruchomienia.

UWAGA: Pierwotna kolejnosc zakladala zrobienie T8 przed T7 (T7 czekal na dane).
Ten warunek juz nie obowiazuje — oba notebooki mozna uruchomic.

Osobne ograniczenie, ktore *nadal* obowiazuje: walk-forward CV
(`walk_forward_cv_nb03`) wymaga >= 50 snapshotow i przy 36 konczy sie
`InsufficientDataError`. To nie blokuje T7, ale oznacza, ze wszystkie
dotychczasowe metryki pochodza z pojedynczego podzialu chronologicznego.

Kluczowe pytania:
- Prophet vs LightGBM: ktory model jest lepszy dla plynnych kart?
- Czy jest widoczna sezonowosc tygodniowa (FNM)?

_(wypelnij po uruchomieniu notebooka)_

---

## T8 — Optuna + SHAP

**Notebook:** `04_shap_optuna.ipynb`
**Status:** [~] Część Optuna uruchomiona (MLflow run `tuned_lightgbm_nb04`, 2026-07-10). Część SHAP — do uzupełnienia.

### Optuna — najlepsze parametry

Run z 2026-07-10 (`9c1ec7de65c7476aa75a199b7189d6f2` — ten sam, który jest obecnie wdrożony w `docker/.env`):

| parametr | wartość | zakres przeszukiwania |
|---|---:|---|
| `num_leaves` | 203 | 32–256 |
| `learning_rate` | 0.0308 | 0.01–0.3 (log) |
| `min_child_samples` | 37 | 20–200 |
| `subsample` | 0.7539 | 0.6–1.0 |
| `colsample_bytree` | 0.8 | stałe |
| `n_estimators` | 1000 | stałe (early stopping) |

Metryki tego runu: `train_mae = 0.0500`, `train_mape = 85.23%`.

**Uwaga o MAPE:** 85% nie jest alarmujące — `log_return_7d` jest bliskie zeru dla większości kart, więc mianownik w MAPE jest bardzo mały nawet po clipie `MAPE_CLIP_MIN = 0.01` (`src/ml/evaluation/metrics.py`). MAE pozostaje właściwą metryką do porównywania modeli; MAPE służy tylko jako niezależna od skali kontrola.

**Uwaga o porównywalności:** zalogowano tylko metryki *treningowe* (`train_mae`), nie testowe — tego runu nie da się bezpośrednio porównać z tabelą per-tier z T6, która raportuje MAE na zbiorze testowym. Przy kolejnym uruchomieniu warto zalogować `mae_test` per tier, żeby odpowiedzieć na pytanie, czy tuning faktycznie poprawił Tier 2 (jedyny tier, w którym LightGBM przegrywa z baseline).

**Porównanie z poprzednim tuningiem (2026-07-06, 32 snapshoty):** `train_mae = 0.0`, `train_mape = 0.0` — degeneratywny wynik z tego samego powodu co opisany w T6 (płaski target). Przy 36 snapshotach już nie występuje.

### SHAP — do uzupełnienia

Pytania bez odpowiedzi (wymagają uruchomienia części SHAP notebooka):
- Kolejnosc waznosci SHAP: czy edhrec_saltiness > is_reserved?
- SHAP print_count: czy spada do zera gdy saltiness jest w modelu? (weryfikacja BA-02)
- Waterfall dla 3 kart: taniej common, Reserved List, tournament staple
