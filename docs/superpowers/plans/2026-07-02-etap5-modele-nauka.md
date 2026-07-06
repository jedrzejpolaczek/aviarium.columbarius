# Etap 5 — Modele: Plan Nauki

**Goal:** Zrozumieć jak X, y → model ML: baseline'y, LightGBM, tiered routing.

**Architecture:** X, y (z pipeline.py) → `baseline.py` (naiwne predyktory) → `lightgbm_model.py` (gradient boosting) → `tiered.py` (routing per przedział cenowy) → predykcja `log_return_7d`.

**Tech Stack:** lightgbm, sklearn (LinearRegression), numpy, pandas.

---

## Pliki do przeczytania (w tej kolejności)

| Plik | Co robi |
|------|---------|
| `src/ml/models/baseline.py` | 4 naiwne modele — punkt odniesienia dla LightGBM |
| `src/ml/models/lightgbm_model.py` | Właściwy model: gradient boosted trees z early stopping |
| `src/ml/models/tiered.py` | Router: różne modele dla różnych przedziałów cenowych |

---

### Zadanie 1: Koncepcja — po co baseline i czym jest LightGBM?

- [X] **Krok 1: Odpowiedz na pytania Claude**

  Claude zadaje pytania jedno po jednym. Ty odpowiadasz.

  Przykładowe pytania:
  - "Po co w ogóle trenować NaiveForecast który zawsze zwraca 0?"
  - "Czym różni się drzewo decyzyjne od sieci neuronowej?"
  - "Co to gradient boosting — co jest 'boostowane'?"

---

### Zadanie 2: Analiza — `baseline.py`

**Pliki:** `src/ml/models/baseline.py`

- [X] **Krok 1: Przeczytaj plik**

  Zwróć uwagę na:
  - `NaiveForecast` — co zwraca `predict`? Dlaczego `fit` nic nie robi?
  - `MeanForecast` — co przechowuje po `fit`?
  - `MovingAverageForecast` — skąd bierze `rolling_mean_7d`?
  - `AR1Forecast` — jedyny który używa `LinearRegression` — dlaczego?

- [X] **Krok 2: Odpowiedz na pytania Claude**

- [X] **Krok 3: Sprawdź zrozumienie**

  - Jeśli LightGBM przegrywa z `NaiveForecast` — co to znaczy o modelu?
  - Jeśli LightGBM przegrywa z `MeanForecast` — co to znaczy?
  - Co testuje `MovingAverageForecast` — jaką hipotezę o rynku MTG?

---

### Zadanie 3: Analiza — `lightgbm_model.py`

**Pliki:** `src/ml/models/lightgbm_model.py`

- [X] **Krok 1: Przeczytaj plik**

  Zwróć uwagę na:
  - `LightGBMParams` — co to `num_leaves`, `learning_rate`, `subsample`?
  - `objective='mae'` — dlaczego MAE a nie MSE? (wskazówka: Pareto α=1.303)
  - `early_stopping(stopping_rounds=50)` — co zatrzymuje trening?
  - `best_iteration` w `predict` — dlaczego nie ostatnia iteracja?

- [X] **Krok 2: Odpowiedz na pytania Claude**

- [X] **Krok 3: Sprawdź zrozumienie**

  - Co to `lgb.Dataset` i czym różni się od pandas DataFrame?
  - Dlaczego `fit` ma osobny `X_val` / `y_val` zamiast robić split wewnętrznie?
  - Co robi `feature_importance(importance_type='gain')`?

---

### Zadanie 4: Analiza — `tiered.py`

**Pliki:** `src/ml/models/tiered.py`

- [X] **Krok 1: Przeczytaj plik**

  Zwróć uwagę na:
  - `assign_tier` — jakie są granice tierów i dlaczego akurat <100 / 100-1000 / >1000?
  - `TieredRouter.fit` — ile modeli LightGBM trenuje?
  - `MIN_TIER2_ROWS = 50` — co się dzieje gdy Tier 2 ma za mało danych?
  - Tier 3 — dlaczego brak modelu ML? Co zwraca `predict` dla Tier 3?

- [X] **Krok 2: Odpowiedz na pytania Claude**

- [X] **Krok 3: Sprawdź zrozumienie**

  - Dlaczego jeden globalny LightGBM na wszystkich kartach byłby gorszy?
  - Co to `setattr(self, attr, model)` i dlaczego zamiast `if tier == 1: ...`?
  - Jeśli karta kosztuje dziś €50 ale model przewiduje €150 — który tier użyty do predykcji?

---

### Zadanie 5: Samodzielna implementacja

**Stwórz:** `_ADDONS/_TRAINING/training_models.py`

- [X] **Krok 1: Przeczytaj spec i zaimplementuj**

  ```
  Napisz trzy klasy:

  1. MiniNaiveForecast
     - fit(X, y) → self  (nic nie robi)
     - predict(X) → np.ndarray samych zer (len(X),)

  2. MiniMeanForecast
     - fit(X, y) → self  (zapisz mean y jako self.mean_)
     - predict(X) → np.ndarray wypełniony self.mean_ (len(X),)

  3. MiniTieredPredict (uproszczony TieredRouter)
     - Nie trenuj LightGBM — użyj MiniMeanForecast osobno dla każdego tieru
     - fit(df, target_col='log_return_7d'):
         - przypisz tier per wiersz: <100→1, 100-1000→2, >1000→3
         - trenuj MiniMeanForecast na każdym tierze osobno
     - predict(df) → pd.Series
         - dla Tier 3: NaN
         - dla Tier 1 i 2: wynik MiniMeanForecast odpowiedniego tieru

  Nie używaj LightGBM. Tylko numpy, pandas.
  ```

- [X] **Krok 2: Przetestuj ręcznie**

  ```python
  import pandas as pd
  import numpy as np

  df_train = pd.DataFrame({
      'eur': [5.0, 50.0, 200.0, 500.0, 2000.0],
      'log_return_7d': [0.01, 0.05, -0.02, 0.10, 0.03]
  })

  router = MiniTieredPredict()
  router.fit(df_train)
  preds = router.predict(df_train)

  # Tier 3 (2000 EUR) → NaN
  # Tier 1 (5, 50 EUR) → mean([0.01, 0.05]) = 0.03
  # Tier 2 (200, 500 EUR) → mean([-0.02, 0.10]) = 0.04
  print(preds)
  ```

- [X] **Krok 3: Porównaj z oryginałem**

  - Co `TieredRouter` robi co `MiniTieredPredict` pomija? (LightGBM, walidacja)
  - Dlaczego oryginał ma `MIN_TIER2_ROWS = 50`? Co by się stało bez tego guardu?
  - Co to `early_stopping` i dlaczego `MiniMeanForecast` tego nie potrzebuje?

---

## Koniec Etapu 5

Gdy skończysz, zaktualizuj spec `docs/superpowers/specs/2026-06-11-learning-collaboration-design.md`:
```markdown
- [x] Etap 5: Modele — ukończony YYYY-MM-DD
```

Następna sesja: powiedz **"Kontynuujemy plan nauki, skończyłem Etap 5 Modele"**.
