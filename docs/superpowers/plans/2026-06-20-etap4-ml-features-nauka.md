# Etap 4 — ML Features: Plan Nauki

**Goal:** Zrozumieć jak Gold DuckDB → model ML: przygotowanie cech, transformacje, obsługa NaN i target.

**Architecture:** Gold DuckDB (gold_price_features, gold_card_features) → `lag.py` (window features + target) → `pipeline.py` (sklearn ColumnTransformer, imputacja) → X, y gotowe do trenowania LightGBM.

**Tech Stack:** duckdb, pandas, numpy, sklearn (Pipeline, ColumnTransformer, SimpleImputer)

---

## Pliki do przeczytania (w tej kolejności)

| Plik | Co robi |
|------|---------|
| `src/ml/features/lag.py` | Buduje cechy czasowe (LAG, rolling) i target z DuckDB |
| `src/ml/features/pipeline.py` | sklearn Pipeline: imputacja, passthrough, log-transform |

---

### Zadanie 1: Koncepcja — dlaczego transformujemy cechy?

- [X] **Krok 1: Odpowiedz na pytania Claude o przygotowanie cech ML**

  Claude zadaje pytania jedno po drugim. Ty odpowiadasz.

  Przykładowe pytania:
  - "Dlaczego Gold nie nadaje się bezpośrednio jako X do modelu?"
  - "Co to jest data leakage i dlaczego niszczy model?"
  - "Czym różni się model liniowy od LightGBM jeśli chodzi o skalowanie cech?"

---

### Zadanie 2: Analiza — `lag.py`

**Pliki:** `src/ml/features/lag.py`

- [X] **Krok 1: Przeczytaj plik**

  Zwróć uwagę na:
  - `build_lag_features` — z jakiej tabeli czyta? Na czym różni się od Gold `build_price_features`?
  - `build_target` — co to `log_return_7d` i dlaczego liczymy log zamiast surowej różnicy?
  - Dlaczego okno w LAG jest `UNBOUNDED` a nie `ROWS BETWEEN N PRECEDING`?

- [X] **Krok 2: Odpowiedz na pytania Claude**

- [X] **Krok 3: Sprawdź zrozumienie**

  - Co zwróci `build_lag_features` dla karty która ma tylko 3 dni historii cenowej?
  - Dlaczego `build_target` używa INNER JOIN zamiast LEFT JOIN?
  - Co to `momentum_7d` i jak jest liczone? Kiedy jest NaN?

---

### Zadanie 3: Analiza — `pipeline.py`

**Pliki:** `src/ml/features/pipeline.py`

- [X] **Krok 1: Przeczytaj plik**

  Zwróć uwagę na:
  - `LEAKAGE_COLS` — co to są kolumny które "wyciekają"? Skąd wiadomo że te konkretne?
  - `IMPUTE_MEDIAN_COLS` vs `IMPUTE_ZERO_COLS` — dlaczego jedne medianą a drugie zerem?
  - `_enrich_card_df` i `_enrich_lag_df` — dlaczego są osobnymi prywatnymi funkcjami?
  - `remainder='drop'` w ColumnTransformer — co to robi?

- [X] **Krok 2: Odpowiedz na pytania Claude**

- [X] **Krok 3: Sprawdź zrozumienie**

  - Co to "training/serving skew" i dlaczego `_enrich_*` zapobiega temu problemowi?
  - Co to `MNAR` (Missing Not At Random) i która kolumna jest MNAR w tym pliku?
  - Dlaczego `log1p` a nie `log`?

---

### Zadanie 4: Samodzielna implementacja

**Stwórz:** `_TRAINING/training_ml_features.py`

- [X] **Krok 1: Przeczytaj spec i zaimplementuj**

  ```
  Napisz klasę MiniFeaturePipeline z metodą:

  1. build(conn, snapshot_date: str) -> tuple[pd.DataFrame, pd.Series]

     Krok A — cechy z DuckDB (wszystko w jednym SQL):
       - Wczytaj z gold_price_features dla snapshot_date i wszystkich poprzednich dat tej karty:
           uuid, snapshot_date, eur
           LAG(eur, 1)  OVER (PARTITION BY uuid ORDER BY snapshot_date) AS lag_1d
           LAG(eur, 7)  OVER (PARTITION BY uuid ORDER BY snapshot_date) AS lag_7d
       - Przefiltruj do snapshot_date

     Krok B — target z DuckDB (osobne zapytanie):
       - Znajdź ceny za 7 dni do przodu dla każdego uuid
       - Oblicz: log_return_7d = log1p(eur_t7) - log1p(eur_t0)

     Krok C — transformacje w pandas:
       - Oblicz log_eur = log1p(eur)
       - Wypełnij NaN w lag_1d i lag_7d medianą kolumny

     Krok D — join i zwróć:
       - Inner join cech z targetem po uuid
       - X = [log_eur, lag_1d, lag_7d]
       - y = log_return_7d
       - Zwróć (X, y)

  Nie używaj sklearn. Tylko duckdb, pandas, numpy.
  ```

- [X] **Krok 2: Przetestuj ręcznie**

  ```python
  import duckdb, numpy as np

  gold = duckdb.connect(":memory:")
  gold.execute("""
      CREATE TABLE gold_price_features AS SELECT * FROM (VALUES
          ('A', '2026-06-01', 10.0),
          ('A', '2026-06-02', 12.0),
          ('A', '2026-06-08', 15.0),
          ('A', '2026-06-09', 14.0),
          ('B', '2026-06-01', 5.0),
          ('B', '2026-06-08', 6.0),
          ('B', '2026-06-09', 7.0)
      ) t(uuid, snapshot_date, eur)
  """)

  pipeline = MiniFeaturePipeline()
  X, y = pipeline.build(gold, '2026-06-02')
  # Tylko karta A ma cenę za 7 dni (2026-06-09 = 14.0)
  # X: log_eur=log1p(12), lag_1d=log1p(10) (po imputacji), lag_7d=NaN→median
  # y: log1p(14) - log1p(12)

  gold.close()
  ```

- [X] **Krok 3: Porównaj z oryginałem**

  Otwórz `lag.py` i `pipeline.py` i porównaj:
  - Co oryginał robi co ty pominąłeś? (sklearn Pipeline, LEAKAGE_COLS, więcej cech)
  - Dlaczego oryginał używa `sklearn.Pipeline` zamiast ręcznych transformacji?
  - Co to `get_feature_names` i po co? (podpowiedź: SHAP)

---

## Koniec Etapu 4

Gdy skończysz, zaktualizuj spec `docs/superpowers/specs/2026-06-11-learning-collaboration-design.md`:
```markdown
- [x] Etap 4: ML Features — ukończony YYYY-MM-DD
```

Następna sesja: powiedz **"Kontynuujemy plan nauki, skończyłem Etap 4 ML Features"**.
