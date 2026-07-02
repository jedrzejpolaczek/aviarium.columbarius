# Etap 3 — Gold Layer: Plan Nauki

**Goal:** Zrozumieć jak Silver → Gold: feature engineering przez SQL window functions i budowanie datasetu ML.

**Architecture:** Silver (clean data) → GoldFeatureBuilders (SQL window functions) → GoldSignalBuilders (eventy, sygnały popytu) → GoldMLDatasetBuilder (JOIN wszystkiego) → Gold DuckDB.

**Tech Stack:** duckdb, pandas, SQL window functions

---

## Pliki do przeczytania (w tej kolejności)

| Plik | Co robi |
|------|---------|
| `src/data/cards/storage/gold/features.py` | Card features + price window features |
| `src/data/cards/storage/gold/signals.py` | Demand signals, ban events, turnieje |
| `src/data/cards/storage/gold/ml_dataset.py` | JOIN wszystkich tabel Gold w jeden dataset ML |
| `src/data/cards/storage/gold/storage.py` | Orkiestracja Silver → Gold |

---

### Zadanie 1: Koncepcja — dlaczego Gold?

- [X] **Krok 1: Odpowiedz na pytania Claude o Silver → Gold**

  Claude zadaje pytania jedno po drugim. Ty odpowiadasz.

  Przykładowe pytania:
  - "Dlaczego Silver nie nadaje się bezpośrednio jako input do modelu ML?"
  - "Co to jest 'feature engineering' i po co?"
  - "Dlaczego Gold zawsze robi full_load zamiast upsert?"

---

### Zadanie 2: Analiza — `features.py`

**Pliki:** `src/data/cards/storage/gold/features.py`

To najtrudniejszy plik w Gold — zawiera SQL window functions.

- [X] **Krok 1: Przeczytaj plik**

  Zwróć uwagę na:
  - `build_card_features` — jakie cechy wyprowadza ze statycznych danych karty?
  - `build_price_features` — co to jest `LAG()`, `AVG() OVER`, `STDDEV() OVER`?
  - Co to `WINDOW w7 AS (PARTITION BY uuid ORDER BY snapshot_date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW)`?

- [X] **Krok 2: Odpowiedz na pytania Claude**

  Claude tłumaczy window functions krok po kroku przez pytania.

- [X] **Krok 3: Sprawdź zrozumienie**

  - Co robi `LAG(eur, 7)` dla wiersza z datą 2026-06-15?
  - Czym różni się `PARTITION BY uuid` od `GROUP BY uuid`?
  - Co to `price_change_7d_pct` i jak jest liczone?
  - Dlaczego używamy `NULLIF(lag_1d, 0)` zamiast `/lag_1d`?

---

### Zadanie 3: Analiza — `signals.py`

**Pliki:** `src/data/cards/storage/gold/signals.py`

- [X] **Krok 1: Przeczytaj plik**

  Zwróć uwagę na:
  - Co to `_load_and_parse_meta` — dlaczego jest osobną metodą?
  - Co to `_has_legality_transitions` i po co jest wywoływane przed budowaniem eventów?
  - Co to `build_ban_price_impact` — jakie okno cenowe liczy?

- [X] **Krok 2: Odpowiedz na pytania Claude**

- [X] **Krok 3: Sprawdź zrozumienie**

  - Co to "legality transition" i kiedy się pojawia?
  - Dlaczego `gold_events` jest potrzebne jako osobna tabela zamiast obliczać eventy on-the-fly?
  - Co to `price_30d_before` vs `price_30d_after` w `build_ban_price_impact`?

---

### Zadanie 4: Analiza — `ml_dataset.py`

**Pliki:** `src/data/cards/storage/gold/ml_dataset.py`

- [X] **Krok 1: Przeczytaj plik**

  Zwróć uwagę na:
  - Co to "spine" w kontekście joinowania tabel?
  - Dlaczego wszystkie joiny są LEFT JOIN a nie INNER JOIN?
  - Co to `target_price_7d` — skąd pochodzi?

- [X] **Krok 2: Odpowiedz na pytania Claude**

- [X] **Krok 3: Sprawdź zrozumienie**

  - Co to "forward price target" i dlaczego to jest nasz `y` w modelu ML?
  - Dlaczego `gold_tournament_signals` nie ma `snapshot_date` w JOIN?
  - Co stanie się z wierszem jeśli `gold_demand_signals` nie ma pary?

---

### Zadanie 5: Analiza — `storage.py`

**Pliki:** `src/data/cards/storage/gold/storage.py`

- [X] **Krok 1: Przeczytaj plik**

  Zwróć uwagę na:
  - Dlaczego `populate()` i `update()` robią to samo?
  - Co to `_MIN_ML_HORIZON_DAYS = 7` i dlaczego pipeline sprawdza ten warunek?

- [X] **Krok 2: Odpowiedz na pytania Claude**

---

### Zadanie 6: Analiza — `writers.py`

**Pliki:** `src/data/cards/storage/gold/writers.py`

- [X] **Krok 1: Przeczytaj plik**

  Zwróć uwagę na:
  - Dlaczego ten plik tak wygląda?

- [X] **Krok 2: Odpowiedz na pytania Claude**

---

### Zadanie 7: Samodzielna implementacja

**Stwórz:** `_TRAINING/training_gold.py`

- [X] **Krok 1: Przeczytaj spec i zaimplementuj**

  ```
  Napisz klasę MiniGoldFeatures która:

  1. __init__(self, silver_con) — przechowuje połączenie do Silver DuckDB

  2. build_price_lags(self) -> pd.DataFrame
     - Wczytaj silver_prices_history (tylko kolumny: uuid, snapshot_date, eur)
     - Użyj DuckDB SQL z window function:
       LAG(eur, 1) OVER (PARTITION BY uuid ORDER BY snapshot_date) AS lag_1d
       LAG(eur, 7) OVER (PARTITION BY uuid ORDER BY snapshot_date) AS lag_7d
     - Zwróć DataFrame z kolumnami: uuid, snapshot_date, eur, lag_1d, lag_7d

  3. build_rolling_avg(self) -> pd.DataFrame
     - Wczytaj silver_prices_history (uuid, snapshot_date, eur)
     - Użyj window function:
       AVG(eur) OVER (PARTITION BY uuid ORDER BY snapshot_date
                      ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS avg_7d
     - Zwróć DataFrame z: uuid, snapshot_date, eur, avg_7d

  Nie używaj pandas do liczenia lagów ani średnich — wszystko w SQL.
  ```

- [X] **Krok 2: Przetestuj ręcznie**

  ```python
  silver = duckdb.connect(":memory:")
  silver.execute("""
      CREATE TABLE silver_prices_history AS SELECT * FROM (VALUES
          ('card-A', '2026-06-01', 10.0),
          ('card-A', '2026-06-02', 12.0),
          ('card-A', '2026-06-03', 11.0),
          ('card-A', '2026-06-08', 15.0),
          ('card-B', '2026-06-01', 5.0),
          ('card-B', '2026-06-02', 6.0)
      ) t(uuid, snapshot_date, eur)
  """)

  features = MiniGoldFeatures(silver)
  print(features.build_price_lags())
  # card-A 2026-06-01: lag_1d=NULL, lag_7d=NULL
  # card-A 2026-06-02: lag_1d=10.0, lag_7d=NULL
  # card-A 2026-06-08: lag_1d=NULL (brak 2026-06-07), lag_7d=10.0

  print(features.build_rolling_avg())
  # card-A 2026-06-03: avg_7d = (10+12+11)/3 = 11.0

  silver.close()
  ```

- [ ] **Krok 3: Porównaj z oryginałem**

  Otwórz `features.py` i porównaj:
  - Co oryginał liczy więcej? (price_change, volatility, rank, is_price_spike)
  - Dlaczego oryginał używa CTE (`WITH price_lags AS (...)`) zamiast subquery?
  - Co byś dodał do swojej implementacji?

---

## Koniec Etapu 3

Gdy skończysz, zaktualizuj spec `docs/superpowers/specs/2026-06-11-learning-collaboration-design.md`:
```markdown
- [x] Etap 3: Gold — ukończony YYYY-MM-DD
```

Następna sesja: powiedz **"Kontynuujemy plan nauki, skończyłem Etap 3 Gold"**.
