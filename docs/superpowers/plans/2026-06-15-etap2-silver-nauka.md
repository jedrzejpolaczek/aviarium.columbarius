# Etap 2 — Silver Layer: Plan Nauki

**Goal:** Zrozumieć jak Bronze → Silver: czyszczenie, normalizacja, łączenie danych i ekstrakcja cen.

**Architecture:** Bronze (surowe dane) → SilverTransforms (config-driven cleaning) → SilverCardJoin (merge MTGJson × Scryfall) → SilverPriceBuilder (ceny z JSON) → Silver DuckDB.

**Tech Stack:** pandas, duckdb, json, dataclasses

---

## Pliki do przeczytania (w tej kolejności)

| Plik | Co robi |
|------|---------|
| `src/data/cards/storage/silver/cleaning.py` | 8 czystych funkcji czyszczących DataFrame |
| `src/data/cards/storage/silver/transforms.py` | Orkiestruje cleaning steps per tabela |
| `src/data/cards/storage/silver/card_join.py` | Łączy MTGJson i Scryfall po scryfall_id |
| `src/data/cards/storage/silver/prices.py` | Ekstrahuje ceny z JSON, forward-fill |
| `src/data/cards/storage/silver/storage.py` | Orkiestracja całego pipeline Bronze → Silver |

---

### Zadanie 1: Koncepcja — dlaczego Silver?

- [X] **Krok 1: Odpowiedz na pytania Claude o Bronze → Silver**

  Claude zadaje pytania jedno po drugim. Ty odpowiadasz.

  Przykładowe pytania:
  - "Co tracisz jeśli wczytujesz dane bezpośrednio z Bronze do modelu ML?"
  - "Dlaczego Silver przechowuje ceny osobno od kart?"
  - "Czym różni się cleaning od normalizacji?"

---

### Zadanie 2: Analiza — `cleaning.py`

**Pliki:** `src/data/cards/storage/silver/cleaning.py`

- [X] **Krok 1: Przeczytaj plik**

  Zwróć uwagę na:
  - Każda funkcja przyjmuje `(df, ..., issues)` — co to jest `issues`?
  - Dlaczego `_filter_rows` resetuje index na końcu?
  - Co robi `_clean_strings` z wartością `"_"`?

- [X] **Krok 2: Odpowiedz na pytania Claude**

  Claude zadaje pytania Sokratejsko.

- [X] **Krok 3: Sprawdź zrozumienie**

  - Jaką wartość zwraca `_clean_numerics` gdy nie może sparsować liczby?
  - Co to `issues` accumulator — dlaczego nie rzucamy wyjątku?
  - Czym różni się `_clean_lists` od `_clean_booleans`?

---

### Zadanie 3: Analiza — `transforms.py`

**Pliki:** `src/data/cards/storage/silver/transforms.py`

- [X] **Krok 1: Przeczytaj plik**

  Zwróć uwagę na:
  - Dlaczego `SilverTransforms.__init__` przyjmuje `language_map` i `legality_map`?
  - Co robi `transform()` — ile kroków?
  - Co to `_parse_type_line` i jak działa (split na "—")?

- [X] **Krok 2: Odpowiedz na pytania Claude**

- [X] **Krok 3: Sprawdź zrozumienie**

  - Które metody są "stateless" a które "stateful"? Dlaczego ten podział?
  - Co `_normalize_values` robi z kodem języka `"en"`?
  - Co `_add_computed_columns` robi gdy `text != original_text`?

---

### Zadanie 4: Analiza — `card_join.py`

**Pliki:** `src/data/cards/storage/silver/card_join.py`

- [X] **Krok 1: Przeczytaj plik**

  To najtrudniejszy plik w Silver. Zwróć uwagę na:
  - Dlaczego OUTER JOIN a nie LEFT JOIN?
  - Co to `has_mtgjson_data` i po co?
  - Co to `canonical_uuid` i dlaczego potrzebujemy go dla kart non-English?

- [X] **Krok 2: Odpowiedz na pytania Claude**

- [X] **Krok 3: Sprawdź zrozumienie**

  - Co się stanie z kartą która jest w MTGJson ale nie ma scryfall_id?
  - Co to `_SCRYFALL_FALLBACK_MAP` i kiedy jest używany?
  - Dlaczego na końcu robimy `drop_duplicates(subset=["scryfall_id"])`?

---

### Zadanie 5: Analiza — `prices.py`

**Pliki:** `src/data/cards/storage/silver/prices.py`

- [X] **Krok 1: Przeczytaj plik**

  Zwróć uwagę na:
  - Co robi `build()` na wysokim poziomie (3 metody prywatne)?
  - Co to `forward-fill` i dlaczego go potrzebujemy dla cen?
  - Co robi SQL `QUALIFY ROW_NUMBER() OVER (PARTITION BY ...)`?

- [X] **Krok 2: Odpowiedz na pytania Claude**

- [X] **Krok 3: Sprawdź zrozumienie**

  - Dlaczego czytamy tylko `snapshot_date = today` zamiast całej historii?
  - Co to `_extract_all_prices` i dlaczego parsuje JSON tylko raz?
  - Czym różni się `silver_prices_history` od `silver_language_prices_history`?

---

### Zadanie 6: Analiza — `storage.py`

**Pliki:** `src/data/cards/storage/silver/storage.py`

- [X] **Krok 1: Przeczytaj plik**

  Zwróć uwagę na:
  - `SilverStorage` dziedziczy po `TransformStorage` — co z tego dostaje?
  - `_pipeline` — ile rzeczy robi? Czy to problem?
  - Dlaczego `silver_cards` zawsze robi `full_load` a nie `upsert`?

- [X] **Krok 2: Odpowiedz na pytania Claude**

---

### Zadanie 7: Samodzielna implementacja

**Stwórz:** `_TRAINING/training_silver.py`

Teraz zamknij pliki projektu i napisz uproszczoną wersję Silver pipeline.

- [X] **Krok 1: Przeczytaj spec i zaimplementuj**

  ```
  Napisz klasę MiniCleaner z metodami:

  1. filter_rows(df, rules: dict[str, object]) -> pd.DataFrame
     - Usuń wiersze gdzie kolumna == wartość
     - Obsłuż przypadek gdy kolumna nie istnieje (po prostu pomiń)

  2. clean_strings(df, ops: dict[str, list[str]]) -> pd.DataFrame
     - Obsługuje operacje: "strip", "lower", "upper"
     - Obsługuje przypadek gdy kolumna nie istnieje

  3. rename_columns(df, mapping: dict[str, str]) -> pd.DataFrame
     - Zmień nazwy kolumn według słownika
     - Pomiń kolumny których nie ma

  Napisz klasę MiniSilverStorage która:

  4. __init__(self, bronze_path: str, silver_path: str)
     - Otwiera dwa połączenia DuckDB: bronze (read_only=True) i silver

  5. process_table(self, bronze_table: str, silver_table: str, rules: dict) -> None
     - Wczytaj całą tabelę z bronze: con.execute("SELECT * FROM ...").df()
     - Zastosuj MiniCleaner.filter_rows i clean_strings i rename_columns
     - Zapisz do silver (użyj CREATE OR REPLACE TABLE ... AS SELECT * FROM staging)

  6. close(self) -> None

  Nie używaj Pydantic. Nie używaj config.json. Płaska, prosta implementacja.
  ```

- [X] **Krok 2: Przetestuj ręcznie**

  ```python
  import duckdb
  import pandas as pd

  # Przygotuj bronze w pamięci
  bronze = duckdb.connect(":memory:")
  bronze.execute("""
      CREATE TABLE bronze_cards AS SELECT * FROM (VALUES
          (1, 'Black Lotus ', 'Artifact', 'power9'),
          (2, 'Mox Ruby  ', 'Artifact', 'power9'),
          (3, 'Token Goblin', 'Token', 'basic')
      ) t(id, name, card_type, tag)
  """)

  silver = duckdb.connect(":memory:")

  cleaner = MiniCleaner()
  
  df = bronze.execute("SELECT * FROM bronze_cards").df()
  df = cleaner.filter_rows(df, {"card_type": "Token"})
  df = cleaner.clean_strings(df, {"name": ["strip", "lower"]})
  df = cleaner.rename_columns(df, {"card_type": "type"})
  
  silver.register("_staging", df)
  silver.execute("CREATE OR REPLACE TABLE silver_cards AS SELECT * FROM _staging")
  silver.unregister("_staging")

  print(silver.execute("SELECT * FROM silver_cards").df())
  # Powinno: 2 wiersze, name bez spacji i małymi literami, kolumna "type"

  bronze.close()
  silver.close()
  ```

- [X] **Krok 3: Porównaj z oryginałem**

  Otwórz `cleaning.py` i `transforms.py` i porównaj:
  - Co oryginał robi co ty pominąłeś? (podpowiedź: issues accumulator, więcej typów czyszczenia)
  - Co ty napisałeś czytelniej?
  - Dlaczego oryginał nie rzuca wyjątków gdy kolumna nie istnieje?

---

## Koniec Etapu 2

Gdy skończysz, zaktualizuj spec `docs/superpowers/specs/2026-06-11-learning-collaboration-design.md`:
```markdown
- [x] Etap 2: Silver — ukończony YYYY-MM-DD
```

Następna sesja: powiedz **"Kontynuujemy plan nauki, skończyłem Etap 2 Silver"**.
