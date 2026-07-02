# Etap 1 — Bronze Layer: Plan Nauki

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Zrozumieć architekturę bronze layer i samodzielnie zaimplementować uproszczoną wersję.

**Architecture:** Dane surowe (Scryfall API, MTGJson) → Pydantic modele → DuckDB tabele. Trzy wzorce zapisu: full replace, upsert, snapshot dzienny.

**Tech Stack:** duckdb, pydantic, pandas, dataclasses

---

## Pliki do przeczytania (w tej kolejności)

| Plik | Co robi |
|------|---------|
| `src/data/cards/storage/base.py` | Klasa bazowa: połączenie z DuckDB, serializacja |
| `src/data/cards/storage/bronze/config.py` | Deklaracja co i jak zapisywać |
| `src/data/cards/storage/bronze/writers.py` | Prymitywy zapisu do DuckDB |
| `src/data/cards/storage/bronze/storage.py` | Orkiestracja: łączy config + writers |

## Plik do napisania samodzielnie

| Plik | Co robi |
|------|---------|
| `notebooks/learning/bronze_mini.py` | Twoja uproszczona implementacja bronze |

---

### Zadanie 1: Koncepcja — dlaczego Bronze/Silver/Gold?

- [X] **Krok 1: Odpowiedz na pytania Claude o architekturę Bronze/Silver/Gold**

  Claude zadaje pytania jedno po drugim. Ty odpowiadasz. Jeśli utkniesz — Claude pomaga sformułować odpowiedź, nie podaje gotowej.

  Przykładowe pytania:
  - "Dlaczego nie trzymamy wszystkiego w jednej tabeli?"
  - "Co tracisz jeśli czyścisz dane przed zapisaniem ich do bazy?"
  - "Czym różni się Silver od Gold?"

- [X] **Krok 2: Odpowiedz na pytania Claude o DuckDB**

  Claude zadaje pytania. Ty odpowiadasz na podstawie swojej wiedzy o bazach danych.

  Przykładowe pytania:
  - "Czym różni się DuckDB od PostgreSQL?"
  - "Kiedy wolisz bazę plikową zamiast serwera?"
  - "Dlaczego DuckDB rozumie pandas DataFrame bezpośrednio?"

---

### Zadanie 2: Analiza — `base.py`

**Pliki:** `src/data/cards/storage/base.py`

- [X] **Krok 1: Otwórz plik i przeczytaj go samodzielnie**

  Zwróć uwagę na:
  - Co robi `_open_connection`?
  - Co robi `_serialize_objects` i dlaczego?
  - Co to jest klasa abstrakcyjna (`ABC`) i `@abstractmethod`?

- [X] **Krok 2: Zapytaj o niejasności**

  Powiedz Claude które linie są niejasne. Przykład:
  > "Nie rozumiem linii 66-69 w base.py — co to jest `select_dtypes` i dlaczego serializujemy do JSON?"

- [X] **Krok 3: Sprawdź zrozumienie — odpowiedz na pytania**

  Zanim pójdziesz dalej, odpowiedz (w głowie lub na papierze):
  - Dlaczego `close()` jest abstrakcyjna?
  - Co się stanie jeśli wpiszemy do DuckDB DataFrame gdzie kolumna ma mix `dict` i `None`?
  - Po co `__enter__` / `__exit__`?

---

### Zadanie 3: Analiza — `config.py`

**Pliki:** `src/data/cards/storage/bronze/config.py`

- [X] **Krok 1: Przeczytaj plik**

  Zwróć uwagę na:
  - `STORAGE_CONFIG` — co to jest za struktura danych?
  - Dla każdego źródła: czy ma tabelę główną? Czy ma snapshoty?
  - Co to `incremental=True`?

- [X] **Krok 2: Zapytaj o niejasności**

- [X] **Krok 3: Sprawdź zrozumienie**

  - Jakie źródła danych mamy? (wymień 5)
  - Które źródło NIE ma tabeli głównej i dlaczego?
  - Czym różni się `snapshot` od tabeli głównej?

---

### Zadanie 4: Analiza — `writers.py`

**Pliki:** `src/data/cards/storage/bronze/writers.py`

- [X] **Krok 1: Przeczytaj plik, skup się na 3 metodach**

  Trzy wzorce zapisu:
  - `_full_load_table` — drop and recreate (pełne zastąpienie)
  - `_incremental_load` — delete-then-insert (upsert)
  - `_snapshot` — append-only z deduplikacją po (klucz, data)

- [X] **Krok 2: Narysuj diagram przepływu danych dla każdej metody**

  Dla `_full_load_table`:
  ```
  records (lista Pydantic) → DataFrame → DuckDB staging → CREATE OR REPLACE TABLE
  ```
  Narysuj analogiczne dla `_incremental_load` i `_snapshot`.

- [X] **Krok 3: Zapytaj o niejasności**

  Szczególnie: co to `self._con.register("_save_staging", df)`? Po co rejestrujemy DataFrame?

---

### Zadanie 5: Analiza — `storage.py`

**Pliki:** `src/data/cards/storage/bronze/storage.py`

- [X] **Krok 1: Przeczytaj plik**

  Zwróć uwagę na:
  - `BronzeStorage` dziedziczy po `BronzeWritersMixin` i `BaseStorage` — po co dwie klasy bazowe?
  - `_process_sources` — jak iteruje po `STORAGE_CONFIG`?
  - Różnica między `populate` a `daily_update`

- [X] **Krok 2: Prześledź przepływ dla jednego źródła**

  Mentalnie wykonaj ten kod dla `source_type = "scryfall"`:
  ```python
  storage.daily_update({"scryfall": ([card1, card2], [])})
  ```
  Co się wywoła krok po kroku?

---

### Zadanie 6: Samodzielna implementacja

**Pliki:**
- Stwórz: `notebooks/learning/bronze_mini.py`

Teraz zamknij wszystkie pliki z projektu. Napisz uproszczoną wersję bronze layer BEZ zaglądania do oryginału.

- [X] **Krok 1: Przeczytaj spec i zaimplementuj**

  Spec:

  ```
  Napisz klasę MiniStorage która:

  1. __init__(self, db_path: str) — otwiera połączenie duckdb.connect(db_path)

  2. full_load(self, data: list[dict], table_name: str) -> None
     - Konwertuje listę słowników na pd.DataFrame
     - Zapisuje do DuckDB jako nową tabelę (zastępuje jeśli istnieje)
     - Użyj: con.register("staging", df), potem SQL: CREATE OR REPLACE TABLE ...

  3. upsert(self, data: list[dict], table_name: str, key: str) -> None
     - Jeśli tabela nie istnieje: utwórz ją
     - Jeśli istnieje: usuń wiersze gdzie key jest w nowych danych, wstaw nowe
     - Sprawdź czy tabela istnieje: SELECT count(*) FROM information_schema.tables ...

  4. snapshot(self, data: list[dict], table_name: str, key: str) -> None
     - Dodaj kolumnę "date" z dzisiejszą datą (datetime.date.today().isoformat())
     - Wstaw tylko wiersze których (key, date) jeszcze nie ma w tabeli
     - Utwórz tabelę jeśli nie istnieje

  5. close(self) — zamyka połączenie

  Nie używaj Pydantic. Nie używaj mixinów. Prosta, płaska klasa.
  ```

- [X] **Krok 2: Przetestuj ręcznie w Pythonie**

  ```python
  storage = MiniStorage(":memory:")

  # Test full_load
  storage.full_load([{"id": 1, "name": "Black Lotus"}, {"id": 2, "name": "Mox Ruby"}], "cards")
  print(storage.con.execute("SELECT * FROM cards").df())

  # Test upsert
  storage.upsert([{"id": 1, "name": "Black Lotus UPDATED"}], "cards", key="id")
  print(storage.con.execute("SELECT * FROM cards").df())  # powinno pokazać updated + Mox Ruby

  # Test snapshot
  storage.snapshot([{"id": 1, "price": 10000}], "prices_history", key="id")
  storage.snapshot([{"id": 1, "price": 10000}], "prices_history", key="id")  # drugi raz — nie powinno zduplikować
  print(storage.con.execute("SELECT * FROM prices_history").df())  # powinien być 1 wiersz

  storage.close()
  ```

- [X] **Krok 3: Porównaj z oryginałem**

  Otwórz `src/data/cards/storage/bronze/writers.py` i porównaj:
  - Co masz podobnie?
  - Co masz inaczej?
  - Co oryginał robi lepiej? (podpowiedź: obsługa błędów, logowanie, Pydantic)
  - Co ty napisałeś czytelniej?

---

## Koniec Etapu 1

Gdy skończysz wszystkie zadania, zaktualizuj spec w `docs/superpowers/specs/2026-06-11-learning-collaboration-design.md`:

```markdown
## Postęp

- [x] Etap 1: Bronze — ukończony YYYY-MM-DD
- [X] Etap 2: Silver - docs\superpowers\plans\2026-06-15-etap2-silver-nauka.md
...
```

Następna sesja: powiedz *"Kontynuujemy plan nauki, skończyłem Etap 1 Bronze"*.
