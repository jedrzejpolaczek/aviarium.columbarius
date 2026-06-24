# Scalar Bronze Price Tables Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace JSON blob price columns in both Bronze price history tables with scalar FLOAT columns, eliminating Python JSON parsing from Silver.

**Architecture:** A one-off migration script reads from `bronze/cards_copy.duckdb` and atomically replaces both price history tables in the live `bronze/cards.duckdb`. Bronze ingestion gains two dedicated snapshot methods that extract scalar prices at write time. Silver price queries simplify to direct column access via SQL files.

**Tech Stack:** DuckDB, pandas, Python, pytest

---

## File Map

| Action | Path |
|--------|------|
| Modify | `src/data/cards/storage/bronze/storage.py` |
| Modify | `src/data/cards/storage/bronze/config.py` |
| Modify | `src/data/cards/storage/silver/prices.py` |
| Create | `src/data/cards/storage/silver/sql/scryfall_prices_base.sql` |
| Create | `src/data/cards/storage/silver/sql/mtgjson_prices_daily.sql` |
| Create | `src/data/cards/storage/silver/sql/scryfall_language_prices_base.sql` |
| Modify | `src/data/cards/storage/health.py` |
| Create | `scripts/migrate_bronze_prices.py` |
| Modify | `tests/data/cards/storage/test_bronze.py` |
| Modify | `tests/data/cards/storage/test_silver.py` |
| Modify | `tests/data/cards/storage/test_health.py` |
| Modify | `tests/cards/test_storage.py` |
| Create | `tests/scripts/test_migrate_bronze_prices.py` |
| Modify | `docs/adr/ADR-003-medallion-architecture.md` |
| Modify | `docs/adr/ADR-012-physical-cards-only.md` |
| Create | `docs/adr/ADR-025-scalar-bronze-prices.md` |
| Modify | `docs/architecture/c4/bronze-storage.md` |
| Modify | `docs/architecture/c4/silver-storage.md` |
| Modify | `docs/architecture/data/table-schemas.md` |
| Modify | `docs/architecture/data/data-lineage.md` |
| Modify | `docs/architecture/data/glossary.md` |

---

## Task 1: Bronze — `_MTGJSON_PRICE_MAP` and `_extract_mtgjson_scalar_prices`

**Files:**
- Modify: `src/data/cards/storage/bronze/storage.py`
- Modify: `tests/data/cards/storage/test_bronze.py`

### Background

Silver's `SilverPriceBuilder._MTGJSON_PRICE_MAP` maps column names to `(retailer, tx_type, finish)` tuples. After this refactor, price extraction happens at Bronze ingestion, so the map moves here. `_extract_mtgjson_scalar_prices` is the shared extraction function used by both `seed_historical_prices` (Task 2) and `_snapshot_mtgjson_prices` (Task 3).

- [ ] **Step 1.1: Write the failing tests**

Add class `TestExtractMtgjsonScalarPrices` to `tests/data/cards/storage/test_bronze.py`. Also add the import at the top:

```python
from src.data.cards.storage.bronze.storage import (
    _filter_prices_to_date,
    _records_to_df,
    _extract_mtgjson_scalar_prices,
    _MTGJSON_PRICE_MAP,
)
```

Test class to add after `TestToDF`:

```python
class TestExtractMtgjsonScalarPrices:
    def test_returns_all_null_for_none_input(self):
        result = _extract_mtgjson_scalar_prices(None, "2026-05-11")
        assert result == {col: None for col in _MTGJSON_PRICE_MAP}

    def test_returns_all_null_for_empty_dict(self):
        result = _extract_mtgjson_scalar_prices({}, "2026-05-11")
        assert result == {col: None for col in _MTGJSON_PRICE_MAP}

    def test_extracts_cardmarket_eur(self):
        paper = {"cardmarket": {"retail": {"normal": {"2026-05-11": 1.5}}}}
        result = _extract_mtgjson_scalar_prices(paper, "2026-05-11")
        assert result["cardmarket_eur"] == pytest.approx(1.5)

    def test_selects_max_date_leq_target(self):
        paper = {
            "cardmarket": {
                "retail": {"normal": {"2026-05-10": 1.4, "2026-05-11": 1.5}}
            }
        }
        result = _extract_mtgjson_scalar_prices(paper, "2026-05-11")
        assert result["cardmarket_eur"] == pytest.approx(1.5)

    def test_excludes_dates_after_target(self):
        paper = {
            "cardmarket": {
                "retail": {"normal": {"2026-05-12": 1.6, "2026-05-10": 1.4}}
            }
        }
        result = _extract_mtgjson_scalar_prices(paper, "2026-05-11")
        assert result["cardmarket_eur"] == pytest.approx(1.4)

    def test_no_date_leq_target_returns_null(self):
        paper = {"cardmarket": {"retail": {"normal": {"2026-05-12": 1.6}}}}
        result = _extract_mtgjson_scalar_prices(paper, "2026-05-11")
        assert result["cardmarket_eur"] is None

    def test_missing_retailer_returns_null_for_that_col(self):
        paper = {"tcgplayer": {"retail": {"normal": {"2026-05-11": 3.5}}}}
        result = _extract_mtgjson_scalar_prices(paper, "2026-05-11")
        assert result["cardmarket_eur"] is None
        assert result["tcgplayer_usd"] == pytest.approx(3.5)

    def test_extracts_all_six_cols(self):
        paper = {
            "cardmarket": {
                "retail": {
                    "normal": {"2026-05-11": 3.20},
                    "foil": {"2026-05-11": 8.50},
                },
                "buylist": {"normal": {"2026-05-11": 1.80}},
            },
            "tcgplayer": {
                "retail": {
                    "normal": {"2026-05-11": 3.50},
                    "foil": {"2026-05-11": 9.00},
                },
                "buylist": {"normal": {"2026-05-11": 2.10}},
            },
        }
        result = _extract_mtgjson_scalar_prices(paper, "2026-05-11")
        assert result["cardmarket_eur"] == pytest.approx(3.20)
        assert result["cardmarket_eur_foil"] == pytest.approx(8.50)
        assert result["cardmarket_buylist_eur"] == pytest.approx(1.80)
        assert result["tcgplayer_usd"] == pytest.approx(3.50)
        assert result["tcgplayer_usd_foil"] == pytest.approx(9.00)
        assert result["tcgplayer_buylist_usd"] == pytest.approx(2.10)

    def test_null_buylist_branch_returns_null(self):
        paper = {
            "cardmarket": {
                "retail": {"normal": {"2026-05-11": 3.20}},
                "buylist": None,
            }
        }
        result = _extract_mtgjson_scalar_prices(paper, "2026-05-11")
        assert result["cardmarket_eur"] == pytest.approx(3.20)
        assert result["cardmarket_buylist_eur"] is None

    def test_returns_float_not_str(self):
        paper = {"cardmarket": {"retail": {"normal": {"2026-05-11": "1.5"}}}}
        result = _extract_mtgjson_scalar_prices(paper, "2026-05-11")
        assert isinstance(result["cardmarket_eur"], float)
```

- [ ] **Step 1.2: Run to verify FAIL**

```
pytest tests/data/cards/storage/test_bronze.py::TestExtractMtgjsonScalarPrices -v
```

Expected: `ImportError` — `_extract_mtgjson_scalar_prices` not yet defined.

- [ ] **Step 1.3: Add `_MTGJSON_PRICE_MAP` and `_extract_mtgjson_scalar_prices` to Bronze storage**

Add after `_filter_prices_to_date` in `src/data/cards/storage/bronze/storage.py` (before the `BronzeStorage` class):

```python
_MTGJSON_PRICE_MAP: dict[str, tuple[str, str, str]] = {
    "cardmarket_eur": ("cardmarket", "retail", "normal"),
    "cardmarket_eur_foil": ("cardmarket", "retail", "foil"),
    "cardmarket_buylist_eur": ("cardmarket", "buylist", "normal"),
    "tcgplayer_usd": ("tcgplayer", "retail", "normal"),
    "tcgplayer_usd_foil": ("tcgplayer", "retail", "foil"),
    "tcgplayer_buylist_usd": ("tcgplayer", "buylist", "normal"),
}


def _extract_mtgjson_scalar_prices(
    paper_dict: dict | None, target_date: str
) -> dict[str, float | None]:
    result: dict[str, float | None] = {col: None for col in _MTGJSON_PRICE_MAP}
    if not paper_dict:
        return result
    for col, (retailer, tx_type, finish) in _MTGJSON_PRICE_MAP.items():
        prices = (
            ((paper_dict.get(retailer) or {}).get(tx_type) or {}).get(finish) or {}
        )
        candidates = {k: v for k, v in prices.items() if k <= target_date}
        result[col] = float(candidates[max(candidates)]) if candidates else None
    return result
```

- [ ] **Step 1.4: Run to verify PASS**

```
pytest tests/data/cards/storage/test_bronze.py::TestExtractMtgjsonScalarPrices -v
```

Expected: all 10 tests PASS.

- [ ] **Step 1.5: Commit**

```bash
git add tests/data/cards/storage/test_bronze.py src/data/cards/storage/bronze/storage.py
git commit -m "feat: add _MTGJSON_PRICE_MAP and _extract_mtgjson_scalar_prices to Bronze"
```

---

## Task 2: Bronze — Rewrite `seed_historical_prices`

**Files:**
- Modify: `src/data/cards/storage/bronze/storage.py`
- Modify: `tests/cards/test_storage.py`

### Background

`seed_historical_prices` currently collects dates from both `paper` and `mtgo` platforms (violating ADR-012) and stores filtered JSON blobs via `_filter_prices_to_date`. After this task it collects dates from `paper` only and stores scalar FLOAT columns via `_extract_mtgjson_scalar_prices`.

- [ ] **Step 2.1: Update failing tests in `tests/cards/test_storage.py`**

**Delete** `test_paper_filtered_to_snapshot_date_only` (lines ~581–592) and `test_dates_collected_from_mtgo_platform` (lines ~615–628).

**Add** three new tests at the end of `TestSeedHistoricalPrices`:

```python
def test_scalar_prices_stored_for_2026_04_01(self, storage):
    record = _PriceRecord(uuid="uuid-1", paper=_PAPER_PRICES)
    storage.seed_historical_prices([record])
    row = storage._con.execute(
        f"SELECT cardmarket_eur, cardmarket_eur_foil FROM {self.HISTORY_TABLE}"
        " WHERE snapshot_date = '2026-04-01'"
    ).fetchone()
    assert row is not None
    assert row[0] == pytest.approx(1.0)   # cardmarket_eur
    assert row[1] == pytest.approx(2.0)   # cardmarket_eur_foil

def test_scalar_prices_stored_for_2026_04_02(self, storage):
    record = _PriceRecord(uuid="uuid-1", paper=_PAPER_PRICES)
    storage.seed_historical_prices([record])
    row = storage._con.execute(
        f"SELECT cardmarket_eur, cardmarket_eur_foil FROM {self.HISTORY_TABLE}"
        " WHERE snapshot_date = '2026-04-02'"
    ).fetchone()
    assert row is not None
    assert row[0] == pytest.approx(1.1)   # cardmarket_eur
    assert row[1] is None                 # foil only exists for 2026-04-01

def test_mtgo_prices_not_collected(self, storage):
    record = _PriceRecord(
        uuid="uuid-1",
        paper=None,
        mtgo={"cardhoarder": {"retail": {"normal": {"2026-03-15": 0.5}}}},
    )
    storage.seed_historical_prices([record])
    tables = storage._con.execute(
        f"SELECT table_name FROM information_schema.tables"
        f" WHERE table_name = '{self.HISTORY_TABLE}'"
    ).fetchall()
    assert tables == []
```

Also add `import pytest` at the top of `tests/cards/test_storage.py` if not already there.

- [ ] **Step 2.2: Run to verify FAIL**

```
pytest tests/cards/test_storage.py::TestSeedHistoricalPrices -v
```

Expected: `test_scalar_prices_stored_for_2026_04_01`, `test_scalar_prices_stored_for_2026_04_02`, `test_mtgo_prices_not_collected` FAIL (column `cardmarket_eur` not found; `mtgo` produces a row when it shouldn't).

- [ ] **Step 2.3: Rewrite `seed_historical_prices` in `storage.py`**

Replace the entire method body (keep the docstring):

```python
def seed_historical_prices(self, records: list[BaseModel]) -> None:
    """One-time seeding: explode AllPrices 90-day history into per-date rows.

    Reads MtgjsonCardPrices instances from AllPrices.json (not
    AllPricesToday.json). Each card's paper price dict contains up to 90
    date-keyed entries; this method expands them so that
    bronze_mtgjson_prices_history gets one row per (uuid, date) with scalar
    FLOAT price columns.

    Already-existing (uuid, snapshot_date) pairs are skipped, so the call
    is idempotent and safe to re-run if interrupted.

    Args:
        records: MtgjsonCardPrices instances from AllPrices.json.

    Raises:
        StorageWriteError: If the DuckDB write fails.
    """
    history_table = "bronze_mtgjson_prices_history"
    if not records:
        logger.warning("No price records to seed into %r — skipping", history_table)
        return

    rows = []
    for record in records:
        dump = record.model_dump(mode="json")
        uuid_str = dump["uuid"]
        paper = dump.get("paper") or {}

        dates: set[str] = set()
        for retailer_data in paper.values():
            for tx_type in ("buylist", "retail"):
                listing = (retailer_data or {}).get(tx_type) or {}
                dates.update((listing.get("foil") or {}).keys())
                dates.update((listing.get("normal") or {}).keys())

        for d in dates:
            rows.append(
                {
                    "uuid": uuid_str,
                    "snapshot_date": d,
                    **_extract_mtgjson_scalar_prices(paper, d),
                }
            )

    if not rows:
        logger.warning("No date-keyed prices found in records — skipping seed")
        return

    DuckDBWriter(self._con).append(pd.DataFrame(rows), history_table, "uuid")
    logger.info("Seeded %d historical price rows into %r", len(rows), history_table)
```

- [ ] **Step 2.4: Run to verify PASS**

```
pytest tests/cards/test_storage.py::TestSeedHistoricalPrices -v
```

Expected: all remaining tests PASS (deleted 2, added 3, total net same).

- [ ] **Step 2.5: Commit**

```bash
git add src/data/cards/storage/bronze/storage.py tests/cards/test_storage.py
git commit -m "feat: rewrite seed_historical_prices to use scalar price columns"
```

---

## Task 3: Bronze — `_snapshot_mtgjson_prices`

**Files:**
- Modify: `src/data/cards/storage/bronze/storage.py`
- Modify: `tests/data/cards/storage/test_bronze.py`

### Background

The daily run currently snapshots MTGJson prices via the generic `_snapshot` method (which stores the full model dump including `paper` JSON). This task replaces that with a dedicated method that extracts scalar columns at snapshot time.

- [ ] **Step 3.1: Write the failing tests**

Add `_MtgjsonPrices` model and `TestSnapshotMtgjsonPrices` class to `tests/data/cards/storage/test_bronze.py`:

```python
class _MtgjsonPrices(BaseModel):
    uuid: str
    paper: dict | None = None


class TestSnapshotMtgjsonPrices:
    HISTORY_TABLE = "bronze_mtgjson_prices_history"

    def test_creates_history_table_on_first_call(self):
        with _bronze() as b:
            record = _MtgjsonPrices(
                uuid="u1",
                paper={"cardmarket": {"retail": {"normal": {"2026-06-24": 3.20}}}},
            )
            b._snapshot_mtgjson_prices([record])
            row = b._con.execute(
                f"SELECT count(*) FROM {self.HISTORY_TABLE}"
            ).fetchone()
        assert row is not None and row[0] == 1

    def test_extracts_scalar_columns(self):
        with _bronze() as b:
            record = _MtgjsonPrices(
                uuid="u1",
                paper={
                    "cardmarket": {
                        "retail": {
                            "normal": {"2026-06-24": 3.20},
                            "foil": {"2026-06-24": 8.50},
                        },
                        "buylist": {"normal": {"2026-06-24": 1.80}},
                    },
                    "tcgplayer": {
                        "retail": {
                            "normal": {"2026-06-24": 3.50},
                            "foil": {"2026-06-24": 9.00},
                        },
                        "buylist": {"normal": {"2026-06-24": 2.10}},
                    },
                },
            )
            b._snapshot_mtgjson_prices([record])
            row = b._con.execute(
                f"SELECT cardmarket_eur, cardmarket_eur_foil, cardmarket_buylist_eur,"
                f" tcgplayer_usd, tcgplayer_usd_foil, tcgplayer_buylist_usd"
                f" FROM {self.HISTORY_TABLE} WHERE uuid = 'u1'"
            ).fetchone()
        assert row is not None
        assert row[0] == pytest.approx(3.20)
        assert row[1] == pytest.approx(8.50)
        assert row[2] == pytest.approx(1.80)
        assert row[3] == pytest.approx(3.50)
        assert row[4] == pytest.approx(9.00)
        assert row[5] == pytest.approx(2.10)

    def test_null_paper_produces_all_null_price_cols(self):
        with _bronze() as b:
            record = _MtgjsonPrices(uuid="u1", paper=None)
            b._snapshot_mtgjson_prices([record])
            row = b._con.execute(
                f"SELECT cardmarket_eur FROM {self.HISTORY_TABLE}"
            ).fetchone()
        assert row is not None and row[0] is None

    def test_idempotent_on_duplicate_uuid_date(self):
        with _bronze() as b:
            record = _MtgjsonPrices(
                uuid="u1",
                paper={"cardmarket": {"retail": {"normal": {"2026-06-24": 3.20}}}},
            )
            b._snapshot_mtgjson_prices([record])
            b._snapshot_mtgjson_prices([record])
            row = b._con.execute(
                f"SELECT count(*) FROM {self.HISTORY_TABLE}"
            ).fetchone()
        assert row is not None and row[0] == 1

    def test_uses_today_as_snapshot_date(self):
        from datetime import date as date_cls

        with _bronze() as b:
            record = _MtgjsonPrices(uuid="u1", paper=None)
            with patch("src.data.cards.storage.bronze.storage.date") as mock_date:
                mock_date.today.return_value = date_cls.fromisoformat("2026-06-24")
                b._snapshot_mtgjson_prices([record])
            row = b._con.execute(
                f"SELECT snapshot_date FROM {self.HISTORY_TABLE}"
            ).fetchone()
        assert row is not None and str(row[0]) == "2026-06-24"

    def test_skips_when_records_empty(self):
        with _bronze() as b:
            b._snapshot_mtgjson_prices([])
            row = b._con.execute(
                "SELECT count(*) FROM information_schema.tables"
                f" WHERE table_name = '{self.HISTORY_TABLE}'"
            ).fetchone()
        assert row is not None and row[0] == 0
```

- [ ] **Step 3.2: Run to verify FAIL**

```
pytest tests/data/cards/storage/test_bronze.py::TestSnapshotMtgjsonPrices -v
```

Expected: `AttributeError` — `_snapshot_mtgjson_prices` not defined.

- [ ] **Step 3.3: Add `_snapshot_mtgjson_prices` to `BronzeStorage`**

Add this method after `seed_historical_prices` in `BronzeStorage`:

```python
def _snapshot_mtgjson_prices(self, records: list[BaseModel]) -> None:
    if not records:
        logger.warning("No MTGJson price records to snapshot — skipping")
        return

    today_iso = date.today().isoformat()
    rows = []
    for record in records:
        dump = record.model_dump(mode="json")
        rows.append(
            {
                "uuid": dump["uuid"],
                "snapshot_date": today_iso,
                **_extract_mtgjson_scalar_prices(dump.get("paper"), today_iso),
            }
        )

    df = pd.DataFrame(rows)
    logger.progress("Snapshotting %d MTGJson price rows", len(df))
    self._writer.append(df, "bronze_mtgjson_prices_history", "uuid")
    logger.info("Snapshotted %d MTGJson price rows for %s", len(rows), today_iso)
```

- [ ] **Step 3.4: Run to verify PASS**

```
pytest tests/data/cards/storage/test_bronze.py::TestSnapshotMtgjsonPrices -v
```

Expected: all 6 tests PASS.

- [ ] **Step 3.5: Commit**

```bash
git add src/data/cards/storage/bronze/storage.py tests/data/cards/storage/test_bronze.py
git commit -m "feat: add _snapshot_mtgjson_prices to BronzeStorage"
```

---

## Task 4: Bronze — `_snapshot_scryfall_prices`

**Files:**
- Modify: `src/data/cards/storage/bronze/storage.py`
- Modify: `tests/data/cards/storage/test_bronze.py`

### Background

The daily run currently snapshots Scryfall prices through the generic `_snapshot` mechanism with `fields=["prices"]`, storing a `prices` JSON column. This task adds a dedicated method that writes scalar `eur`, `eur_foil`, `usd`, `usd_foil` FLOAT columns.

- [ ] **Step 4.1: Write the failing tests**

Add `_ScryfallCard` model and `TestSnapshotScryfallPrices` to `tests/data/cards/storage/test_bronze.py`:

```python
class _ScryfallCard(BaseModel):
    id: str
    prices: dict | None = None


class TestSnapshotScryfallPrices:
    HISTORY_TABLE = "bronze_scryfall_prices_history"

    def test_creates_history_table_on_first_call(self):
        with _bronze() as b:
            record = _ScryfallCard(
                id="s1", prices={"eur": "3.20", "eur_foil": None, "usd": None, "usd_foil": None}
            )
            b._snapshot_scryfall_prices([record])
            row = b._con.execute(
                f"SELECT count(*) FROM {self.HISTORY_TABLE}"
            ).fetchone()
        assert row is not None and row[0] == 1

    def test_extracts_eur_as_float(self):
        with _bronze() as b:
            record = _ScryfallCard(
                id="s1",
                prices={"eur": "3.20", "eur_foil": "8.50", "usd": "3.50", "usd_foil": "9.00"},
            )
            b._snapshot_scryfall_prices([record])
            row = b._con.execute(
                f"SELECT eur, eur_foil, usd, usd_foil FROM {self.HISTORY_TABLE}"
            ).fetchone()
        assert row is not None
        assert row[0] == pytest.approx(3.20)
        assert row[1] == pytest.approx(8.50)
        assert row[2] == pytest.approx(3.50)
        assert row[3] == pytest.approx(9.00)

    def test_null_price_fields_produce_null_columns(self):
        with _bronze() as b:
            record = _ScryfallCard(
                id="s1",
                prices={"eur": "3.20", "eur_foil": None, "usd": None, "usd_foil": None},
            )
            b._snapshot_scryfall_prices([record])
            row = b._con.execute(
                f"SELECT eur, eur_foil FROM {self.HISTORY_TABLE}"
            ).fetchone()
        assert row is not None
        assert row[0] == pytest.approx(3.20)
        assert row[1] is None

    def test_tix_key_is_ignored(self):
        with _bronze() as b:
            record = _ScryfallCard(
                id="s1",
                prices={"eur": "3.20", "tix": "0.05"},
            )
            b._snapshot_scryfall_prices([record])
            cols = {r[0] for r in b._con.execute(
                f"DESCRIBE {self.HISTORY_TABLE}"
            ).fetchall()}
        assert "tix" not in cols
        assert "eur" in cols

    def test_none_prices_dict_produces_all_null_columns(self):
        with _bronze() as b:
            record = _ScryfallCard(id="s1", prices=None)
            b._snapshot_scryfall_prices([record])
            row = b._con.execute(
                f"SELECT eur FROM {self.HISTORY_TABLE}"
            ).fetchone()
        assert row is not None and row[0] is None

    def test_idempotent_on_duplicate_id_date(self):
        with _bronze() as b:
            record = _ScryfallCard(
                id="s1", prices={"eur": "3.20"}
            )
            b._snapshot_scryfall_prices([record])
            b._snapshot_scryfall_prices([record])
            row = b._con.execute(
                f"SELECT count(*) FROM {self.HISTORY_TABLE}"
            ).fetchone()
        assert row is not None and row[0] == 1

    def test_skips_when_records_empty(self):
        with _bronze() as b:
            b._snapshot_scryfall_prices([])
            row = b._con.execute(
                "SELECT count(*) FROM information_schema.tables"
                f" WHERE table_name = '{self.HISTORY_TABLE}'"
            ).fetchone()
        assert row is not None and row[0] == 0
```

- [ ] **Step 4.2: Run to verify FAIL**

```
pytest tests/data/cards/storage/test_bronze.py::TestSnapshotScryfallPrices -v
```

Expected: `AttributeError` — `_snapshot_scryfall_prices` not defined.

- [ ] **Step 4.3: Add `_snapshot_scryfall_prices` to `BronzeStorage`**

Add this method after `_snapshot_mtgjson_prices` in `BronzeStorage`:

```python
def _snapshot_scryfall_prices(self, records: list[BaseModel]) -> None:
    if not records:
        logger.warning("No Scryfall records to snapshot prices for — skipping")
        return

    today_iso = date.today().isoformat()
    rows = []
    for record in records:
        dump = record.model_dump(mode="json")
        prices = dump.get("prices") or {}
        rows.append(
            {
                "id": dump["id"],
                "snapshot_date": today_iso,
                "eur": float(prices["eur"]) if prices.get("eur") is not None else None,
                "eur_foil": float(prices["eur_foil"]) if prices.get("eur_foil") is not None else None,
                "usd": float(prices["usd"]) if prices.get("usd") is not None else None,
                "usd_foil": float(prices["usd_foil"]) if prices.get("usd_foil") is not None else None,
            }
        )

    df = pd.DataFrame(rows)
    logger.progress("Snapshotting %d Scryfall price rows", len(df))
    self._writer.append(df, "bronze_scryfall_prices_history", "id")
    logger.info("Snapshotted %d Scryfall price rows for %s", len(rows), today_iso)
```

- [ ] **Step 4.4: Run to verify PASS**

```
pytest tests/data/cards/storage/test_bronze.py::TestSnapshotScryfallPrices -v
```

Expected: all 7 tests PASS.

- [ ] **Step 4.5: Commit**

```bash
git add src/data/cards/storage/bronze/storage.py tests/data/cards/storage/test_bronze.py
git commit -m "feat: add _snapshot_scryfall_prices to BronzeStorage"
```

---

## Task 5: Bronze — Wire up `STORAGE_CONFIG`, `daily_update`, and `populate`

**Files:**
- Modify: `src/data/cards/storage/bronze/config.py`
- Modify: `src/data/cards/storage/bronze/storage.py`
- Modify: `tests/data/cards/storage/test_bronze.py`
- Modify: `tests/cards/test_storage.py`

### Background

After removing price snapshots from `STORAGE_CONFIG`, both `daily_update` and `populate` must call the new dedicated methods explicitly. The test that asserts `mock_snap.call_count == 4` must drop to 2 and gain assertions for the two new methods.

- [ ] **Step 5.1: Update `TestDailyUpdate` in `tests/cards/test_storage.py`**

Find the test named `test_daily_update_calls_snapshot` (currently asserts `mock_snap.call_count == 4`) and replace it entirely:

```python
def test_daily_update_calls_snapshot(self, storage):
    with (
        patch.object(storage, "_incremental_load"),
        patch.object(storage, "_snapshot") as mock_snap,
        patch.object(storage, "_snapshot_scryfall_prices") as mock_scryfall_prices,
        patch.object(storage, "_snapshot_mtgjson_prices") as mock_mtgjson_prices,
    ):
        storage.daily_update(
            {
                "scryfall": ([MagicMock()], []),
                "mtgjson_prices": ([MagicMock()], []),
                "mtgjson_cards": ([], []),
            }
        )
    # _process_sources: scryfall meta (1) + format_staples (1) = 2
    assert mock_snap.call_count == 2
    mock_scryfall_prices.assert_called_once()
    mock_mtgjson_prices.assert_called_once()
```

Also find `test_snapshot_called_for_scryfall_snapshots` in `TestPopulate` in `tests/data/cards/storage/test_bronze.py` (line ~433) and replace it:

```python
def test_snapshot_called_for_scryfall_meta_history(self):
    with _bronze() as b:
        with (
            patch.object(b, "_full_load_table"),
            patch.object(b, "_snapshot") as mock_snap,
            patch.object(b, "seed_historical_prices"),
            patch.object(b, "_snapshot_scryfall_prices"),
        ):
            b.populate({"scryfall": ([], [])})
        history_tables = [
            c.kwargs["history_table"] for c in mock_snap.call_args_list
        ]
        assert "bronze_scryfall_meta_history" in history_tables
        assert "bronze_scryfall_prices_history" not in history_tables
```

Also find `test_snapshot_called_for_sources_with_snapshot_config` in `TestProcessSources` (line ~385) and update the comment:

```python
def test_snapshot_called_for_sources_with_snapshot_config(self):
    with _bronze() as b:
        with (
            patch.object(b, "_full_load_table"),
            patch.object(b, "_snapshot") as mock_snap,
        ):
            b._process_sources({"scryfall": ([], [])}, update=False)
        # scryfall: 1 (meta_history), format_staples: 1 = 2 total
        assert mock_snap.call_count >= 1
```

- [ ] **Step 5.2: Run to verify the snapshot-count test FAILS (and old tests still pass)**

```
pytest tests/cards/test_storage.py::TestDailyUpdate::test_daily_update_calls_snapshot -v
```

Expected: PASS (patching new methods) — but the broader test run should show no regressions yet.

Actually, at this point the implementation hasn't changed yet, so:
- `mock_snap.call_count` would be 4 (old behavior still) → the updated assertion `== 2` FAILS.

```
pytest tests/cards/test_storage.py::TestDailyUpdate -v
```

Expected: `test_daily_update_calls_snapshot` FAILS with `AssertionError: assert 4 == 2`.

- [ ] **Step 5.3: Update `STORAGE_CONFIG` in `config.py`**

Replace the full `STORAGE_CONFIG` dict:

```python
STORAGE_CONFIG: dict[str, SourceStorageConfig] = {
    "scryfall": SourceStorageConfig(
        table="bronze_scryfall_cards",
        key="id",
        snapshots=[
            # bronze_scryfall_prices_history is handled by _snapshot_scryfall_prices
            SnapshotConfig(
                "bronze_scryfall_meta_history",
                fields=[
                    "legalities",
                    "edhrec_rank",
                    "reserved",
                    "promo_types",
                    "finishes",
                ],
            ),
        ],
    ),
    "mtgjson_cards": SourceStorageConfig(
        table="bronze_mtgjson_cards",
        key="uuid",
    ),
    "mtgjson_prices": SourceStorageConfig(
        table=None,
        key="uuid",
        snapshots=[],  # handled by _snapshot_mtgjson_prices
    ),
    "format_staples": SourceStorageConfig(
        table=None,
        key="id",
        snapshots=[SnapshotConfig("bronze_format_staples_history")],
    ),
    "tournament_results": SourceStorageConfig(
        table="bronze_tournament_results",
        key="id",
        incremental=True,
    ),
}
```

- [ ] **Step 5.4: Add explicit calls to `daily_update` and `populate` in `storage.py`**

Update `daily_update`:

```python
def daily_update(
    self, results: dict[str, tuple[list[BaseModel], list[dict[str, object]]]]
) -> None:
    """Incrementally update card data and append a daily snapshot.

    Intended to be run once per day after the initial populate call.
    Sources marked incremental=True are upserted; others are fully replaced.
    Snapshot history tables accumulate one row per card per day.

    Sources and their write strategy are declared in STORAGE_CONFIG via
    _process_sources. If one source fails the others are still processed.
    Price snapshots use dedicated methods (_snapshot_scryfall_prices and
    _snapshot_mtgjson_prices) called after _process_sources.

    Args:
        results: Output of ingesting_pipeline — maps source type to
            a (records, errors) tuple.
    """
    logger.info("Starting DuckDB update")
    self._process_sources(results, update=True)

    scryfall_records, _ = results.get("scryfall", ([], []))
    self._snapshot_scryfall_prices(scryfall_records)

    mtgjson_records, _ = results.get("mtgjson_prices", ([], []))
    self._snapshot_mtgjson_prices(mtgjson_records)
```

Update `populate`:

```python
def populate(
    self, results: dict[str, tuple[list[BaseModel], list[dict[str, object]]]]
) -> None:
    """Full load of all sources into Bronze tables and an initial snapshot.

    Intended for the initial database population or a full rebuild.
    All Bronze tables are dropped and recreated via _process_sources.
    Snapshot history tables are created automatically on first call.

    After the config loop, _snapshot_scryfall_prices captures today's
    Scryfall prices as scalars, and seed_historical_prices backfills
    the 90-day MTGJson price history from AllPrices.json.

    Args:
        results: Output of ingesting_pipeline — maps source type to
            a (records, errors) tuple.
    """
    logger.info("Starting DuckDB populate")
    self._process_sources(results, update=False)

    scryfall_records, _ = results.get("scryfall", ([], []))
    try:
        self._snapshot_scryfall_prices(scryfall_records)
    except StorageWriteError as e:
        logger.error(
            "Scryfall price snapshot failed during populate: %s — skipping",
            e,
            exc_info=True,
        )

    prices_records, _ = results.get("mtgjson_prices", ([], []))
    try:
        self.seed_historical_prices(prices_records)
    except StorageWriteError as e:
        logger.error(
            "Historical price seed failed during populate: %s — skipping",
            e,
            exc_info=True,
        )
```

- [ ] **Step 5.5: Run to verify PASS**

```
pytest tests/cards/test_storage.py::TestDailyUpdate tests/data/cards/storage/test_bronze.py::TestProcessSources tests/data/cards/storage/test_bronze.py::TestPopulate -v
```

Expected: all PASS.

- [ ] **Step 5.6: Run full test suite to check for regressions**

```
pytest tests/ -v --tb=short 2>&1 | tail -30
```

Expected: no new failures.

- [ ] **Step 5.7: Commit**

```bash
git add src/data/cards/storage/bronze/config.py src/data/cards/storage/bronze/storage.py tests/data/cards/storage/test_bronze.py tests/cards/test_storage.py
git commit -m "feat: wire _snapshot_scryfall_prices and _snapshot_mtgjson_prices into daily_update and populate"
```

---

## Task 6: Bronze — Delete `_filter_prices_to_date`

**Files:**
- Modify: `src/data/cards/storage/bronze/storage.py`
- Modify: `tests/data/cards/storage/test_bronze.py`

### Background

`_filter_prices_to_date` was only called by `seed_historical_prices` (now replaced) and `_snapshot` (no longer invoked for price tables). It has no callers and can be deleted.

- [ ] **Step 6.1: Update `test_bronze.py` imports and delete `TestFilterPricesToDate`**

In `tests/data/cards/storage/test_bronze.py`, line 11, change the import:

```python
# Old:
from src.data.cards.storage.bronze.storage import _filter_prices_to_date, _records_to_df

# New:
from src.data.cards.storage.bronze.storage import (
    _extract_mtgjson_scalar_prices,
    _MTGJSON_PRICE_MAP,
    _records_to_df,
)
```

Delete the entire `TestFilterPricesToDate` class (lines ~41–118 — from the `class TestFilterPricesToDate:` line to just before `class TestToDF:`).

- [ ] **Step 6.2: Run to verify no import errors**

```
pytest tests/data/cards/storage/test_bronze.py -v --tb=short 2>&1 | tail -20
```

Expected: all tests PASS (no `TestFilterPricesToDate` tests exist to fail, import resolved).

- [ ] **Step 6.3: Delete `_filter_prices_to_date` from `storage.py`**

Remove the entire function definition (lines ~35–56 in the current file):

```python
def _filter_prices_to_date(
    platform_prices: dict[str, Any] | None, target_date: str
) -> dict[str, Any] | None:
    ...
```

Also remove `from typing import Any` if it is now unused (check no other reference to `Any` in the file first — `BronzeStorage._process_sources` uses `dict[str, object]` not `Any`, so `Any` can be removed).

- [ ] **Step 6.4: Run to verify PASS**

```
pytest tests/data/cards/storage/test_bronze.py tests/cards/test_storage.py -v --tb=short 2>&1 | tail -20
```

Expected: all PASS.

- [ ] **Step 6.5: Commit**

```bash
git add src/data/cards/storage/bronze/storage.py tests/data/cards/storage/test_bronze.py
git commit -m "refactor: delete _filter_prices_to_date and its tests"
```

---

## Task 7: Migration Script

**Files:**
- Create: `scripts/migrate_bronze_prices.py`
- Create: `tests/scripts/__init__.py`
- Create: `tests/scripts/test_migrate_bronze_prices.py`

### Background

The migration reads from `cards_copy.duckdb` (backup of the original Bronze DB with JSON columns) and atomically replaces both price history tables in the live `cards.duckdb` with scalar schemas. The live DB is never left in partial state — if the INSERT phase fails, the old table is untouched.

- [ ] **Step 7.1: Create `tests/scripts/__init__.py`**

Create an empty file at `tests/scripts/__init__.py`.

- [ ] **Step 7.2: Write the failing tests**

Create `tests/scripts/test_migrate_bronze_prices.py`:

```python
"""Tests for scripts/migrate_bronze_prices.py."""

import json
from pathlib import Path

import duckdb
import pytest

from scripts.migrate_bronze_prices import migrate_mtgjson_prices, migrate_scryfall_prices


def _make_mtgjson_source(path: str) -> None:
    """Create a Bronze DB with old JSON-column schema for bronze_mtgjson_prices_history."""
    con = duckdb.connect(path)
    con.execute("""
        CREATE TABLE bronze_mtgjson_prices_history (
            uuid          VARCHAR,
            snapshot_date VARCHAR,
            paper         VARCHAR,
            mtgo          VARCHAR
        )
    """)
    paper = json.dumps({
        "cardmarket": {
            "retail": {"normal": {"2026-05-11": 3.20}, "foil": {"2026-05-11": 8.50}},
            "buylist": {"normal": {"2026-05-11": 1.80}},
        },
        "tcgplayer": {
            "retail": {"normal": {"2026-05-11": 3.50}, "foil": {"2026-05-11": 9.00}},
            "buylist": {"normal": {"2026-05-11": 2.10}},
        },
    })
    con.execute(
        "INSERT INTO bronze_mtgjson_prices_history VALUES (?, ?, ?, NULL)",
        ["uuid-1", "2026-05-11", paper],
    )
    con.close()


def _make_scryfall_source(path: str) -> None:
    """Create a Bronze DB with old JSON-column schema for bronze_scryfall_prices_history."""
    con = duckdb.connect(path)
    con.execute("""
        CREATE TABLE bronze_scryfall_prices_history (
            id            VARCHAR,
            snapshot_date VARCHAR,
            prices        VARCHAR
        )
    """)
    prices = json.dumps({"eur": "3.20", "eur_foil": "8.50", "usd": "3.50", "usd_foil": None, "tix": "0.05"})
    con.execute(
        "INSERT INTO bronze_scryfall_prices_history VALUES (?, ?, ?)",
        ["s1", "2026-05-11", prices],
    )
    con.close()


def _make_target_with_old_mtgjson(source_path: str, target_path: str) -> None:
    """Copy the old mtgjson prices table into the target DB (simulates live DB)."""
    import shutil
    shutil.copy(source_path, target_path)


def _make_target_with_old_scryfall(source_path: str, target_path: str) -> None:
    import shutil
    shutil.copy(source_path, target_path)


class TestMigrateMtgjsonPrices:
    def test_creates_scalar_columns(self, tmp_path):
        source = str(tmp_path / "source.duckdb")
        target = str(tmp_path / "target.duckdb")
        _make_mtgjson_source(source)
        _make_target_with_old_mtgjson(source, target)

        migrate_mtgjson_prices(source, target)

        con = duckdb.connect(target, read_only=True)
        cols = {r[0] for r in con.execute("DESCRIBE bronze_mtgjson_prices_history").fetchall()}
        con.close()
        assert "cardmarket_eur" in cols
        assert "tcgplayer_usd" in cols
        assert "paper" not in cols
        assert "mtgo" not in cols

    def test_extracts_correct_scalar_values(self, tmp_path):
        source = str(tmp_path / "source.duckdb")
        target = str(tmp_path / "target.duckdb")
        _make_mtgjson_source(source)
        _make_target_with_old_mtgjson(source, target)

        migrate_mtgjson_prices(source, target)

        con = duckdb.connect(target, read_only=True)
        row = con.execute(
            "SELECT cardmarket_eur, cardmarket_eur_foil, cardmarket_buylist_eur,"
            " tcgplayer_usd, tcgplayer_usd_foil, tcgplayer_buylist_usd"
            " FROM bronze_mtgjson_prices_history WHERE uuid = 'uuid-1'"
        ).fetchone()
        con.close()
        assert row is not None
        assert row[0] == pytest.approx(3.20)
        assert row[1] == pytest.approx(8.50)
        assert row[2] == pytest.approx(1.80)
        assert row[3] == pytest.approx(3.50)
        assert row[4] == pytest.approx(9.00)
        assert row[5] == pytest.approx(2.10)

    def test_preserves_uuid_and_snapshot_date(self, tmp_path):
        source = str(tmp_path / "source.duckdb")
        target = str(tmp_path / "target.duckdb")
        _make_mtgjson_source(source)
        _make_target_with_old_mtgjson(source, target)

        migrate_mtgjson_prices(source, target)

        con = duckdb.connect(target, read_only=True)
        row = con.execute(
            "SELECT uuid, snapshot_date FROM bronze_mtgjson_prices_history"
        ).fetchone()
        con.close()
        assert row is not None
        assert row[0] == "uuid-1"
        assert str(row[1]) == "2026-05-11"

    def test_returns_row_count(self, tmp_path):
        source = str(tmp_path / "source.duckdb")
        target = str(tmp_path / "target.duckdb")
        _make_mtgjson_source(source)
        _make_target_with_old_mtgjson(source, target)

        count = migrate_mtgjson_prices(source, target)
        assert count == 1

    def test_empty_source_table_produces_no_rows(self, tmp_path):
        source = str(tmp_path / "source.duckdb")
        target = str(tmp_path / "target.duckdb")
        con = duckdb.connect(source)
        con.execute("""
            CREATE TABLE bronze_mtgjson_prices_history (
                uuid VARCHAR, snapshot_date VARCHAR, paper VARCHAR, mtgo VARCHAR
            )
        """)
        con.close()
        _make_target_with_old_mtgjson(source, target)

        count = migrate_mtgjson_prices(source, target)

        assert count == 0
        con = duckdb.connect(target, read_only=True)
        row = con.execute(
            "SELECT count(*) FROM bronze_mtgjson_prices_history"
        ).fetchone()
        con.close()
        assert row is not None and row[0] == 0


class TestMigrateScryfallPrices:
    def test_creates_scalar_columns(self, tmp_path):
        source = str(tmp_path / "source.duckdb")
        target = str(tmp_path / "target.duckdb")
        _make_scryfall_source(source)
        _make_target_with_old_scryfall(source, target)

        migrate_scryfall_prices(source, target)

        con = duckdb.connect(target, read_only=True)
        cols = {r[0] for r in con.execute("DESCRIBE bronze_scryfall_prices_history").fetchall()}
        con.close()
        assert "eur" in cols
        assert "eur_foil" in cols
        assert "usd" in cols
        assert "usd_foil" in cols
        assert "prices" not in cols
        assert "tix" not in cols

    def test_extracts_correct_scalar_values(self, tmp_path):
        source = str(tmp_path / "source.duckdb")
        target = str(tmp_path / "target.duckdb")
        _make_scryfall_source(source)
        _make_target_with_old_scryfall(source, target)

        migrate_scryfall_prices(source, target)

        con = duckdb.connect(target, read_only=True)
        row = con.execute(
            "SELECT eur, eur_foil, usd, usd_foil FROM bronze_scryfall_prices_history"
        ).fetchone()
        con.close()
        assert row is not None
        assert row[0] == pytest.approx(3.20)
        assert row[1] == pytest.approx(8.50)
        assert row[2] == pytest.approx(3.50)
        assert row[3] is None

    def test_null_prices_dict_row_produces_all_null(self, tmp_path):
        source = str(tmp_path / "source.duckdb")
        target = str(tmp_path / "target.duckdb")
        con = duckdb.connect(source)
        con.execute("""
            CREATE TABLE bronze_scryfall_prices_history (
                id VARCHAR, snapshot_date VARCHAR, prices VARCHAR
            )
        """)
        con.execute(
            "INSERT INTO bronze_scryfall_prices_history VALUES (?, ?, NULL)",
            ["s1", "2026-05-11"],
        )
        con.close()
        _make_target_with_old_scryfall(source, target)

        migrate_scryfall_prices(source, target)

        con = duckdb.connect(target, read_only=True)
        row = con.execute(
            "SELECT eur FROM bronze_scryfall_prices_history"
        ).fetchone()
        con.close()
        assert row is not None and row[0] is None

    def test_returns_row_count(self, tmp_path):
        source = str(tmp_path / "source.duckdb")
        target = str(tmp_path / "target.duckdb")
        _make_scryfall_source(source)
        _make_target_with_old_scryfall(source, target)

        count = migrate_scryfall_prices(source, target)
        assert count == 1
```

- [ ] **Step 7.3: Run to verify FAIL**

```
pytest tests/scripts/test_migrate_bronze_prices.py -v
```

Expected: `ModuleNotFoundError` — `scripts.migrate_bronze_prices` not found.

- [ ] **Step 7.4: Create `scripts/migrate_bronze_prices.py`**

```python
"""One-time migration: replace JSON price columns with scalar FLOAT columns.

Usage:
    python scripts/migrate_bronze_prices.py \\
        --source data/bronze/cards_copy.duckdb \\
        --target data/bronze/cards.duckdb

The old tables in `target` are atomically replaced. `source` is opened
read-only and is not modified. Run this EXACTLY ONCE after deploying the
Bronze ingestion changes (Tasks 1–6) but BEFORE deploying Silver changes
(Tasks 9–12).
"""

import argparse
import json
import sys
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).parents[1]))
from src.data.cards.storage.bronze.storage import (
    _MTGJSON_PRICE_MAP,
    _extract_mtgjson_scalar_prices,
)


def migrate_mtgjson_prices(source_path: str, target_path: str) -> int:
    """Migrate bronze_mtgjson_prices_history from JSON paper column to scalar columns.

    Returns:
        Number of rows migrated.
    """
    src = duckdb.connect(source_path, read_only=True)
    tgt = duckdb.connect(target_path, read_only=False)

    try:
        rows = src.execute(
            "SELECT uuid, snapshot_date, paper FROM bronze_mtgjson_prices_history"
        ).fetchall()

        scalar_col_defs = ", ".join(f"{col} FLOAT" for col in _MTGJSON_PRICE_MAP)
        tgt.execute(f"""
            CREATE TABLE bronze_mtgjson_prices_history_new (
                uuid          VARCHAR NOT NULL,
                snapshot_date VARCHAR NOT NULL,
                {scalar_col_defs}
            )
        """)

        col_names = list(_MTGJSON_PRICE_MAP.keys())
        placeholders = ", ".join(["?"] * (2 + len(col_names)))
        insert_sql = (
            f"INSERT INTO bronze_mtgjson_prices_history_new"
            f" (uuid, snapshot_date, {', '.join(col_names)})"
            f" VALUES ({placeholders})"
        )

        batch = []
        for uuid, snapshot_date, paper_json in rows:
            paper = json.loads(paper_json) if isinstance(paper_json, str) else paper_json
            scalars = _extract_mtgjson_scalar_prices(paper, str(snapshot_date))
            batch.append([uuid, str(snapshot_date)] + [scalars[col] for col in col_names])

        if batch:
            tgt.executemany(insert_sql, batch)

        tgt.execute("DROP TABLE IF EXISTS bronze_mtgjson_prices_history")
        tgt.execute(
            "ALTER TABLE bronze_mtgjson_prices_history_new"
            " RENAME TO bronze_mtgjson_prices_history"
        )
        tgt.execute("CHECKPOINT")

        return len(rows)
    finally:
        src.close()
        tgt.close()


def migrate_scryfall_prices(source_path: str, target_path: str) -> int:
    """Migrate bronze_scryfall_prices_history from JSON prices column to scalar columns.

    Returns:
        Number of rows migrated.
    """
    src = duckdb.connect(source_path, read_only=True)
    tgt = duckdb.connect(target_path, read_only=False)

    try:
        rows = src.execute(
            "SELECT id, snapshot_date, prices FROM bronze_scryfall_prices_history"
        ).fetchall()

        tgt.execute("""
            CREATE TABLE bronze_scryfall_prices_history_new (
                id            VARCHAR NOT NULL,
                snapshot_date VARCHAR NOT NULL,
                eur           FLOAT,
                eur_foil      FLOAT,
                usd           FLOAT,
                usd_foil      FLOAT
            )
        """)

        batch = []
        for scryfall_id, snapshot_date, prices_json in rows:
            if prices_json is None:
                prices: dict = {}
            elif isinstance(prices_json, str):
                prices = json.loads(prices_json)
            else:
                prices = prices_json

            batch.append([
                scryfall_id,
                str(snapshot_date),
                float(prices["eur"]) if prices.get("eur") is not None else None,
                float(prices["eur_foil"]) if prices.get("eur_foil") is not None else None,
                float(prices["usd"]) if prices.get("usd") is not None else None,
                float(prices["usd_foil"]) if prices.get("usd_foil") is not None else None,
            ])

        if batch:
            tgt.executemany(
                "INSERT INTO bronze_scryfall_prices_history_new VALUES (?, ?, ?, ?, ?, ?)",
                batch,
            )

        tgt.execute("DROP TABLE IF EXISTS bronze_scryfall_prices_history")
        tgt.execute(
            "ALTER TABLE bronze_scryfall_prices_history_new"
            " RENAME TO bronze_scryfall_prices_history"
        )
        tgt.execute("CHECKPOINT")

        return len(rows)
    finally:
        src.close()
        tgt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate Bronze price tables to scalar columns")
    parser.add_argument("--source", required=True, help="Path to cards_copy.duckdb (backup)")
    parser.add_argument("--target", required=True, help="Path to live cards.duckdb")
    args = parser.parse_args()

    print(f"Migrating MTGJson prices: {args.source} → {args.target}")
    n = migrate_mtgjson_prices(args.source, args.target)
    print(f"  Migrated {n} rows")

    print(f"Migrating Scryfall prices: {args.source} → {args.target}")
    n = migrate_scryfall_prices(args.source, args.target)
    print(f"  Migrated {n} rows")

    print("Migration complete.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 7.5: Run to verify PASS**

```
pytest tests/scripts/test_migrate_bronze_prices.py -v
```

Expected: all 9 tests PASS.

- [ ] **Step 7.6: Commit**

```bash
git add scripts/migrate_bronze_prices.py tests/scripts/__init__.py tests/scripts/test_migrate_bronze_prices.py
git commit -m "feat: add migration script for scalar Bronze price tables"
```

---

## Task 8: Run Migration on Live Data

> **Manual step — no code changes.** This is a runtime boundary.

- [ ] **Step 8.1: Verify backup exists**

```bash
ls -lh data/bronze/cards_copy.duckdb
```

Expected: file exists and is non-zero.

- [ ] **Step 8.2: Run the migration**

```bash
python scripts/migrate_bronze_prices.py \
    --source data/bronze/cards_copy.duckdb \
    --target data/bronze/cards.duckdb
```

Expected output:
```
Migrating MTGJson prices: data/bronze/cards_copy.duckdb → data/bronze/cards.duckdb
  Migrated XXXXXXX rows
Migrating Scryfall prices: data/bronze/cards_copy.duckdb → data/bronze/cards.duckdb
  Migrated XXXXXXX rows
Migration complete.
```

- [ ] **Step 8.3: Verify schema in live DB**

```python
import duckdb
con = duckdb.connect("data/bronze/cards.duckdb", read_only=True)
print(con.execute("DESCRIBE bronze_mtgjson_prices_history").fetchall())
print(con.execute("DESCRIBE bronze_scryfall_prices_history").fetchall())
con.close()
```

Expected: `bronze_mtgjson_prices_history` has `cardmarket_eur`, no `paper`. `bronze_scryfall_prices_history` has `eur`, no `prices`.

---

## Task 9: Silver — `_build_scryfall_base` → SQL file

**Files:**
- Create: `src/data/cards/storage/silver/sql/scryfall_prices_base.sql`
- Modify: `src/data/cards/storage/silver/prices.py`
- Modify: `tests/data/cards/storage/test_silver.py`

### Background

`_build_scryfall_base` currently contains inline SQL with `json_extract_string(prices, '$.eur')`. After the migration, `eur`, `eur_foil`, `usd`, `usd_foil` are direct FLOAT columns. The SQL moves to a `.sql` file and the query simplifies. Tests that create `bronze_scryfall_prices_history` fixtures must drop the `prices` JSON column and use scalar columns.

- [ ] **Step 9.1: Update all Scryfall price history fixtures in `test_silver.py`**

Every `pd.DataFrame` that creates a `bronze_scryfall_prices_history` fixture currently uses `"prices": [_SCRYFALL_PRICES_JSON]`. Replace ALL such occurrences. The values from `_SCRYFALL_PRICES_JSON` are: `eur=3.20`, `eur_foil=8.50`, `usd=3.50`, `usd_foil=9.00`.

The pattern to replace (shown generically — the number of rows in `id` and `snapshot_date` varies):

```python
# OLD — every fixture that creates bronze_scryfall_prices_history:
pd.DataFrame({
    "id": [...],
    "snapshot_date": [...],
    "prices": [_SCRYFALL_PRICES_JSON, ...],  # one entry per row
})

# NEW:
pd.DataFrame({
    "id": [...],
    "snapshot_date": [...],
    "eur":      [3.20, ...],   # same count as id
    "eur_foil": [8.50, ...],
    "usd":      [3.50, ...],
    "usd_foil": [9.00, ...],
})
```

Apply this change to ALL of these tests in `test_silver.py`:

1. `TestSilverPriceBuilder.test_returns_empty_dataframe_when_silver_cards_missing` (line ~524): 1 row
2. `TestSilverPriceBuilder.test_happy_path_both_sources_present` (line ~546): 1 row
3. `TestSilverPriceBuilder.test_happy_path_has_all_expected_columns` (line ~580): 1 row
4. `TestSilverPriceBuilder.test_mtgjson_missing_fills_columns_with_none` (line ~611): 1 row
5. `TestSilverPriceBuilder.test_scryfall_card_with_no_silver_match_is_dropped` (line ~629): 2 rows — both use the same scalar values
6. `TestSilverPriceBuilder.test_mtgjson_card_with_no_scryfall_history_row_is_excluded` (line ~646): 1 row for `bronze_scryfall_prices_history`
7. `TestSilverPriceBuilder.test_build_ignores_bronze_rows_from_other_dates` (line ~674): 2 rows (dates `2026-05-10` and `2026-05-11`)
8. `TestSilverPriceBuilder.test_english_card_with_stale_scryfall_id_uses_canonical_uuid` (line ~691): 1 row
9. `TestSilverPriceBuilder.test_non_english_canonical_uuid_card_excluded_from_main_prices` (line ~716): 2 rows
10. `TestBuildLanguagePrices.test_returns_empty_when_silver_cards_missing` (line ~1027): 1 row
11. `TestBuildLanguagePrices.test_returns_empty_when_no_language_variant_cards` (line ~1041): 1 row
12. `TestBuildLanguagePrices.test_happy_path_language_variant_gets_prices` (line ~1061): 1 row
13. `TestBuildLanguagePrices.test_has_expected_columns` (line ~1088): 1 row

After making all these changes, the `_SCRYFALL_PRICES_JSON` constant is no longer referenced by fixtures (only by `_MTGJSON_PAPER_JSON` for Silver tests). Delete the `_SCRYFALL_PRICES_JSON` constant line (line ~465-467).

Also delete `import json` from `test_silver.py` if `_SCRYFALL_PRICES_JSON` and `_MTGJSON_PAPER_JSON` were its only uses (check: `json.dumps` won't appear after removing both constants — confirm before deleting).

**Example of full updated fixture for `test_happy_path_both_sources_present`:**

```python
def test_happy_path_both_sources_present(self, tmp_path):
    scryfall_hist = pd.DataFrame(
        {
            "id": ["s1"],
            "snapshot_date": ["2026-05-11"],
            "eur": [3.20],
            "eur_foil": [8.50],
            "usd": [3.50],
            "usd_foil": [9.00],
        }
    )
    mtgjson_hist = pd.DataFrame(
        {
            "uuid": ["u1"],
            "snapshot_date": ["2026-05-11"],
            "paper": [_MTGJSON_PAPER_JSON],  # unchanged in this task
        }
    )
    with _make_storage_with_bronze(
        tmp_path,
        {
            "bronze_scryfall_prices_history": scryfall_hist,
            "bronze_mtgjson_prices_history": mtgjson_hist,
        },
    ) as s:
        _seed_silver_cards(s, [("u1", "s1")])
        result = s._prices.build("2026-05-11")

        assert len(result) == 1
        row = result.iloc[0]
        assert row["uuid"] == "u1"
        assert row["scryfall_id"] == "s1"
        assert row["eur"] == pytest.approx(3.20)
        assert row["cardmarket_eur"] == pytest.approx(3.20)
        assert row["cardmarket_buylist_eur"] == pytest.approx(1.80)
        assert row["tcgplayer_usd"] == pytest.approx(3.50)
```

- [ ] **Step 9.2: Run failing tests to confirm the fixture issue**

```
pytest tests/data/cards/storage/test_silver.py::TestSilverPriceBuilder -v --tb=short 2>&1 | tail -30
```

Expected: tests that call `s._prices.build()` FAIL because the Silver SQL still uses `json_extract_string(prices, ...)` but the fixture no longer has a `prices` column.

- [ ] **Step 9.3: Create `scryfall_prices_base.sql`**

Create `src/data/cards/storage/silver/sql/scryfall_prices_base.sql`:

```sql
SELECT
    id              AS scryfall_id,
    snapshot_date,
    eur,
    eur_foil,
    usd,
    usd_foil
FROM bronze_scryfall_prices_history
WHERE snapshot_date = ?
```

- [ ] **Step 9.4: Update `_build_scryfall_base` in `prices.py`**

Add at the module level of `prices.py` (after the existing imports):

```python
from pathlib import Path

_SQL_DIR = Path(__file__).parent / "sql"
```

Replace the inline SQL block in `_build_scryfall_base`:

```python
def _build_scryfall_base(self, today: str) -> pd.DataFrame:
    """Read today's Scryfall price snapshot and join to silver_cards for UUID.

    Filters bronze_scryfall_prices_history to snapshot_date = today so that
    only a single day's rows are processed instead of the full history table.

    UUID resolution: COALESCE(uuid, canonical_uuid) captures English paper
    cards whose scryfall_id→MTGJson direct join missed (uuid=NULL) but whose
    (set_code, collector_number) fallback resolved a canonical_uuid. The most
    common cause is MTGJson holding a stale scryfall_id after Scryfall
    reissues a card identifier. The language='English' guard prevents
    non-English language variants from creating duplicate price rows.

    Args:
        today: ISO date string; only rows with this snapshot_date are read.
    """
    card_map = self._silver_con.execute(
        "SELECT COALESCE(uuid, canonical_uuid) AS uuid, scryfall_id"
        " FROM silver_cards"
        " WHERE scryfall_id IS NOT NULL"
        "   AND COALESCE(uuid, canonical_uuid) IS NOT NULL"
        "   AND (uuid IS NOT NULL OR language = 'English')"
    ).df()

    sql = (_SQL_DIR / "scryfall_prices_base.sql").read_text(encoding="utf-8")
    scryfall_prices = self._bronze_con.execute(sql, [today]).df()

    return scryfall_prices.merge(card_map, on="scryfall_id", how="inner")
```

- [ ] **Step 9.5: Run to verify PASS**

```
pytest tests/data/cards/storage/test_silver.py::TestSilverPriceBuilder -v
```

Expected: all pass (MTGJson tests may still fail if `paper` column is expected — those are fixed in Task 10).

- [ ] **Step 9.6: Commit**

```bash
git add src/data/cards/storage/silver/sql/scryfall_prices_base.sql src/data/cards/storage/silver/prices.py tests/data/cards/storage/test_silver.py
git commit -m "feat: simplify _build_scryfall_base to read scalar Scryfall price columns"
```

---

## Task 10: Silver — `_join_mtgjson_prices` → SQL file

**Files:**
- Create: `src/data/cards/storage/silver/sql/mtgjson_prices_daily.sql`
- Modify: `src/data/cards/storage/silver/prices.py`
- Modify: `tests/data/cards/storage/test_silver.py`

### Background

`_join_mtgjson_prices` currently reads `paper` JSON from Bronze, runs a Python list comprehension to parse each row, then `pd.concat`. After this task it runs a DuckDB SELECT on scalar columns and does a pandas LEFT merge.

- [ ] **Step 10.1: Update MTGJson price history fixtures in `test_silver.py`**

Find all `pd.DataFrame` calls that create a `bronze_mtgjson_prices_history` fixture with a `"paper"` column. The values from `_MTGJSON_PAPER_JSON` are: `cardmarket_eur=3.20`, `cardmarket_eur_foil=8.50`, `cardmarket_buylist_eur=1.80`, `tcgplayer_usd=3.50`, `tcgplayer_usd_foil=9.00`, `tcgplayer_buylist_usd=2.10`.

Apply this pattern replacement:

```python
# OLD:
pd.DataFrame({
    "uuid": ["u1"],
    "snapshot_date": ["2026-05-11"],
    "paper": [_MTGJSON_PAPER_JSON],
})

# NEW:
pd.DataFrame({
    "uuid": ["u1"],
    "snapshot_date": ["2026-05-11"],
    "cardmarket_eur":         [3.20],
    "cardmarket_eur_foil":    [8.50],
    "cardmarket_buylist_eur": [1.80],
    "tcgplayer_usd":          [3.50],
    "tcgplayer_usd_foil":     [9.00],
    "tcgplayer_buylist_usd":  [2.10],
})
```

Apply to all occurrences:
1. `TestSilverPriceBuilder.test_happy_path_both_sources_present` — 1 mtgjson row
2. `TestSilverPriceBuilder.test_mtgjson_card_with_no_scryfall_history_row_is_excluded` — 2 mtgjson rows (both use same scalar values)

After applying, the `_MTGJSON_PAPER_JSON` constant is no longer referenced. Delete it (lines ~469-480). Delete `import json` if now unused.

- [ ] **Step 10.2: Run to verify FAIL**

```
pytest tests/data/cards/storage/test_silver.py::TestSilverPriceBuilder::test_happy_path_both_sources_present -v
```

Expected: FAIL — `_join_mtgjson_prices` still tries to read `paper` column which no longer exists in fixture.

- [ ] **Step 10.3: Create `mtgjson_prices_daily.sql`**

Create `src/data/cards/storage/silver/sql/mtgjson_prices_daily.sql`:

```sql
SELECT
    uuid,
    snapshot_date,
    cardmarket_eur,
    cardmarket_eur_foil,
    cardmarket_buylist_eur,
    tcgplayer_usd,
    tcgplayer_usd_foil,
    tcgplayer_buylist_usd
FROM bronze_mtgjson_prices_history
WHERE snapshot_date = ?
```

- [ ] **Step 10.4: Update `_join_mtgjson_prices` in `prices.py`**

Replace the entire method body:

```python
def _join_mtgjson_prices(
    self, df: pd.DataFrame, bronze_tables: set[str], today: str
) -> pd.DataFrame:
    """Join today's MTGJson paper prices onto the Scryfall base DataFrame.

    Reads scalar FLOAT columns directly from bronze_mtgjson_prices_history
    for snapshot_date = today. Left-joined to the Scryfall base on
    (uuid, snapshot_date).

    Args:
        df: Scryfall base DataFrame (uuid, scryfall_id, snapshot_date, …).
        bronze_tables: Set of table names present in Bronze DuckDB.
        today: ISO date string used to filter bronze_mtgjson_prices_history.
    """
    mtgjson_cols = self._PRICE_COLS[4:]  # cardmarket_eur … tcgplayer_buylist_usd
    if "bronze_mtgjson_prices_history" not in bronze_tables:
        logger.warning(
            "bronze_mtgjson_prices_history not found — MTGJson prices omitted"
        )
        for col in mtgjson_cols:
            df[col] = None
        return df

    sql = (_SQL_DIR / "mtgjson_prices_daily.sql").read_text(encoding="utf-8")
    mtgjson = self._bronze_con.execute(sql, [today]).df()

    for col in mtgjson_cols:
        if col not in mtgjson.columns:
            mtgjson[col] = None

    return df.merge(mtgjson, on=["uuid", "snapshot_date"], how="left")
```

- [ ] **Step 10.5: Run to verify PASS**

```
pytest tests/data/cards/storage/test_silver.py::TestSilverPriceBuilder -v
```

Expected: all PASS.

- [ ] **Step 10.6: Commit**

```bash
git add src/data/cards/storage/silver/sql/mtgjson_prices_daily.sql src/data/cards/storage/silver/prices.py tests/data/cards/storage/test_silver.py
git commit -m "feat: replace _join_mtgjson_prices JSON parsing with DuckDB scalar column read"
```

---

## Task 11: Silver — `build_language_prices` → SQL file

**Files:**
- Create: `src/data/cards/storage/silver/sql/scryfall_language_prices_base.sql`
- Modify: `src/data/cards/storage/silver/prices.py`
- Modify: `tests/data/cards/storage/test_silver.py`

### Background

`build_language_prices` has the same inline `json_extract_string(prices, '$.eur')` query as `_build_scryfall_base`. After the migration it reads scalar columns. The Scryfall history fixtures for language price tests already have scalar columns (changed in Task 9), so only the SQL file and the method need updating here.

- [ ] **Step 11.1: Run to verify language prices tests currently FAIL**

```
pytest tests/data/cards/storage/test_silver.py::TestBuildLanguagePrices -v --tb=short
```

Expected: tests FAIL because fixtures now have scalar columns but `build_language_prices` still uses `json_extract_string(prices, ...)`.

- [ ] **Step 11.2: Create `scryfall_language_prices_base.sql`**

Create `src/data/cards/storage/silver/sql/scryfall_language_prices_base.sql`:

```sql
SELECT
    id              AS scryfall_id,
    snapshot_date,
    eur,
    eur_foil,
    usd,
    usd_foil
FROM bronze_scryfall_prices_history
WHERE snapshot_date = ?
```

- [ ] **Step 11.3: Update `build_language_prices` in `prices.py`**

Replace the inline SQL block (the second `self._bronze_con.execute("""...""")` in `build_language_prices`):

```python
sql = (_SQL_DIR / "scryfall_language_prices_base.sql").read_text(encoding="utf-8")
scryfall_prices = self._bronze_con.execute(sql, [today]).df()
```

The rest of the method is unchanged.

- [ ] **Step 11.4: Run to verify PASS**

```
pytest tests/data/cards/storage/test_silver.py::TestBuildLanguagePrices -v
```

Expected: all PASS.

- [ ] **Step 11.5: Commit**

```bash
git add src/data/cards/storage/silver/sql/scryfall_language_prices_base.sql src/data/cards/storage/silver/prices.py
git commit -m "feat: simplify build_language_prices to read scalar Scryfall price columns"
```

---

## Task 12: Silver — Remove dead code

**Files:**
- Modify: `src/data/cards/storage/silver/prices.py`

### Background

`_extract_all_prices`, `_MTGJSON_PRICE_MAP` (class attribute), and `import json` are no longer called or referenced. Removing them keeps the file clean.

- [ ] **Step 12.1: Run the full Silver test suite to establish a baseline**

```
pytest tests/data/cards/storage/test_silver.py -v --tb=short 2>&1 | tail -10
```

Expected: all PASS. Confirm before making deletions.

- [ ] **Step 12.2: Remove dead code from `prices.py`**

Make the following deletions:

1. Delete `import json` (line 3).
2. Delete `_MTGJSON_PRICE_MAP` class attribute (lines ~36–43: the dict definition plus the comment block above it).
3. Delete `_extract_all_prices` static method (the entire method definition, lines ~387–415).

- [ ] **Step 12.3: Run to verify nothing broke**

```
pytest tests/data/cards/storage/test_silver.py -v
```

Expected: all PASS.

- [ ] **Step 12.4: Run the full test suite**

```
pytest tests/ -v --tb=short 2>&1 | tail -30
```

Expected: no failures.

- [ ] **Step 12.5: Commit**

```bash
git add src/data/cards/storage/silver/prices.py
git commit -m "refactor: remove dead Silver code (_extract_all_prices, _MTGJSON_PRICE_MAP, import json)"
```

---

## Task 13: Health Check — `_check_bronze_prices_coverage`

**Files:**
- Modify: `src/data/cards/storage/health.py`
- Modify: `tests/data/cards/storage/test_health.py`

### Background

The new scalar columns enable a meaningful coverage check: verify that at least one card has non-NULL prices for today in each Bronze price table. If the scalar extraction failed silently, this check catches it.

- [ ] **Step 13.1: Write the failing tests**

Add `_check_bronze_prices_coverage` to the import in `tests/data/cards/storage/test_health.py`:

```python
from src.data.cards.storage.health import (
    CheckResult,
    _check_table_has_rows,
    _check_snapshot_date_today,
    _check_no_nulls,
    _check_no_duplicate_canonical_uuid,
    _check_oracle_id_conflicts,
    _check_silver_prices_no_negative_eur,
    _check_gold_ml_dataset_has_target,
    _check_bronze_prices_coverage,
    run_health_checks,
)
```

Add this test class to the test file:

```python
class TestCheckBronzePricesCoverage:
    TODAY = datetime.date(2026, 6, 24)

    def _make_scryfall_hist(
        self,
        con: duckdb.DuckDBPyConnection,
        eur_val: float | None,
    ) -> None:
        con.execute("""
            CREATE TABLE bronze_scryfall_prices_history (
                id VARCHAR, snapshot_date DATE, eur FLOAT, eur_foil FLOAT,
                usd FLOAT, usd_foil FLOAT
            )
        """)
        con.execute(
            "INSERT INTO bronze_scryfall_prices_history VALUES (?, ?, ?, NULL, NULL, NULL)",
            ["s1", self.TODAY, eur_val],
        )

    def _make_mtgjson_hist(
        self,
        con: duckdb.DuckDBPyConnection,
        cardmarket_eur: float | None,
    ) -> None:
        con.execute("""
            CREATE TABLE bronze_mtgjson_prices_history (
                uuid VARCHAR, snapshot_date DATE,
                cardmarket_eur FLOAT, cardmarket_eur_foil FLOAT,
                cardmarket_buylist_eur FLOAT, tcgplayer_usd FLOAT,
                tcgplayer_usd_foil FLOAT, tcgplayer_buylist_usd FLOAT
            )
        """)
        con.execute(
            "INSERT INTO bronze_mtgjson_prices_history VALUES"
            " (?, ?, ?, NULL, NULL, NULL, NULL, NULL)",
            ["u1", self.TODAY, cardmarket_eur],
        )

    def test_pass_when_both_tables_have_prices_today(self):
        con = duckdb.connect(":memory:")
        self._make_scryfall_hist(con, 3.20)
        self._make_mtgjson_hist(con, 3.20)
        result = _check_bronze_prices_coverage(con, self.TODAY)
        assert result.status == "PASS"
        con.close()

    def test_fail_when_scryfall_eur_all_null_today(self):
        con = duckdb.connect(":memory:")
        self._make_scryfall_hist(con, None)
        self._make_mtgjson_hist(con, 3.20)
        result = _check_bronze_prices_coverage(con, self.TODAY)
        assert result.status == "FAIL"
        assert "scryfall" in result.detail.lower()
        con.close()

    def test_fail_when_mtgjson_cardmarket_eur_all_null_today(self):
        con = duckdb.connect(":memory:")
        self._make_scryfall_hist(con, 3.20)
        self._make_mtgjson_hist(con, None)
        result = _check_bronze_prices_coverage(con, self.TODAY)
        assert result.status == "FAIL"
        assert "mtgjson" in result.detail.lower()
        con.close()

    def test_fail_when_no_rows_for_today(self):
        con = duckdb.connect(":memory:")
        con.execute("""
            CREATE TABLE bronze_scryfall_prices_history (
                id VARCHAR, snapshot_date DATE, eur FLOAT, eur_foil FLOAT,
                usd FLOAT, usd_foil FLOAT
            )
        """)
        con.execute("""
            CREATE TABLE bronze_mtgjson_prices_history (
                uuid VARCHAR, snapshot_date DATE,
                cardmarket_eur FLOAT, cardmarket_eur_foil FLOAT,
                cardmarket_buylist_eur FLOAT, tcgplayer_usd FLOAT,
                tcgplayer_usd_foil FLOAT, tcgplayer_buylist_usd FLOAT
            )
        """)
        result = _check_bronze_prices_coverage(con, self.TODAY)
        assert result.status == "FAIL"
        con.close()
```

- [ ] **Step 13.2: Run to verify FAIL**

```
pytest tests/data/cards/storage/test_health.py::TestCheckBronzePricesCoverage -v
```

Expected: `ImportError` — `_check_bronze_prices_coverage` not defined.

- [ ] **Step 13.3: Add `_check_bronze_prices_coverage` to `health.py`**

Add this function after `_check_silver_prices_no_negative_eur`:

```python
def _check_bronze_prices_coverage(
    con: duckdb.DuckDBPyConnection, today: datetime.date
) -> CheckResult:
    scryfall_count: int = con.execute(
        "SELECT COUNT(*) FROM bronze_scryfall_prices_history"
        " WHERE snapshot_date = ? AND eur IS NOT NULL",
        [today],
    ).fetchone()[0]  # type: ignore[index]
    if scryfall_count == 0:
        return CheckResult(
            "bronze prices coverage",
            "bronze",
            "FAIL",
            f"no scryfall eur prices for {today}",
        )

    mtgjson_count: int = con.execute(
        "SELECT COUNT(*) FROM bronze_mtgjson_prices_history"
        " WHERE snapshot_date = ? AND cardmarket_eur IS NOT NULL",
        [today],
    ).fetchone()[0]  # type: ignore[index]
    if mtgjson_count == 0:
        return CheckResult(
            "bronze prices coverage",
            "bronze",
            "FAIL",
            f"no mtgjson cardmarket_eur prices for {today}",
        )

    return CheckResult(
        "bronze prices coverage",
        "bronze",
        "PASS",
        f"{scryfall_count} scryfall / {mtgjson_count} mtgjson rows with prices for {today}",
    )
```

Wire it into `run_health_checks`. After the `bronze_structure` block (currently after `results.extend(bronze_structure)`), add:

```python
if all(r.status == "PASS" for r in bronze_structure):
    results.append(_check_bronze_prices_coverage(bronze_con, today))
```

- [ ] **Step 13.4: Run to verify PASS**

```
pytest tests/data/cards/storage/test_health.py -v
```

Expected: all PASS.

- [ ] **Step 13.5: Commit**

```bash
git add src/data/cards/storage/health.py tests/data/cards/storage/test_health.py
git commit -m "feat: add _check_bronze_prices_coverage health check"
```

---

## Task 14: Documentation

**Files:**
- Modify: `docs/adr/ADR-003-medallion-architecture.md`
- Modify: `docs/adr/ADR-012-physical-cards-only.md`
- Create: `docs/adr/ADR-025-scalar-bronze-prices.md`
- Modify: `docs/architecture/c4/bronze-storage.md`
- Modify: `docs/architecture/c4/silver-storage.md`
- Modify: `docs/architecture/data/table-schemas.md`
- Modify: `docs/architecture/data/data-lineage.md`
- Modify: `docs/architecture/data/glossary.md`

- [ ] **Step 14.1: Create ADR-025**

Create `docs/adr/ADR-025-scalar-bronze-prices.md`:

```markdown
# ADR-025: Scalar Price Columns in Bronze Price History Tables

**Date:** 2026-06-24
**Status:** Accepted

## Context

`bronze_scryfall_prices_history` stored prices as a `prices` VARCHAR JSON column
(`{"eur": "1.50", ...}`). `bronze_mtgjson_prices_history` stored prices as a `paper`
VARCHAR JSON column with nested structure `{retailer → tx_type → finish → date → price}`.

Silver's `SilverPriceBuilder` was forced to:
1. Parse Scryfall prices with `json_extract_string` inside DuckDB SQL.
2. Parse MTGJson `paper` JSON in a Python list comprehension — one `json.loads` call per
   row — violating ADR-024 (DuckDB as compute layer).

## Decision

Scalarise both Bronze price history tables at ingestion time:

- `bronze_scryfall_prices_history`: replace `prices` VARCHAR with `eur FLOAT`,
  `eur_foil FLOAT`, `usd FLOAT`, `usd_foil FLOAT`.
- `bronze_mtgjson_prices_history`: replace `paper` VARCHAR and `mtgo` VARCHAR with
  `cardmarket_eur FLOAT`, `cardmarket_eur_foil FLOAT`, `cardmarket_buylist_eur FLOAT`,
  `tcgplayer_usd FLOAT`, `tcgplayer_usd_foil FLOAT`, `tcgplayer_buylist_usd FLOAT`.

Price extraction logic (`_extract_mtgjson_scalar_prices`) runs once at Bronze ingestion.
Silver reads scalar columns directly — no JSON parsing remains in the Silver layer.

## Consequences

- Silver `SilverPriceBuilder` queries simplify; `_extract_all_prices` and the Python
  `pd.concat` step are deleted.
- `tix` (MTGO digital) and `mtgo` columns are not preserved — explicitly excluded per
  ADR-012.
- Existing data migrated from `cards_copy.duckdb` via `scripts/migrate_bronze_prices.py`.
- `_MTGJSON_PRICE_MAP` moved from Silver to Bronze as the extraction mapping is now a
  Bronze concern.
```

- [ ] **Step 14.2: Update `ADR-003-medallion-architecture.md`**

Find the section describing `bronze_mtgjson_prices_history` and `bronze_scryfall_prices_history` schemas. Update the column descriptions to reflect scalar columns (remove `paper VARCHAR`, `mtgo VARCHAR`, `prices VARCHAR`; add the scalar columns listed in the new schemas above).

- [ ] **Step 14.3: Update `ADR-012-physical-cards-only.md`**

Add a note that the `mtgo` column has been removed from `bronze_mtgjson_prices_history` as a direct consequence of this decision (physical-cards-only constraint now enforced at the schema level).

- [ ] **Step 14.4: Update `docs/architecture/c4/bronze-storage.md`**

Find the description of `_snapshot` for the `mtgjson_prices` and `scryfall` sources. Update to describe `_snapshot_scryfall_prices` and `_snapshot_mtgjson_prices` as dedicated methods that extract scalar prices at write time.

- [ ] **Step 14.5: Update `docs/architecture/c4/silver-storage.md`**

Find Step 6 (or the section describing how Silver reads Bronze prices). Update "extracts prices from JSON" or similar wording to "reads scalar FLOAT price columns directly from Bronze".

- [ ] **Step 14.6: Update `docs/architecture/data/table-schemas.md`**

Replace the schema tables for `bronze_mtgjson_prices_history` and `bronze_scryfall_prices_history` with the new scalar column schemas (as documented in the design spec at `docs/superpowers/specs/2026-06-24-scalar-bronze-prices-design.md`).

- [ ] **Step 14.7: Update `docs/architecture/data/data-lineage.md`**

Find any steps describing "Bronze snapshots Scryfall prices JSON" or "Silver parses paper JSON". Update to: Bronze writes scalar prices at ingestion; Silver reads scalar FLOAT columns.

- [ ] **Step 14.8: Update `docs/architecture/data/glossary.md`**

Find the "Price snapshot" entry (or similar). Remove any reference to JSON price blobs. Clarify that price snapshots are rows with scalar FLOAT price columns, one per (id/uuid, snapshot_date).

- [ ] **Step 14.9: Commit docs**

```bash
git add -f docs/adr/ADR-025-scalar-bronze-prices.md docs/adr/ADR-003-medallion-architecture.md docs/adr/ADR-012-physical-cards-only.md docs/architecture/c4/bronze-storage.md docs/architecture/c4/silver-storage.md docs/architecture/data/table-schemas.md docs/architecture/data/data-lineage.md docs/architecture/data/glossary.md
git commit -m "docs: document scalar Bronze price schema in ADRs and architecture docs"
```

> Note: `git add -f` is required because `docs/superpowers/` is in `.gitignore`. The architecture docs under `docs/architecture/` and `docs/adr/` may or may not require `-f` — check `.gitignore` first and add normally if not excluded.

---

## Self-Review

Checked against the design spec `docs/superpowers/specs/2026-06-24-scalar-bronze-prices-design.md`:

| Spec requirement | Covered in task |
|-----------------|----------------|
| `bronze_scryfall_prices_history` → scalar columns | Tasks 4, 9, 11 |
| `bronze_mtgjson_prices_history` → scalar columns | Tasks 1, 2, 3, 10 |
| Migration script from `cards_copy.duckdb` | Task 7, 8 |
| `_MTGJSON_PRICE_MAP` moves to Bronze | Task 1 |
| `_filter_prices_to_date` deleted | Task 6 |
| `_snapshot_scryfall_prices` added | Task 4 |
| `_snapshot_mtgjson_prices` added | Task 3 |
| `STORAGE_CONFIG` updated | Task 5 |
| `daily_update` + `populate` wired | Task 5 |
| Silver `_build_scryfall_base` → SQL | Task 9 |
| Silver `_join_mtgjson_prices` → SQL | Task 10 |
| Silver `build_language_prices` → SQL | Task 11 |
| Dead Silver code removed | Task 12 |
| `_check_bronze_prices_coverage` | Task 13 |
| Health check wired into `run_health_checks` | Task 13 |
| `TestFilterPricesToDate` deleted | Task 6 |
| `test_paper_filtered_to_snapshot_date_only` deleted | Task 2 |
| `test_dates_collected_from_mtgo_platform` deleted | Task 2 |
| Snapshot count test updated from 4 → 2 | Task 5 |
| Silver fixtures use scalar columns | Tasks 9, 10, 11 |
| ADR-025 created | Task 14 |
| Architecture docs updated | Task 14 |
| `mtgo` column not preserved (ADR-012) | Task 2, Task 7 |
| Gold layer untouched | (no task needed) |
| `_PRICE_COLS` unchanged in Silver | Task 10 confirms `[4:]` slice |

No gaps found.
