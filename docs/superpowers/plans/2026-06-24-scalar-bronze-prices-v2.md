# Scalar Bronze Price Tables V2: EAV + Tix

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor Bronze MTGJson prices to EAV schema (no data loss, no semantic selection in Bronze); add `tix` to Scryfall Bronze; add WARN health check status with schema drift detection.

**Architecture:** Bronze = raw normalization only. MTGJson: EAV `(uuid, snapshot_date, retailer, tx_type, finish, price)` — one row per price point, all retailers captured, schema never changes. Scryfall: scalar columns including `tix`. Silver = semantic selection only: CASE WHEN pivot from EAV to the 6 wide columns downstream consumers need. Health checks detect drift between Bronze EAV and Silver map.

**Tech Stack:** DuckDB, pandas, Python, pytest

---

## Context: What Was Already Done (V1 Plan)

Tasks 1–7 of `2026-06-24-scalar-bronze-prices.md` are committed. They implemented WIDE-column Bronze extraction (6 pre-selected columns from `_MTGJSON_PRICE_MAP`). This plan REPLACES that approach.

**Do NOT run the Task 8 manual migration yet.** The migration script changes in Task R5.

Tasks 5 and 6 (STORAGE_CONFIG wiring, `_filter_prices_to_date` deletion) are unaffected and do not need to be redone.

---

## File Map

| Action | Path |
|--------|------|
| Modify | `src/data/cards/storage/base/writers.py` |
| Modify | `src/data/cards/storage/bronze/storage.py` |
| Modify | `src/data/cards/storage/silver/prices.py` |
| Create | `src/data/cards/storage/silver/sql/scryfall_prices_base.sql` |
| Create | `src/data/cards/storage/silver/sql/mtgjson_prices_daily.sql` |
| Create | `src/data/cards/storage/silver/sql/scryfall_language_prices_base.sql` |
| Modify | `src/data/cards/storage/health.py` |
| Modify | `scripts/migrate_bronze_prices.py` |
| Modify | `tests/data/cards/storage/test_bronze.py` |
| Modify | `tests/data/cards/storage/test_silver.py` |
| Modify | `tests/data/cards/storage/test_health.py` |
| Modify | `tests/cards/test_storage.py` |
| Modify | `tests/scripts/test_migrate_bronze_prices.py` |
| Modify | docs files |

---

## Task R0: `DuckDBWriter.append` — composite key support

**Files:**
- Modify: `src/data/cards/storage/base/writers.py`
- Modify: `tests/data/cards/storage/test_writers.py` (or wherever writer tests live)

### Background

`DuckDBWriter.append` currently accepts `key_column: str` and deduplicates on `(key_column, snapshot_date)`:

```python
f"LEFT JOIN {table_name} t "
f"  ON t.{key_column} = s.{key_column} "
f"  AND t.snapshot_date = s.snapshot_date "
f"WHERE t.{key_column} IS NULL"
```

EAV rows require deduplication on `(uuid, snapshot_date, retailer, tx_type, finish)`. Extend the signature to `key_column: str | list[str]`. When a list is passed, all listed columns plus `snapshot_date` form the composite dedup key.

- [ ] **Step R0.1: Locate existing writer tests**

```
find tests/ -name "*.py" | xargs grep -l "DuckDBWriter\|test_writer" 2>/dev/null
```

- [ ] **Step R0.2: Write failing test for composite key dedup**

In the writer test file, add:

```python
def test_append_composite_key_deduplicates_on_all_columns():
    con = duckdb.connect(":memory:")
    writer = DuckDBWriter(con)

    df1 = pd.DataFrame([
        {"uuid": "u1", "snapshot_date": "2026-06-24", "retailer": "cardmarket",
         "tx_type": "retail", "finish": "normal", "price": 3.20},
    ])
    writer.append(df1, "t", ["uuid", "retailer", "tx_type", "finish"])

    # Same composite key — must be skipped
    writer.append(df1, "t", ["uuid", "retailer", "tx_type", "finish"])
    count = con.execute("SELECT count(*) FROM t").fetchone()[0]
    assert count == 1

def test_append_composite_key_allows_different_finish():
    con = duckdb.connect(":memory:")
    writer = DuckDBWriter(con)

    row_normal = pd.DataFrame([
        {"uuid": "u1", "snapshot_date": "2026-06-24", "retailer": "cardmarket",
         "tx_type": "retail", "finish": "normal", "price": 3.20},
    ])
    row_foil = pd.DataFrame([
        {"uuid": "u1", "snapshot_date": "2026-06-24", "retailer": "cardmarket",
         "tx_type": "retail", "finish": "foil", "price": 8.50},
    ])
    writer.append(row_normal, "t", ["uuid", "retailer", "tx_type", "finish"])
    writer.append(row_foil,   "t", ["uuid", "retailer", "tx_type", "finish"])
    count = con.execute("SELECT count(*) FROM t").fetchone()[0]
    assert count == 2
```

- [ ] **Step R0.3: Run to verify FAIL**

```
pytest tests/ -k "composite_key" -v
```

Expected: `TypeError` — `append` does not accept a list.

- [ ] **Step R0.4: Extend `DuckDBWriter.append`**

Change signature and SQL generation:

```python
def append(
    self,
    df: pd.DataFrame,
    table_name: str,
    key_column: str | list[str],
) -> None:
    """Append rows to a history table, skipping already-snapshotted pairs.

    Deduplication key: (key_column, snapshot_date) when key_column is a str;
    (col1, col2, …, snapshot_date) when key_column is a list.
    History tables accumulate one snapshot per day and must never lose rows.

    Args:
        df: DataFrame to append.
        table_name: Target history table name.
        key_column: Column name(s) forming the composite dedup key together
            with snapshot_date.
    """
    if df.empty:
        logger.warning("No data to append into %r — skipping", table_name)
        return

    key_cols = [key_column] if isinstance(key_column, str) else key_column
    join_conditions = " AND ".join(
        [f"t.snapshot_date = s.snapshot_date"]
        + [f"t.{col} = s.{col}" for col in key_cols]
    )
    null_check = f"t.{key_cols[0]} IS NULL"

    staging = self._serialize(df)
    self._con.register("_staging", staging)
    try:
        if not self._table_exists(table_name):
            self._con.execute(
                f"CREATE TABLE {table_name} AS SELECT * FROM _staging"
            )
            logger.info(
                "Created history table %r with %d rows", table_name, len(df)
            )
        else:
            self._con.execute(
                f"INSERT INTO {table_name} "
                f"SELECT s.* FROM _staging s "
                f"LEFT JOIN {table_name} t ON {join_conditions} "
                f"WHERE {null_check}"
            )
            logger.info("Appended %d rows into %r", len(df), table_name)
    except duckdb.Error as e:
        raise StorageWriteError(f"Failed to append into {table_name!r}: {e}") from e
    finally:
        self._con.unregister("_staging")
```

- [ ] **Step R0.5: Run to verify PASS**

```
pytest tests/ -k "composite_key or append" -v
```

- [ ] **Step R0.6: Run full suite**

```
pytest tests/ -x --tb=short -q 2>&1 | tail -10
```

Expected: all pass — existing callers pass `str`, still works.

- [ ] **Step R0.7: Commit**

```bash
git add src/data/cards/storage/base/writers.py tests/
git commit -m "feat: DuckDBWriter.append supports composite key_column list"
```

---

## Task R1: Bronze — EAV extraction (replaces Task 1 wide columns)

**Files:**
- Modify: `src/data/cards/storage/bronze/storage.py`
- Modify: `tests/data/cards/storage/test_bronze.py`

### Background

Remove `_MTGJSON_PRICE_MAP` and `_extract_mtgjson_scalar_prices` from Bronze entirely. Add `_extract_paper_eav_rows` which iterates every `(retailer, tx_type, finish)` in `paper_dict` — no pre-selection — and emits one dict per price using look-back semantics.

- [ ] **Step R1.1: Update import and replace `TestExtractMtgjsonScalarPrices` with `TestExtractPaperEavRows`**

In `tests/data/cards/storage/test_bronze.py`:

Change import (remove `_extract_mtgjson_scalar_prices` and `_MTGJSON_PRICE_MAP`, add `_extract_paper_eav_rows`):

```python
from src.data.cards.storage.bronze.storage import (
    _extract_paper_eav_rows,
    _records_to_df,
)
```

Delete the entire `TestExtractMtgjsonScalarPrices` class (10 tests).

Add `TestExtractPaperEavRows` after `TestToDF`:

```python
class TestExtractPaperEavRows:
    def test_returns_empty_for_none(self):
        assert _extract_paper_eav_rows(None, "u1", "2026-05-11") == []

    def test_returns_empty_for_empty_dict(self):
        assert _extract_paper_eav_rows({}, "u1", "2026-05-11") == []

    def test_emits_one_row_per_price_point(self):
        paper = {"cardmarket": {"retail": {"normal": {"2026-05-11": 3.20}}}}
        rows = _extract_paper_eav_rows(paper, "u1", "2026-05-11")
        assert len(rows) == 1
        assert rows[0] == {
            "uuid": "u1",
            "snapshot_date": "2026-05-11",
            "retailer": "cardmarket",
            "tx_type": "retail",
            "finish": "normal",
            "price": pytest.approx(3.20),
        }

    def test_captures_all_retailers_including_cardkingdom(self):
        paper = {
            "cardmarket":  {"retail": {"normal": {"2026-05-11": 3.20}}},
            "tcgplayer":   {"retail": {"normal": {"2026-05-11": 3.50}}},
            "cardkingdom": {"retail": {"normal": {"2026-05-11": 4.00}}},
        }
        rows = _extract_paper_eav_rows(paper, "u1", "2026-05-11")
        retailers = {r["retailer"] for r in rows}
        assert retailers == {"cardmarket", "tcgplayer", "cardkingdom"}

    def test_lookback_selects_max_date_leq_target(self):
        paper = {"cardmarket": {"retail": {"normal": {"2026-05-10": 1.0, "2026-05-11": 3.20}}}}
        rows = _extract_paper_eav_rows(paper, "u1", "2026-05-11")
        assert rows[0]["price"] == pytest.approx(3.20)

    def test_excludes_dates_after_target(self):
        paper = {"cardmarket": {"retail": {"normal": {"2026-05-12": 5.00, "2026-05-10": 1.0}}}}
        rows = _extract_paper_eav_rows(paper, "u1", "2026-05-11")
        assert len(rows) == 1
        assert rows[0]["price"] == pytest.approx(1.0)

    def test_no_date_leq_target_emits_no_row(self):
        paper = {"cardmarket": {"retail": {"normal": {"2026-05-12": 5.00}}}}
        assert _extract_paper_eav_rows(paper, "u1", "2026-05-11") == []

    def test_captures_all_tx_types(self):
        paper = {
            "cardmarket": {
                "retail":  {"normal": {"2026-05-11": 3.20}},
                "buylist": {"normal": {"2026-05-11": 1.80}},
            }
        }
        rows = _extract_paper_eav_rows(paper, "u1", "2026-05-11")
        assert {r["tx_type"] for r in rows} == {"retail", "buylist"}

    def test_captures_all_finishes_including_etched(self):
        paper = {
            "cardmarket": {
                "retail": {
                    "normal": {"2026-05-11": 3.20},
                    "foil":   {"2026-05-11": 8.50},
                    "etched": {"2026-05-11": 12.00},
                }
            }
        }
        rows = _extract_paper_eav_rows(paper, "u1", "2026-05-11")
        assert {r["finish"] for r in rows} == {"normal", "foil", "etched"}

    def test_price_is_float_not_str(self):
        paper = {"cardmarket": {"retail": {"normal": {"2026-05-11": "3.20"}}}}
        rows = _extract_paper_eav_rows(paper, "u1", "2026-05-11")
        assert isinstance(rows[0]["price"], float)
```

- [ ] **Step R1.2: Run to verify FAIL**

```
pytest tests/data/cards/storage/test_bronze.py::TestExtractPaperEavRows -v
```

Expected: `ImportError` — `_extract_paper_eav_rows` not defined.

- [ ] **Step R1.3: Replace `_MTGJSON_PRICE_MAP` + `_extract_mtgjson_scalar_prices` with `_extract_paper_eav_rows` in `storage.py`**

Remove both existing definitions. Add:

```python
def _extract_paper_eav_rows(
    paper_dict: dict | None, uuid: str, snapshot_date: str
) -> list[dict]:
    """Extract EAV rows from an MTGJson paper dict for a given snapshot date.

    Iterates every (retailer, tx_type, finish) found in paper_dict without
    pre-selection — all retailers present in the feed are captured. Uses
    look-back semantics: selects the most recent price per combination where
    date key <= snapshot_date. Appropriate for daily snapshots.
    For historical seeding, iterate prices items directly (see seed_historical_prices).
    """
    if not paper_dict:
        return []
    rows = []
    for retailer, retailer_data in paper_dict.items():
        if not retailer_data:
            continue
        for tx_type in ("buylist", "retail"):
            listing = (retailer_data.get(tx_type)) or {}
            for finish, prices in listing.items():
                if not isinstance(prices, dict):
                    continue
                candidates = {k: v for k, v in prices.items() if k <= snapshot_date}
                if candidates:
                    rows.append({
                        "uuid": uuid,
                        "snapshot_date": snapshot_date,
                        "retailer": retailer,
                        "tx_type": tx_type,
                        "finish": finish,
                        "price": float(candidates[max(candidates)]),
                    })
    return rows
```

- [ ] **Step R1.4: Run to verify PASS**

```
pytest tests/data/cards/storage/test_bronze.py::TestExtractPaperEavRows -v
```

- [ ] **Step R1.5: Run full suite**

```
pytest tests/ -x --tb=short -q 2>&1 | tail -10
```

- [ ] **Step R1.6: Commit**

```bash
git add src/data/cards/storage/bronze/storage.py tests/data/cards/storage/test_bronze.py
git commit -m "refactor: replace wide MTGJson extraction with EAV _extract_paper_eav_rows"
```

---

## Task R2: Bronze — `seed_historical_prices` → EAV

**Files:**
- Modify: `src/data/cards/storage/bronze/storage.py`
- Modify: `tests/cards/test_storage.py`

### Background

Currently emits 6 wide columns. Must now emit EAV rows — one per `(uuid, date, retailer, tx_type, finish, price)`. For historical data (AllPrices.json), all dates in the prices dicts are valid; we iterate them directly (no look-back needed — every leaf `{date: price}` pair becomes one row).

- [ ] **Step R2.1: Update `_PAPER_PRICES` fixture to include a second retailer**

In `tests/cards/test_storage.py`, find `_PAPER_PRICES`. Update so it covers at least two retailers and two dates:

```python
_PAPER_PRICES = {
    "cardmarket": {
        "retail": {
            "normal": {"2026-04-01": 1.0, "2026-04-02": 1.1},
            "foil":   {"2026-04-01": 2.0},
        },
    },
    "cardkingdom": {
        "retail": {"normal": {"2026-04-01": 3.5}},
    },
}
```

- [ ] **Step R2.2: Replace wide-column tests with EAV tests in `TestSeedHistoricalPrices`**

Delete `test_scalar_prices_stored_for_2026_04_01` and `test_scalar_prices_stored_for_2026_04_02`.

Add:

```python
def test_eav_row_count_matches_leaf_price_entries(self, storage):
    record = _PriceRecord(uuid="uuid-1", paper=_PAPER_PRICES)
    storage.seed_historical_prices([record])
    count = storage._con.execute(
        f"SELECT count(*) FROM {self.HISTORY_TABLE}"
    ).fetchone()[0]
    # cardmarket retail normal: 2; cardmarket retail foil: 1; cardkingdom retail normal: 1
    assert count == 4

def test_eav_row_has_correct_columns(self, storage):
    record = _PriceRecord(uuid="uuid-1", paper=_PAPER_PRICES)
    storage.seed_historical_prices([record])
    row = storage._con.execute(
        f"SELECT uuid, snapshot_date, retailer, tx_type, finish, price"
        f" FROM {self.HISTORY_TABLE}"
        f" WHERE retailer='cardmarket' AND tx_type='retail'"
        f"   AND finish='normal' AND snapshot_date='2026-04-01'"
    ).fetchone()
    assert row is not None
    assert row[0] == "uuid-1"
    assert str(row[1]) == "2026-04-01"
    assert row[2] == "cardmarket"
    assert row[3] == "retail"
    assert row[4] == "normal"
    assert row[5] == pytest.approx(1.0)

def test_captures_cardkingdom_not_in_silver_map(self, storage):
    record = _PriceRecord(uuid="uuid-1", paper=_PAPER_PRICES)
    storage.seed_historical_prices([record])
    retailers = {r[0] for r in storage._con.execute(
        f"SELECT DISTINCT retailer FROM {self.HISTORY_TABLE}"
    ).fetchall()}
    assert "cardkingdom" in retailers

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

- [ ] **Step R2.3: Run to verify FAIL**

```
pytest tests/cards/test_storage.py::TestSeedHistoricalPrices -v
```

- [ ] **Step R2.4: Rewrite `seed_historical_prices`**

```python
def seed_historical_prices(self, records: list[BaseModel]) -> None:
    """One-time seeding: expand AllPrices 90-day history into EAV rows.

    Each card's paper dict contains date-keyed prices per (retailer, tx_type,
    finish). This method expands every leaf {date: price} pair into an
    individual EAV row — one row = one price point at one date. All retailers
    in the feed are captured without pre-selection.

    Already-existing (uuid, snapshot_date, retailer, tx_type, finish) rows
    are skipped, making the call idempotent.

    Args:
        records: MtgjsonCardPrices instances from AllPrices.json.
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

        for retailer, retailer_data in paper.items():
            if not retailer_data:
                continue
            for tx_type in ("buylist", "retail"):
                listing = (retailer_data.get(tx_type)) or {}
                for finish, prices in listing.items():
                    if not isinstance(prices, dict):
                        continue
                    for d, val in prices.items():
                        if val is not None:
                            rows.append({
                                "uuid": uuid_str,
                                "snapshot_date": d,
                                "retailer": retailer,
                                "tx_type": tx_type,
                                "finish": finish,
                                "price": float(val),
                            })

    if not rows:
        logger.warning("No price rows found in records — skipping seed")
        return

    DuckDBWriter(self._con).append(
        pd.DataFrame(rows),
        history_table,
        ["uuid", "retailer", "tx_type", "finish"],
    )
    logger.info("Seeded %d EAV price rows into %r", len(rows), history_table)
```

- [ ] **Step R2.5: Run to verify PASS**

```
pytest tests/cards/test_storage.py::TestSeedHistoricalPrices -v
```

- [ ] **Step R2.6: Commit**

```bash
git add src/data/cards/storage/bronze/storage.py tests/cards/test_storage.py
git commit -m "refactor: seed_historical_prices emits EAV rows, captures all retailers"
```

---

## Task R3: Bronze — `_snapshot_mtgjson_prices` → EAV

**Files:**
- Modify: `src/data/cards/storage/bronze/storage.py`
- Modify: `tests/data/cards/storage/test_bronze.py`

### Background

Replace wide-column output with EAV via `_extract_paper_eav_rows`.

- [ ] **Step R3.1: Replace `TestSnapshotMtgjsonPrices` entirely**

```python
class TestSnapshotMtgjsonPrices:
    HISTORY_TABLE = "bronze_mtgjson_prices_history"

    def test_table_has_eav_schema(self):
        with _bronze() as b:
            record = _MtgjsonPrices(
                uuid="u1",
                paper={"cardmarket": {"retail": {"normal": {"2026-06-24": 3.20}}}},
            )
            b._snapshot_mtgjson_prices([record])
            cols = {r[0] for r in b._con.execute(
                f"DESCRIBE {self.HISTORY_TABLE}"
            ).fetchall()}
        assert cols == {"uuid", "snapshot_date", "retailer", "tx_type", "finish", "price"}

    def test_emits_one_row_per_price_point(self):
        with _bronze() as b:
            record = _MtgjsonPrices(
                uuid="u1",
                paper={
                    "cardmarket": {
                        "retail":  {"normal": {"2026-06-24": 3.20}, "foil": {"2026-06-24": 8.50}},
                        "buylist": {"normal": {"2026-06-24": 1.80}},
                    },
                    "tcgplayer": {
                        "retail": {"normal": {"2026-06-24": 3.50}},
                    },
                },
            )
            b._snapshot_mtgjson_prices([record])
            count = b._con.execute(
                f"SELECT count(*) FROM {self.HISTORY_TABLE}"
            ).fetchone()[0]
        assert count == 4

    def test_captures_unlisted_retailer(self):
        with _bronze() as b:
            record = _MtgjsonPrices(
                uuid="u1",
                paper={
                    "cardmarket":  {"retail": {"normal": {"2026-06-24": 3.20}}},
                    "cardkingdom": {"retail": {"normal": {"2026-06-24": 4.00}}},
                },
            )
            b._snapshot_mtgjson_prices([record])
            retailers = {r[0] for r in b._con.execute(
                f"SELECT DISTINCT retailer FROM {self.HISTORY_TABLE}"
            ).fetchall()}
        assert "cardkingdom" in retailers

    def test_null_paper_produces_no_rows(self):
        with _bronze() as b:
            b._snapshot_mtgjson_prices([_MtgjsonPrices(uuid="u1", paper=None)])
            count = b._con.execute(
                "SELECT count(*) FROM information_schema.tables"
                f" WHERE table_name = '{self.HISTORY_TABLE}'"
            ).fetchone()[0]
        assert count == 0

    def test_idempotent_on_duplicate(self):
        with _bronze() as b:
            record = _MtgjsonPrices(
                uuid="u1",
                paper={"cardmarket": {"retail": {"normal": {"2026-06-24": 3.20}}}},
            )
            b._snapshot_mtgjson_prices([record])
            b._snapshot_mtgjson_prices([record])
            count = b._con.execute(
                f"SELECT count(*) FROM {self.HISTORY_TABLE}"
            ).fetchone()[0]
        assert count == 1

    def test_uses_today_as_snapshot_date(self):
        from datetime import date as date_cls
        with _bronze() as b:
            record = _MtgjsonPrices(
                uuid="u1",
                paper={"cardmarket": {"retail": {"normal": {"2026-06-24": 3.20}}}},
            )
            with patch("src.data.cards.storage.bronze.storage.date") as mock_date:
                mock_date.today.return_value = date_cls.fromisoformat("2026-06-24")
                b._snapshot_mtgjson_prices([record])
            snap = b._con.execute(
                f"SELECT snapshot_date FROM {self.HISTORY_TABLE}"
            ).fetchone()[0]
        assert str(snap) == "2026-06-24"

    def test_skips_when_records_empty(self):
        with _bronze() as b:
            b._snapshot_mtgjson_prices([])
            count = b._con.execute(
                "SELECT count(*) FROM information_schema.tables"
                f" WHERE table_name = '{self.HISTORY_TABLE}'"
            ).fetchone()[0]
        assert count == 0
```

- [ ] **Step R3.2: Run to verify FAIL**

```
pytest tests/data/cards/storage/test_bronze.py::TestSnapshotMtgjsonPrices -v
```

- [ ] **Step R3.3: Rewrite `_snapshot_mtgjson_prices`**

```python
def _snapshot_mtgjson_prices(self, records: list[BaseModel]) -> None:
    """Snapshot today's MTGJson paper prices into bronze_mtgjson_prices_history (EAV).

    Extracts all (retailer, tx_type, finish) combinations present in each
    record's paper dict using look-back semantics. One row = one price point.
    All retailers in the feed are captured without pre-selection.

    Args:
        records: Pydantic model instances with uuid and paper fields.
    """
    if not records:
        logger.warning("No MTGJson price records to snapshot — skipping")
        return

    today_iso = date.today().isoformat()
    rows: list[dict] = []
    for record in records:
        dump = record.model_dump(mode="json")
        rows.extend(
            _extract_paper_eav_rows(dump.get("paper"), dump["uuid"], today_iso)
        )

    if not rows:
        logger.warning("No paper price rows found — skipping MTGJson snapshot")
        return

    df = pd.DataFrame(rows)
    logger.progress("Snapshotting %d MTGJson EAV price rows", len(df))
    self._writer.append(
        df,
        "bronze_mtgjson_prices_history",
        ["uuid", "retailer", "tx_type", "finish"],
    )
    logger.info("Snapshotted %d MTGJson EAV rows for %s", len(rows), today_iso)
```

- [ ] **Step R3.4: Run to verify PASS**

```
pytest tests/data/cards/storage/test_bronze.py::TestSnapshotMtgjsonPrices -v
```

- [ ] **Step R3.5: Run full suite**

```
pytest tests/ -x --tb=short -q 2>&1 | tail -10
```

- [ ] **Step R3.6: Commit**

```bash
git add src/data/cards/storage/bronze/storage.py tests/data/cards/storage/test_bronze.py
git commit -m "refactor: _snapshot_mtgjson_prices emits EAV rows via _extract_paper_eav_rows"
```

---

## Task R4: Bronze — `_snapshot_scryfall_prices` adds `tix`

**Files:**
- Modify: `src/data/cards/storage/bronze/storage.py`
- Modify: `tests/data/cards/storage/test_bronze.py`

### Background

`tix` is a valid Scryfall price field. The decision not to use it downstream is Silver's concern, not Bronze's. Add it.

- [ ] **Step R4.1: Update `TestSnapshotScryfallPrices` — rename tix test and add tix value test**

Replace `test_tix_key_is_ignored` with two tests:

```python
def test_tix_column_present_in_schema(self):
    with _bronze() as b:
        record = _ScryfallCard(
            id="s1",
            prices={"eur": "3.20", "tix": "0.05"},
        )
        b._snapshot_scryfall_prices([record])
        cols = {r[0] for r in b._con.execute(
            f"DESCRIBE {self.HISTORY_TABLE}"
        ).fetchall()}
    assert "tix" in cols

def test_tix_stored_as_float(self):
    with _bronze() as b:
        record = _ScryfallCard(
            id="s1",
            prices={"eur": "3.20", "tix": "0.05"},
        )
        b._snapshot_scryfall_prices([record])
        row = b._con.execute(
            f"SELECT tix FROM {self.HISTORY_TABLE}"
        ).fetchone()
    assert row is not None
    assert row[0] == pytest.approx(0.05)

def test_null_tix_produces_null_column(self):
    with _bronze() as b:
        record = _ScryfallCard(id="s1", prices={"eur": "3.20", "tix": None})
        b._snapshot_scryfall_prices([record])
        row = b._con.execute(f"SELECT tix FROM {self.HISTORY_TABLE}").fetchone()
    assert row is not None and row[0] is None
```

- [ ] **Step R4.2: Run to verify FAIL**

```
pytest tests/data/cards/storage/test_bronze.py::TestSnapshotScryfallPrices -v
```

- [ ] **Step R4.3: Add `tix` to `_snapshot_scryfall_prices`**

In the `rows.append(...)` call, add:

```python
"tix": float(prices["tix"]) if prices.get("tix") is not None else None,
```

- [ ] **Step R4.4: Run to verify PASS**

```
pytest tests/data/cards/storage/test_bronze.py::TestSnapshotScryfallPrices -v
```

- [ ] **Step R4.5: Commit**

```bash
git add src/data/cards/storage/bronze/storage.py tests/data/cards/storage/test_bronze.py
git commit -m "feat: add tix column to bronze_scryfall_prices_history"
```

---

## Task R5: Migration Script — EAV + tix

**Files:**
- Modify: `scripts/migrate_bronze_prices.py`
- Modify: `tests/scripts/test_migrate_bronze_prices.py`

### Background

Replace the wide-column MTGJson migration with EAV output. Add `tix` to the Scryfall migration. Source is still `cards_copy.duckdb` with old JSON `paper`/`prices` columns.

- [ ] **Step R5.1: Replace `TestMigrateMtgjsonPrices` entirely**

```python
class TestMigrateMtgjsonPrices:
    def test_creates_eav_schema(self, tmp_path):
        source = str(tmp_path / "source.duckdb")
        target = str(tmp_path / "target.duckdb")
        _make_mtgjson_source(source)
        _make_target_with_old_mtgjson(source, target)

        migrate_mtgjson_prices(source, target)

        con = duckdb.connect(target, read_only=True)
        cols = {r[0] for r in con.execute("DESCRIBE bronze_mtgjson_prices_history").fetchall()}
        con.close()
        assert cols == {"uuid", "snapshot_date", "retailer", "tx_type", "finish", "price"}

    def test_no_paper_or_mtgo_columns(self, tmp_path):
        source = str(tmp_path / "source.duckdb")
        target = str(tmp_path / "target.duckdb")
        _make_mtgjson_source(source)
        _make_target_with_old_mtgjson(source, target)

        migrate_mtgjson_prices(source, target)

        con = duckdb.connect(target, read_only=True)
        cols = {r[0] for r in con.execute("DESCRIBE bronze_mtgjson_prices_history").fetchall()}
        con.close()
        assert "paper" not in cols
        assert "mtgo" not in cols

    def test_extracts_all_retailers_from_source(self, tmp_path):
        source = str(tmp_path / "source.duckdb")
        target = str(tmp_path / "target.duckdb")
        _make_mtgjson_source(source)
        _make_target_with_old_mtgjson(source, target)

        migrate_mtgjson_prices(source, target)

        con = duckdb.connect(target, read_only=True)
        retailers = {r[0] for r in con.execute(
            "SELECT DISTINCT retailer FROM bronze_mtgjson_prices_history"
        ).fetchall()}
        con.close()
        assert "cardmarket" in retailers
        assert "tcgplayer" in retailers

    def test_extracts_correct_price_value(self, tmp_path):
        source = str(tmp_path / "source.duckdb")
        target = str(tmp_path / "target.duckdb")
        _make_mtgjson_source(source)
        _make_target_with_old_mtgjson(source, target)

        migrate_mtgjson_prices(source, target)

        con = duckdb.connect(target, read_only=True)
        row = con.execute(
            "SELECT price FROM bronze_mtgjson_prices_history"
            " WHERE retailer='cardmarket' AND tx_type='retail' AND finish='normal'"
        ).fetchone()
        con.close()
        assert row is not None
        assert row[0] == pytest.approx(3.20)

    def test_preserves_uuid_and_date(self, tmp_path):
        source = str(tmp_path / "source.duckdb")
        target = str(tmp_path / "target.duckdb")
        _make_mtgjson_source(source)
        _make_target_with_old_mtgjson(source, target)

        migrate_mtgjson_prices(source, target)

        con = duckdb.connect(target, read_only=True)
        row = con.execute(
            "SELECT uuid, snapshot_date FROM bronze_mtgjson_prices_history LIMIT 1"
        ).fetchone()
        con.close()
        assert row[0] == "uuid-1"
        assert str(row[1]) == "2026-05-11"

    def test_returns_eav_row_count(self, tmp_path):
        source = str(tmp_path / "source.duckdb")
        target = str(tmp_path / "target.duckdb")
        _make_mtgjson_source(source)
        _make_target_with_old_mtgjson(source, target)

        count = migrate_mtgjson_prices(source, target)
        # source fixture: cardmarket retail normal+foil+buylist + tcgplayer retail normal+foil+buylist = 6
        assert count == 6

    def test_empty_source_produces_no_rows(self, tmp_path):
        source = str(tmp_path / "source.duckdb")
        target = str(tmp_path / "target.duckdb")
        con = duckdb.connect(source)
        con.execute(
            "CREATE TABLE bronze_mtgjson_prices_history"
            " (uuid VARCHAR, snapshot_date VARCHAR, paper VARCHAR, mtgo VARCHAR)"
        )
        con.close()
        _make_target_with_old_mtgjson(source, target)

        count = migrate_mtgjson_prices(source, target)
        assert count == 0
```

Add two tests to `TestMigrateScryfallPrices`:

```python
def test_tix_column_present(self, tmp_path):
    source = str(tmp_path / "source.duckdb")
    target = str(tmp_path / "target.duckdb")
    _make_scryfall_source(source)
    _make_target_with_old_scryfall(source, target)

    migrate_scryfall_prices(source, target)

    con = duckdb.connect(target, read_only=True)
    cols = {r[0] for r in con.execute("DESCRIBE bronze_scryfall_prices_history").fetchall()}
    con.close()
    assert "tix" in cols

def test_tix_value_extracted(self, tmp_path):
    source = str(tmp_path / "source.duckdb")
    target = str(tmp_path / "target.duckdb")
    _make_scryfall_source(source)
    _make_target_with_old_scryfall(source, target)

    migrate_scryfall_prices(source, target)

    con = duckdb.connect(target, read_only=True)
    row = con.execute("SELECT tix FROM bronze_scryfall_prices_history").fetchone()
    con.close()
    assert row is not None
    assert row[0] == pytest.approx(0.05)
```

Note: `_make_scryfall_source` fixture already includes `"tix": "0.05"` in the JSON — no fixture change needed.

- [ ] **Step R5.2: Run to verify FAIL**

```
pytest tests/scripts/test_migrate_bronze_prices.py -v
```

- [ ] **Step R5.3: Rewrite `migrate_mtgjson_prices`**

```python
def migrate_mtgjson_prices(source_path: str, target_path: str) -> int:
    """Migrate bronze_mtgjson_prices_history from JSON paper column to EAV rows.

    Returns:
        Total number of EAV rows written.
    """
    src = duckdb.connect(source_path, read_only=True)
    tgt = duckdb.connect(target_path, read_only=False)

    try:
        source_rows = src.execute(
            "SELECT uuid, snapshot_date, paper FROM bronze_mtgjson_prices_history"
        ).fetchall()

        tgt.execute("""
            CREATE TABLE bronze_mtgjson_prices_history_new (
                uuid          VARCHAR NOT NULL,
                snapshot_date VARCHAR NOT NULL,
                retailer      VARCHAR NOT NULL,
                tx_type       VARCHAR NOT NULL,
                finish        VARCHAR NOT NULL,
                price         FLOAT
            )
        """)

        eav_rows: list[list] = []
        for uuid, snapshot_date, paper_json in source_rows:
            paper = json.loads(paper_json) if isinstance(paper_json, str) else (paper_json or {})
            snap_str = str(snapshot_date)
            for retailer, retailer_data in paper.items():
                if not retailer_data:
                    continue
                for tx_type in ("buylist", "retail"):
                    listing = (retailer_data.get(tx_type)) or {}
                    for finish, prices in listing.items():
                        if not isinstance(prices, dict):
                            continue
                        candidates = {k: v for k, v in prices.items() if k <= snap_str}
                        if candidates:
                            eav_rows.append([
                                uuid, snap_str, retailer, tx_type, finish,
                                float(candidates[max(candidates)]),
                            ])

        if eav_rows:
            tgt.executemany(
                "INSERT INTO bronze_mtgjson_prices_history_new"
                " (uuid, snapshot_date, retailer, tx_type, finish, price)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                eav_rows,
            )

        tgt.execute("DROP TABLE IF EXISTS bronze_mtgjson_prices_history")
        tgt.execute(
            "ALTER TABLE bronze_mtgjson_prices_history_new"
            " RENAME TO bronze_mtgjson_prices_history"
        )
        tgt.execute("CHECKPOINT")
        return len(eav_rows)
    finally:
        src.close()
        tgt.close()
```

Update `migrate_scryfall_prices` — add `tix` to CREATE TABLE and batch:

```python
tgt.execute("""
    CREATE TABLE bronze_scryfall_prices_history_new (
        id            VARCHAR NOT NULL,
        snapshot_date VARCHAR NOT NULL,
        eur           FLOAT,
        eur_foil      FLOAT,
        usd           FLOAT,
        usd_foil      FLOAT,
        tix           FLOAT
    )
""")

# In the batch loop, append:
batch.append([
    scryfall_id,
    str(snapshot_date),
    float(prices["eur"])      if prices.get("eur")      is not None else None,
    float(prices["eur_foil"]) if prices.get("eur_foil") is not None else None,
    float(prices["usd"])      if prices.get("usd")      is not None else None,
    float(prices["usd_foil"]) if prices.get("usd_foil") is not None else None,
    float(prices["tix"])      if prices.get("tix")      is not None else None,
])

# UPDATE INSERT to 7 placeholders:
tgt.executemany(
    "INSERT INTO bronze_scryfall_prices_history_new VALUES (?, ?, ?, ?, ?, ?, ?)",
    batch,
)
```

- [ ] **Step R5.4: Run to verify PASS**

```
pytest tests/scripts/test_migrate_bronze_prices.py -v
```

- [ ] **Step R5.5: Run full suite**

```
pytest tests/ -x --tb=short -q 2>&1 | tail -10
```

- [ ] **Step R5.6: Commit**

```bash
git add scripts/migrate_bronze_prices.py tests/scripts/test_migrate_bronze_prices.py
git commit -m "refactor: migration produces EAV rows for MTGJson, adds tix to Scryfall"
```

---

## Task 8: Run Migration on Live Data (MANUAL)

> **Stop here — manual step required before continuing to Silver tasks.**

⚠️ The old Task 7 script produced wide columns. This new script produces EAV. Run the new one:

```bash
python scripts/migrate_bronze_prices.py \
    --source data/bronze/cards_copy.duckdb \
    --target data/bronze/cards.duckdb
```

Verify schema:

```python
import duckdb
con = duckdb.connect("data/bronze/cards.duckdb", read_only=True)
print(con.execute("DESCRIBE bronze_mtgjson_prices_history").fetchall())
# Expected cols: uuid, snapshot_date, retailer, tx_type, finish, price
print(con.execute("DESCRIBE bronze_scryfall_prices_history").fetchall())
# Expected cols: id, snapshot_date, eur, eur_foil, usd, usd_foil, tix
con.close()
```

---

## Task 9: Silver — `_build_scryfall_base` → `scryfall_prices_base.sql`

**Files:**
- Create: `src/data/cards/storage/silver/sql/scryfall_prices_base.sql`
- Modify: `src/data/cards/storage/silver/prices.py`
- Modify: `tests/data/cards/storage/test_silver.py`

### Background

`_build_scryfall_base` has inline SQL with `json_extract_string(prices, '$.eur')`. After the migration, `eur`, `eur_foil`, `usd`, `usd_foil` are direct FLOAT columns. Silver does **not** select `tix` — that decision lives here, in the SQL. Move query to a `.sql` file. Update test fixtures to use scalar columns.

- [ ] **Step 9.1: Find all Scryfall price history fixtures in `test_silver.py`**

```
grep -n "bronze_scryfall_prices_history\|prices.*json\|json_extract" tests/data/cards/storage/test_silver.py
```

Update every fixture that creates `bronze_scryfall_prices_history` to use scalar columns instead of a `prices` JSON column:

```python
# Old fixture:
con.execute("CREATE TABLE bronze_scryfall_prices_history (id VARCHAR, snapshot_date DATE, prices VARCHAR)")
con.execute("INSERT ... VALUES ('s1', '2026-06-01', '{\"eur\": \"3.20\", ...}')")

# New fixture:
con.execute("""
    CREATE TABLE bronze_scryfall_prices_history (
        id VARCHAR, snapshot_date DATE,
        eur FLOAT, eur_foil FLOAT, usd FLOAT, usd_foil FLOAT, tix FLOAT
    )
""")
con.execute("INSERT ... VALUES ('s1', '2026-06-01', 3.20, NULL, 1.95, NULL, NULL)")
```

- [ ] **Step 9.2: Run to verify tests are now failing**

```
pytest tests/data/cards/storage/test_silver.py -k "scryfall_base or build_scryfall" -v
```

- [ ] **Step 9.3: Create `src/data/cards/storage/silver/sql/scryfall_prices_base.sql`**

```sql
SELECT
    id                                                    AS scryfall_id,
    snapshot_date,
    eur,
    eur_foil,
    usd,
    usd_foil
FROM bronze_scryfall_prices_history
WHERE snapshot_date = ?
```

- [ ] **Step 9.4: Rewrite `_build_scryfall_base` to load from SQL file**

```python
def _build_scryfall_base(self, today: str) -> pd.DataFrame:
    sql_path = Path(__file__).parent / "sql" / "scryfall_prices_base.sql"
    sql = sql_path.read_text()

    card_map = self._silver_con.execute(
        "SELECT COALESCE(uuid, canonical_uuid) AS uuid, scryfall_id"
        " FROM silver_cards"
        " WHERE scryfall_id IS NOT NULL"
        "   AND COALESCE(uuid, canonical_uuid) IS NOT NULL"
        "   AND (uuid IS NOT NULL OR language = 'English')"
    ).df()

    scryfall_prices = self._bronze_con.execute(sql, [today]).df()
    return scryfall_prices.merge(card_map, on="scryfall_id", how="inner")
```

Add `from pathlib import Path` import if not present.

- [ ] **Step 9.5: Run to verify PASS**

```
pytest tests/data/cards/storage/test_silver.py -k "scryfall_base or build_scryfall" -v
```

- [ ] **Step 9.6: Commit**

```bash
git add src/data/cards/storage/silver/sql/scryfall_prices_base.sql src/data/cards/storage/silver/prices.py tests/data/cards/storage/test_silver.py
git commit -m "refactor: _build_scryfall_base reads scalar Bronze columns via SQL file"
```

---

## Task 10: Silver — `_join_mtgjson_prices` → CASE WHEN pivot

**Files:**
- Create: `src/data/cards/storage/silver/sql/mtgjson_prices_daily.sql`
- Modify: `src/data/cards/storage/silver/prices.py`
- Modify: `tests/data/cards/storage/test_silver.py`

### Background

Bronze now stores EAV. Silver needs to pivot EAV → 6 wide columns. The CASE WHEN SQL is the single place where `(retailer, tx_type, finish) → column_name` mapping lives; `_MTGJSON_PRICE_MAP` in `SilverPriceBuilder` remains as documentation and for the fallback path.

Replace `_extract_all_prices` + `pd.concat` approach with a single SQL query.

- [ ] **Step 10.1: Update MTGJson Bronze fixtures in `test_silver.py`**

Replace every fixture that creates `bronze_mtgjson_prices_history` with a `paper` VARCHAR column:

```python
# Old:
con.execute("CREATE TABLE bronze_mtgjson_prices_history (uuid VARCHAR, snapshot_date DATE, paper VARCHAR)")
con.execute("INSERT ... VALUES ('u1', '2026-06-01', '{...json...}')")

# New EAV:
con.execute("""
    CREATE TABLE bronze_mtgjson_prices_history (
        uuid VARCHAR, snapshot_date DATE,
        retailer VARCHAR, tx_type VARCHAR, finish VARCHAR, price FLOAT
    )
""")
con.execute("INSERT ... VALUES ('u1', '2026-06-01', 'cardmarket', 'retail', 'normal', 3.20)")
con.execute("INSERT ... VALUES ('u1', '2026-06-01', 'cardmarket', 'retail', 'foil', 8.50)")
# etc. for each price point
```

- [ ] **Step 10.2: Run to verify FAIL**

```
pytest tests/data/cards/storage/test_silver.py -k "mtgjson or join_mtgjson" -v
```

- [ ] **Step 10.3: Create `src/data/cards/storage/silver/sql/mtgjson_prices_daily.sql`**

```sql
SELECT
    uuid,
    snapshot_date,
    MAX(CASE WHEN retailer = 'cardmarket' AND tx_type = 'retail'  AND finish = 'normal' THEN price END) AS cardmarket_eur,
    MAX(CASE WHEN retailer = 'cardmarket' AND tx_type = 'retail'  AND finish = 'foil'   THEN price END) AS cardmarket_eur_foil,
    MAX(CASE WHEN retailer = 'cardmarket' AND tx_type = 'buylist' AND finish = 'normal' THEN price END) AS cardmarket_buylist_eur,
    MAX(CASE WHEN retailer = 'tcgplayer'  AND tx_type = 'retail'  AND finish = 'normal' THEN price END) AS tcgplayer_usd,
    MAX(CASE WHEN retailer = 'tcgplayer'  AND tx_type = 'retail'  AND finish = 'foil'   THEN price END) AS tcgplayer_usd_foil,
    MAX(CASE WHEN retailer = 'tcgplayer'  AND tx_type = 'buylist' AND finish = 'normal' THEN price END) AS tcgplayer_buylist_usd
FROM bronze_mtgjson_prices_history
WHERE snapshot_date = ?
GROUP BY uuid, snapshot_date
```

- [ ] **Step 10.4: Rewrite `_join_mtgjson_prices`**

```python
def _join_mtgjson_prices(
    self, df: pd.DataFrame, bronze_tables: set[str], today: str
) -> pd.DataFrame:
    if "bronze_mtgjson_prices_history" not in bronze_tables:
        logger.warning(
            "bronze_mtgjson_prices_history not found — MTGJson prices omitted"
        )
        for col in self._MTGJSON_PRICE_MAP:
            df[col] = None
        return df

    sql_path = Path(__file__).parent / "sql" / "mtgjson_prices_daily.sql"
    sql = sql_path.read_text()
    mtgjson = self._bronze_con.execute(sql, [today]).df()

    for col in self._MTGJSON_PRICE_MAP:
        if col not in mtgjson.columns:
            mtgjson[col] = None

    return df.merge(mtgjson, on=["uuid", "snapshot_date"], how="left")
```

- [ ] **Step 10.5: Run to verify PASS**

```
pytest tests/data/cards/storage/test_silver.py -k "mtgjson or join_mtgjson" -v
```

- [ ] **Step 10.6: Run full suite**

```
pytest tests/ -x --tb=short -q 2>&1 | tail -10
```

- [ ] **Step 10.7: Commit**

```bash
git add src/data/cards/storage/silver/sql/mtgjson_prices_daily.sql src/data/cards/storage/silver/prices.py tests/data/cards/storage/test_silver.py
git commit -m "refactor: _join_mtgjson_prices pivots EAV Bronze to wide via CASE WHEN SQL"
```

---

## Task 11: Silver — `build_language_prices` → `scryfall_language_prices_base.sql`

**Files:**
- Create: `src/data/cards/storage/silver/sql/scryfall_language_prices_base.sql`
- Modify: `src/data/cards/storage/silver/prices.py`
- Modify: `tests/data/cards/storage/test_silver.py`

### Background

`build_language_prices` contains an identical `json_extract_string(prices, '$.eur')` query. Update fixtures to use scalar columns; move SQL to file; simplify query to direct column access.

- [ ] **Step 11.1: Update language prices fixtures in `test_silver.py`**

Same fixture update as Task 9 — any `bronze_scryfall_prices_history` creation with a `prices VARCHAR` column must be updated to scalar columns (eur, eur_foil, usd, usd_foil, tix). Check for fixtures specific to `build_language_prices` tests.

- [ ] **Step 11.2: Run to verify FAIL**

```
pytest tests/data/cards/storage/test_silver.py -k "language_price" -v
```

- [ ] **Step 11.3: Create `src/data/cards/storage/silver/sql/scryfall_language_prices_base.sql`**

```sql
SELECT
    id           AS scryfall_id,
    snapshot_date,
    eur,
    eur_foil,
    usd,
    usd_foil
FROM bronze_scryfall_prices_history
WHERE snapshot_date = ?
```

- [ ] **Step 11.4: Rewrite the inner query in `build_language_prices`**

Replace the inline `json_extract_string` query block with:

```python
sql_path = Path(__file__).parent / "sql" / "scryfall_language_prices_base.sql"
sql = sql_path.read_text()
scryfall_prices = self._bronze_con.execute(sql, [today]).df()
```

- [ ] **Step 11.5: Run to verify PASS**

```
pytest tests/data/cards/storage/test_silver.py -k "language_price" -v
```

- [ ] **Step 11.6: Commit**

```bash
git add src/data/cards/storage/silver/sql/scryfall_language_prices_base.sql src/data/cards/storage/silver/prices.py tests/data/cards/storage/test_silver.py
git commit -m "refactor: build_language_prices reads scalar columns via SQL file"
```

---

## Task 12: Silver — Remove dead code

**Files:**
- Modify: `src/data/cards/storage/silver/prices.py`

### Background

`_extract_all_prices` and `import json` are now dead. `_MTGJSON_PRICE_MAP` **stays** — it documents the semantic mapping and is used by the fallback path in `_join_mtgjson_prices`.

- [ ] **Step 12.1: Write failing test (import check)**

In `test_silver.py`, add:

```python
def test_extract_all_prices_removed():
    import inspect
    import src.data.cards.storage.silver.prices as mod
    assert not hasattr(mod.SilverPriceBuilder, "_extract_all_prices"), \
        "_extract_all_prices should have been removed"
```

- [ ] **Step 12.2: Run to verify FAIL**

```
pytest tests/data/cards/storage/test_silver.py::test_extract_all_prices_removed -v
```

- [ ] **Step 12.3: Remove `_extract_all_prices` and `import json` from `prices.py`**

Delete the entire `_extract_all_prices` static method.
Remove `import json` from the top of the file if it has no other usages (grep first).

- [ ] **Step 12.4: Run to verify PASS**

```
pytest tests/data/cards/storage/test_silver.py -v
```

- [ ] **Step 12.5: Run full suite**

```
pytest tests/ -x --tb=short -q 2>&1 | tail -10
```

- [ ] **Step 12.6: Commit**

```bash
git add src/data/cards/storage/silver/prices.py tests/data/cards/storage/test_silver.py
git commit -m "refactor: remove _extract_all_prices and import json from Silver prices"
```

---

## Task 13: Health — WARN status + schema drift check

**Files:**
- Modify: `src/data/cards/storage/health.py`
- Modify: `tests/data/cards/storage/test_health.py`

### Background

Two changes:
1. `CheckResult.status` gains a `"WARN"` option. `run_health_checks` logs WARNs but does not `sys.exit(1)` for them.
2. New function `_check_bronze_prices_schema_drift` queries distinct `(retailer, tx_type, finish)` from today's EAV snapshot and compares to Silver's `_MTGJSON_PRICE_MAP`. New combinations → WARN; expected combinations missing → WARN.

- [ ] **Step 13.1: Write failing tests**

In `tests/data/cards/storage/test_health.py`, add:

```python
from src.data.cards.storage.health import _check_bronze_prices_schema_drift

_EXPECTED_COMBOS = {
    ("cardmarket", "retail",  "normal"),
    ("cardmarket", "retail",  "foil"),
    ("cardmarket", "buylist", "normal"),
    ("tcgplayer",  "retail",  "normal"),
    ("tcgplayer",  "retail",  "foil"),
    ("tcgplayer",  "buylist", "normal"),
}


class TestCheckBronzePricesSchemaWarn:
    def _make_eav_table(self, con, rows):
        con.execute("""
            CREATE TABLE bronze_mtgjson_prices_history (
                uuid VARCHAR, snapshot_date VARCHAR,
                retailer VARCHAR, tx_type VARCHAR, finish VARCHAR, price FLOAT
            )
        """)
        for r in rows:
            con.execute(
                "INSERT INTO bronze_mtgjson_prices_history VALUES (?, ?, ?, ?, ?, ?)",
                list(r),
            )

    def test_pass_when_combos_match_expected(self):
        con = duckdb.connect(":memory:")
        today = datetime.date(2026, 6, 24)
        rows = [(f"u{i}", today.isoformat(), r, t, f, 1.0)
                for i, (r, t, f) in enumerate(_EXPECTED_COMBOS)]
        self._make_eav_table(con, rows)
        results = _check_bronze_prices_schema_drift(con, today, _EXPECTED_COMBOS)
        assert all(r.status == "PASS" for r in results)
        con.close()

    def test_warn_on_new_retailer_not_in_map(self):
        con = duckdb.connect(":memory:")
        today = datetime.date(2026, 6, 24)
        rows = [("u1", today.isoformat(), "newretailer", "retail", "normal", 1.0)]
        self._make_eav_table(con, rows)
        results = _check_bronze_prices_schema_drift(con, today, _EXPECTED_COMBOS)
        statuses = {r.status for r in results}
        assert "WARN" in statuses

    def test_warn_on_missing_expected_combo(self):
        con = duckdb.connect(":memory:")
        today = datetime.date(2026, 6, 24)
        # Only cardmarket retail normal — all others missing
        rows = [("u1", today.isoformat(), "cardmarket", "retail", "normal", 1.0)]
        self._make_eav_table(con, rows)
        results = _check_bronze_prices_schema_drift(con, today, _EXPECTED_COMBOS)
        statuses = {r.status for r in results}
        assert "WARN" in statuses

    def test_warn_does_not_cause_exit_in_run_health_checks(self, tmp_path):
        today = datetime.date(2026, 6, 24)
        bronze_path, silver_path, gold_path = _make_all_dbs(tmp_path, today)
        # Add EAV table with a new unknown retailer
        b = duckdb.connect(bronze_path)
        b.execute("""
            CREATE TABLE bronze_mtgjson_prices_history (
                uuid VARCHAR, snapshot_date VARCHAR,
                retailer VARCHAR, tx_type VARCHAR, finish VARCHAR, price FLOAT
            )
        """)
        b.execute(
            "INSERT INTO bronze_mtgjson_prices_history VALUES (?, ?, ?, ?, ?, ?)",
            ["u1", today.isoformat(), "newretailer", "retail", "normal", 1.0],
        )
        b.close()
        # Should not raise SystemExit — WARN is not FAIL
        results = run_health_checks(bronze_path, silver_path, gold_path, today)
        assert any(r.status == "WARN" for r in results)
```

Also add test that `CheckResult` accepts `"WARN"` status:

```python
def test_check_result_warn():
    r = CheckResult(
        name="schema drift", layer="bronze", status="WARN",
        detail="new combo: ('newretailer', 'retail', 'normal')"
    )
    assert r.status == "WARN"
```

- [ ] **Step 13.2: Run to verify FAIL**

```
pytest tests/data/cards/storage/test_health.py -k "warn or drift" -v
```

- [ ] **Step 13.3: Add WARN to `CheckResult` and update `run_health_checks`**

In `health.py`:

```python
from typing import Literal

@dataclass
class CheckResult:
    name: str
    layer: str
    status: Literal["PASS", "WARN", "FAIL"]
    detail: str
```

In `run_health_checks`, after collecting all results:

```python
warn_results = [r for r in results if r.status == "WARN"]
fail_results = [r for r in results if r.status == "FAIL"]

for r in warn_results:
    logger.warning("[WARN] %s (%s): %s", r.name, r.layer, r.detail)

if fail_results:
    sys.exit(1)
```

- [ ] **Step 13.4: Implement `_check_bronze_prices_schema_drift`**

```python
from src.data.cards.storage.silver.prices import SilverPriceBuilder

_SILVER_PRICE_COMBOS: frozenset[tuple[str, str, str]] = frozenset(
    SilverPriceBuilder._MTGJSON_PRICE_MAP.values()
)


def _check_bronze_prices_schema_drift(
    bronze_con: duckdb.DuckDBPyConnection,
    today: datetime.date,
    expected: set[tuple[str, str, str]] | None = None,
) -> list[CheckResult]:
    """Check for (retailer, tx_type, finish) combinations not in Silver's map.

    Returns WARN results for new combinations (captured in Bronze but unknown
    to Silver) and for missing expected combinations (Silver expects them but
    they didn't appear in today's snapshot).
    """
    if expected is None:
        expected = set(_SILVER_PRICE_COMBOS)

    try:
        rows = bronze_con.execute(
            "SELECT DISTINCT retailer, tx_type, finish "
            "FROM bronze_mtgjson_prices_history "
            "WHERE snapshot_date = ?",
            [today.isoformat()],
        ).fetchall()
    except Exception:
        return []

    actual = {(r[0], r[1], r[2]) for r in rows}
    results = []

    new_combos = actual - expected
    if new_combos:
        results.append(CheckResult(
            name="bronze mtgjson price schema drift",
            layer="bronze",
            status="WARN",
            detail=f"{len(new_combos)} new (retailer,tx_type,finish) combinations"
                   f" not in Silver map: {sorted(new_combos)}",
        ))

    missing = expected - actual
    if missing:
        results.append(CheckResult(
            name="bronze mtgjson price schema drift",
            layer="bronze",
            status="WARN",
            detail=f"{len(missing)} expected combinations absent from today's snapshot:"
                   f" {sorted(missing)}",
        ))

    if not results:
        results.append(CheckResult(
            name="bronze mtgjson price schema drift",
            layer="bronze",
            status="PASS",
            detail=f"All {len(actual)} combinations match Silver map",
        ))

    return results
```

Wire into `run_health_checks`:

```python
drift_results = _check_bronze_prices_schema_drift(bronze_con, today)
results.extend(drift_results)
```

Also replace the original `_check_bronze_prices_coverage` function (which checked for a `cardmarket_eur` column that no longer exists) with EAV-compatible logic:

```python
def _check_bronze_prices_coverage(
    bronze_con: duckdb.DuckDBPyConnection, today: datetime.date
) -> CheckResult:
    """Check that at least one MTGJson price row exists for today."""
    try:
        row = bronze_con.execute(
            "SELECT count(*) FROM bronze_mtgjson_prices_history WHERE snapshot_date = ?",
            [today.isoformat()],
        ).fetchone()
        count = row[0] if row else 0
    except Exception:
        return CheckResult(
            name="bronze mtgjson prices today",
            layer="bronze",
            status="FAIL",
            detail="bronze_mtgjson_prices_history not found",
        )

    if count == 0:
        return CheckResult(
            name="bronze mtgjson prices today",
            layer="bronze",
            status="FAIL",
            detail=f"No MTGJson price rows for {today}",
        )
    return CheckResult(
        name="bronze mtgjson prices today",
        layer="bronze",
        status="PASS",
        detail=f"{count} EAV price rows for {today}",
    )
```

- [ ] **Step 13.5: Run to verify PASS**

```
pytest tests/data/cards/storage/test_health.py -v
```

- [ ] **Step 13.6: Run full suite**

```
pytest tests/ -x --tb=short -q 2>&1 | tail -10
```

- [ ] **Step 13.7: Commit**

```bash
git add src/data/cards/storage/health.py tests/data/cards/storage/test_health.py
git commit -m "feat: add WARN status to CheckResult, schema drift health check for Bronze EAV"
```

---

## Task 14: Documentation + ADR-025

**Files:**
- Modify: `docs/adr/ADR-003-medallion-architecture.md`
- Modify: `docs/adr/ADR-012-physical-cards-only.md`
- Create: `docs/adr/ADR-025-scalar-bronze-prices.md`
- Modify: `docs/architecture/c4/bronze-storage.md`
- Modify: `docs/architecture/c4/silver-storage.md`
- Modify: `docs/architecture/data/table-schemas.md`
- Modify: `docs/architecture/data/data-lineage.md`
- Modify: `docs/architecture/data/glossary.md`

### Step 14.1: `ADR-025-scalar-bronze-prices.md`

```markdown
# ADR-025: Scalar Bronze Price Tables with EAV for MTGJson

**Date:** 2026-06-24
**Status:** Accepted

## Context

Bronze price history tables previously stored prices as JSON blobs (`paper`, `prices` columns).
Silver was forced to parse JSON in Python (violating ADR-024) and pre-select specific retailers
at Bronze ingestion time (violating the medallion architecture principle that Bronze = raw,
Silver = semantic selection).

## Decision

1. **MTGJson Bronze → EAV**: `bronze_mtgjson_prices_history` stores one row per price point:
   `(uuid, snapshot_date, retailer, tx_type, finish, price)`. All retailers present in the
   MTGJson feed are captured without pre-selection. Silver pivots to wide columns via CASE WHEN SQL.

2. **Scryfall Bronze → scalar + tix**: `bronze_scryfall_prices_history` stores scalar FLOAT columns
   including `tix`. Silver does not select `tix`; the exclusion decision lives in Silver SQL.

3. **Bronze = structural normalization only. Silver = semantic selection only.**

## Consequences

- New retailers appearing in MTGJson (e.g. `cardkingdom`) are captured automatically in Bronze
  without code changes. Silver's CASE WHEN SQL is the single place that decides what downstream
  consumers see.
- `_MTGJSON_PRICE_MAP` lives exclusively in `SilverPriceBuilder`. Bronze has no concept of
  which combinations matter.
- Schema drift is detected post-pipeline by `_check_bronze_prices_schema_drift` (health.py),
  which produces WARN (not FAIL) when new or missing combinations are observed.
- One-time migration via `scripts/migrate_bronze_prices.py` from `cards_copy.duckdb`.
```

### Step 14.2: Update `table-schemas.md`

Update `bronze_mtgjson_prices_history` schema section:

```markdown
### bronze_mtgjson_prices_history

**Grain:** 1 row per (uuid, snapshot_date, retailer, tx_type, finish)

| Column        | Type    | Description                            |
|---------------|---------|----------------------------------------|
| uuid          | VARCHAR | MTGJson card UUID                      |
| snapshot_date | VARCHAR | Date this price was recorded           |
| retailer      | VARCHAR | Source retailer (cardmarket, tcgplayer, …) |
| tx_type       | VARCHAR | Transaction type: retail or buylist    |
| finish        | VARCHAR | Card finish: normal, foil, etched      |
| price         | FLOAT   | Price in retailer's native currency    |
```

Update `bronze_scryfall_prices_history`:

```markdown
### bronze_scryfall_prices_history

**Grain:** 1 row per (id, snapshot_date)

| Column        | Type   | Description                  |
|---------------|--------|------------------------------|
| id            | VARCHAR| Scryfall UUID                |
| snapshot_date | VARCHAR| Date this snapshot was taken |
| eur           | FLOAT? | EUR non-foil price           |
| eur_foil      | FLOAT? | EUR foil price               |
| usd           | FLOAT? | USD non-foil price           |
| usd_foil      | FLOAT? | USD foil price               |
| tix           | FLOAT? | MTGO ticket price (captured; not used downstream) |
```

### Step 14.3: Update remaining docs

- `ADR-003`: Update Bronze price table schemas.
- `ADR-012`: Note that `tix` is now stored in Bronze but excluded at Silver level.
- `bronze-storage.md`: MTGJson price snapshot now emits EAV rows.
- `silver-storage.md`: Step for MTGJson prices now says "CASE WHEN pivot from EAV".
- `data-lineage.md`: Bronze→Silver price join now described as EAV pivot.
- `glossary.md`: Update "Price snapshot" definition.

- [ ] **Step 14.4: Commit**

```bash
git add -f docs/
git commit -m "docs: update schemas and ADR-025 for EAV Bronze prices + tix"
```
