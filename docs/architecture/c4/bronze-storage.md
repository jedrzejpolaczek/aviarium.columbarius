# C4 — BronzeStorage Code

BronzeStorage coordinates persistence of raw card ingestion data into DuckDB. It orchestrates source-to-table routing via STORAGE_CONFIG and provides three write patterns (full_load, upsert, append) that abstract DuckDB complexity from higher layers.

```mermaid
classDiagram
  class DuckDBWriter {
    -_con: DuckDBPyConnection
    +full_load(df: DataFrame, table_name: str, column_types: dict | None) None
    +upsert(df: DataFrame, table_name: str, key_column: str, column_types: dict | None) None
    +append(df: DataFrame, table_name: str, key_column: str, column_types: dict | None) None
    -_table_exists(table_name: str) bool
    -_schema_differs(staging_view: str, table_name: str) bool
  }

  class BronzeWritersMixin {
    -_con: DuckDBPyConnection
    +_full_load_table(records: list, table_name: str) None
    +_incremental_load(records: list, table_name: str, key_column: str) None
    +_snapshot(records: list, key_column: str, history_table: str, fields: list | None) None
  }

  class BronzeStorage {
    -_con: DuckDBPyConnection
    +__init__(bronze_datadb_path: str) None
    +populate(results: dict) None
    +daily_update(results: dict) None
    +seed_historical_prices(records: list) None
    +close() None
    -_process_sources(results: dict, update: bool) None
  }

  class BaseStorage {
    #_open_connection(db_path: str, read_only: bool)$ DuckDBPyConnection
    +close()* None
    +__enter__() Self
    +__exit__(...) None
  }

  BronzeStorage --|> BronzeWritersMixin : inherits
  BronzeStorage --|> BaseStorage : inherits
  BronzeWritersMixin --> DuckDBWriter : delegates writes
  
  note for DuckDBWriter "Unified write primitives: DROP+CREATE, DELETE+INSERT by key, and anti-join dedup append"
  note for BronzeStorage "Orchestrates source routing and historical price backfill via STORAGE_CONFIG"
  note for BronzeWritersMixin "Pydantic → DataFrame conversion, snapshot pre-processing"
```

## Class Responsibilities

| Class | Responsibility |
|-------|-----------------|
| **DuckDBWriter** | Provide three low-level write patterns (full_load, upsert, append) that handle PyArrow type casting, staging table registration, and DuckDB error translation. |
| **BronzeWritersMixin** | Convert Pydantic records to DataFrames and perform snapshot preprocessing (field selection, snapshot_date injection) before delegating to DuckDBWriter. |
| **BronzeStorage** | Orchestrate source-to-table routing via STORAGE_CONFIG, manage the DuckDB connection lifecycle, and perform one-time historical price backfill. |
| **BaseStorage** | Provide static connection opening, context-manager protocol, and error translation for StorageConnectionError. |

## Write Patterns

**full_load** — `DROP TABLE IF EXISTS` + `CREATE TABLE` (initial populate or full rebuild)
- Used when source is not yet in the database or a complete rebuild is required
- Intended for initial BronzeStorage.populate() calls
- Delegates to DuckDBWriter.full_load()

**upsert** — `DELETE FROM table WHERE key IN (SELECT key FROM staging)` + `INSERT INTO table`
- Used for daily updates when current-state tables should replace rows
- Sources marked `incremental=True` in STORAGE_CONFIG use this pattern in daily_update mode
- Allows schema evolution without migration steps (new/removed columns are handled)
- Delegates to DuckDBWriter.upsert()

**append** — `LEFT JOIN anti-join` to `INSERT INTO table`
- Used for history/snapshot tables that must accumulate rows and never lose data
- Deduplicates via LEFT JOIN on (key_column, snapshot_date) so identical snapshots are skipped
- Multiple calls per day are idempotent
- Delegates to DuckDBWriter.append()
