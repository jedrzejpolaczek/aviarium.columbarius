# C4 — GoldStorage Code

GoldStorage orchestrates the transformation of Silver layer data into Gold layer features, signals, and ML datasets. It uses specialized builders to construct card features, price features, demand signals, ban/format events, and the final ML training frame with t+7 price targets.

```mermaid
classDiagram
    class TransformStorage {
        <<abstract>>
        +populate() None
        +update() None
        #_pipeline(update: bool) None*
    }

    class GoldStorage {
        -_silver_con: DuckDBPyConnection
        -_gold_con: DuckDBPyConnection
        -_features: GoldFeatureBuilders
        -_signals: GoldSignalBuilders
        -_writer: GoldWriter
        -_ml: GoldMLDatasetBuilder
        +_pipeline(update: bool) None
        +close() None
    }

    class GoldFeatureBuilders {
        +build_card_features() DataFrame
        +build_price_features() DataFrame
        +build_language_premiums() DataFrame
    }

    class GoldSignalBuilders {
        +build_demand_signals() DataFrame
        +build_events() DataFrame
        +build_format_staples() DataFrame
        +build_ban_price_impact() DataFrame
        +build_tournament_signals() DataFrame
    }

    class GoldMLDatasetBuilder {
        +build_ml_dataset() DataFrame
    }

    class GoldWriter {
        +full_load(df: DataFrame, table_name: str) None
    }

    GoldStorage --|> TransformStorage
    GoldStorage --> GoldFeatureBuilders
    GoldStorage --> GoldSignalBuilders
    GoldStorage --> GoldWriter
    GoldStorage --> GoldMLDatasetBuilder
```

## Class Responsibilities

| Class | Responsibility |
|-------|-----------------|
| **TransformStorage** | Abstract base defining populate/update contract; subclasses implement _pipeline for specific data layers |
| **GoldStorage** | Orchestrates full Silver→Gold transformation: manages connections, coordinates builders, calls writers |
| **GoldFeatureBuilders** | Constructs card and price features with lag/window transformations from Silver |
| **GoldSignalBuilders** | Builds event and demand signals: bans, tournaments, format staples, demand indicators |
| **GoldMLDatasetBuilder** | Joins features and signals into final ML training frame with t+7 price targets |
| **GoldWriter** | Writes DataFrames to Gold layer tables (always full_load, never incremental) |

## Gold Tables Produced

| Table | Builder | Description |
|-------|---------|-------------|
| gold_card_features | GoldFeatureBuilders | Card metadata features |
| gold_price_features | GoldFeatureBuilders | Price + lag features |
| gold_language_premiums | GoldFeatureBuilders | Non-English price premiums |
| gold_demand_signals | GoldSignalBuilders | Meta demand indicators |
| gold_events | GoldSignalBuilders | Ban/unban events |
| gold_format_staples | GoldSignalBuilders | Format staple rankings |
| gold_ban_price_impact | GoldSignalBuilders | Price impact of bans |
| gold_tournament_signals | GoldSignalBuilders | Tournament performance signals |
| gold_ml_dataset | GoldMLDatasetBuilder | Full ML training frame with t+7 targets |

## Why Full Rebuild for Both populate() and update()

Gold layer features leverage window functions (lag, moving averages, cumulative sums) that span the full price history. These window-based features cannot be incrementally patched: when new price data arrives, historical window values change retroactively (e.g., a 7-day moving average for day N changes when day N+1's price is added). Consequently, both `populate()` and `update()` execute the complete `_pipeline()` to ensure consistency—partial updates would leave stale windowed features in the database.
