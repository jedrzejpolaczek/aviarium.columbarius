# C3 — Data Pipeline Components

The Data Pipeline container orchestrates the ingestion and transformation of Magic: The Gathering card data through three sequential storage layers: Bronze (raw), Silver (cleaned), and Gold (ML-ready). It coordinates multiple specialized components that handle data fetching from external sources, validation against Pydantic schemas, and layer-specific transformations driven by configuration.

```mermaid
C4Component
  title Component diagram for Data Pipeline

  System_Ext(external_sources, "External Data Sources", "Scryfall, MTGJson, MTGGoldfish, MTGTop8")
  ContainerDb_Ext(bronze_db, "Bronze DB", "DuckDB file")
  ContainerDb_Ext(silver_db, "Silver DB", "DuckDB file")
  ContainerDb_Ext(gold_db, "Gold DB", "DuckDB file")

  Container_Boundary(dp, "Data Pipeline") {
    Component(orchestrator, "PipelineOrchestrator", "", "Entry point: runs initial_pipeline (full load) or daily_pipeline (incremental)")
    Component(source_registry, "SourceRegistry", "", "Registry of all scrapers and extractors — maps source names to handler classes")
    Component(http_client, "HttpClient", "", "HTTP client with exponential backoff retry logic")
    Component(scrapers, "Scrapers", "", "MTGGoldfish and MTGTop8 HTML scrapers")
    Component(validators, "PydanticValidators", "", "Pydantic v2 models validating incoming Scryfall/MTGJson schemas")
    Component(bronze_storage, "BronzeStorage", "", "Writes validated records to Bronze DB — initial populate and daily update")
    Component(silver_storage, "SilverStorage", "", "Reads Bronze DB, applies config-driven transformations, writes Silver DB")
    Component(gold_storage, "GoldStorage", "", "Builds ML-ready features from Silver DB and writes Gold DB")
  }

  Rel(orchestrator, source_registry, "Looks up handlers for each source")
  Rel(source_registry, http_client, "Uses for HTTP fetching")
  Rel(source_registry, scrapers, "Delegates HTML scraping")
  Rel(http_client, external_sources, "Fetches JSON data")
  Rel(scrapers, external_sources, "Scrapes HTML pages")
  Rel(http_client, validators, "Raw JSON → validated records")
  Rel(scrapers, validators, "Raw HTML data → validated records")
  Rel(validators, bronze_storage, "Writes validated data")
  Rel(bronze_storage, bronze_db, "persist")
  Rel(orchestrator, silver_storage, "Runs silver transforms")
  Rel(silver_storage, bronze_db, "Reads raw data")
  Rel(silver_storage, silver_db, "Writes clean data")
  Rel(orchestrator, gold_storage, "Builds ML features")
  Rel(gold_storage, silver_db, "Reads clean data")
  Rel(gold_storage, gold_db, "Writes ML-ready features")
```

## Components

| Component | Responsibility | ADR References |
|---|---|---|
| **PipelineOrchestrator** | Entry point orchestrating the full data pipeline execution; delegates to storage layer components for initial load or daily incremental updates | ADR-005 |
| **SourceRegistry** | Central registry mapping external source names to handler classes; abstracts scraper and extractor selection logic | ADR-004 |
| **HttpClient** | Handles all HTTP requests to external APIs with exponential backoff retry logic to manage rate limits and transient failures | ADR-014 |
| **Scrapers** | Specialized HTML parsers for MTGGoldfish and MTGTop8; extract structured data from web pages where APIs unavailable | (ADR-004 via SourceRegistry) |
| **PydanticValidators** | Validates incoming raw data against Pydantic v2 schemas for Scryfall, MTGJson, and custom formats; enforces data shape before persistence | ADR-001 |
| **BronzeStorage** | Persists validated records to Bronze layer; handles both initial full load and daily incremental updates from all external sources | ADR-005 |
| **SilverStorage** | Reads raw Bronze data, applies config-driven transformation rules, and writes cleaned, deduplicated records to Silver layer | ADR-008 |
| **GoldStorage** | Builds ML-ready feature sets from Silver layer data; computes derived columns and encodings required for model training | ADR-003 |

## Pipeline Modes

The PipelineOrchestrator supports two execution modes:

**initial_pipeline**: Performs a full load from all external sources, clearing previous Bronze/Silver/Gold data and rebuilding each layer from scratch. Used during setup or when full data refresh is required.

**daily_pipeline**: Incremental update mode that fetches only new or modified data from external sources since the last run, updates Bronze layer with incremental changes, and cascades transformations through Silver and Gold layers. Designed for efficiency and scheduled daily execution.
