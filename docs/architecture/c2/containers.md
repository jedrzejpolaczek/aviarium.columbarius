# C2 — Containers

The system decomposes into six core containers: a data pipeline for ETL, a FastAPI server for REST queries, and three DuckDB databases representing the Bronze-Silver-Gold data maturity model, plus MLflow for ML experiment tracking and model versioning.

```mermaid
C4Container
  title Container diagram for aviarium.columbarius
  
  Person(dev_analyst, "Developer/Analyst", "Runs pipeline, trains models")
  Person(end_user, "End User", "Queries API for predictions")
  System_Ext(data_sources, "External Data Sources", "Scryfall API, MTGJson, MTGGoldfish, MTGTop8")
  
  System_Boundary(sb, "aviarium.columbarius") {
    Container(data_pipeline, "Data Pipeline", "Python 3.13, uv", "Ingest from external sources; process through Bronze → Silver → Gold transformations")
    Container(fastapi_server, "FastAPI Server", "FastAPI, uvicorn", "REST API exposing /predict, /similar, /underpriced, /cards, /health endpoints")
    ContainerDb(bronze_db, "Bronze Database", "DuckDB file", "Raw validated data: bronze_cards, bronze_prices_snapshot, bronze_meta")
    ContainerDb(silver_db, "Silver Database", "DuckDB file", "Cleaned and joined flat data: silver_cards, silver_prices_history, silver_meta_history")
    ContainerDb(gold_db, "Gold Database", "DuckDB file", "ML-ready features and outputs: gold_card_features, gold_price_features, gold_predictions, gold_events")
    Container(mlflow, "MLflow Server", "MLflow tracking server", "Experiment tracking, model registry, model aliases")
  }
  
  Rel(dev_analyst, data_pipeline, "Runs initial_pipeline / daily_pipeline")
  Rel(end_user, fastapi_server, "GET /predict, /similar, /underpriced, /cards")
  Rel(data_pipeline, data_sources, "Fetches and scrapes card/price data")
  Rel(data_pipeline, bronze_db, "Writes validated raw data")
  Rel(bronze_db, silver_db, "Pipeline reads bronze, writes silver")
  Rel(silver_db, gold_db, "Pipeline reads silver, writes gold features")
  Rel(gold_db, fastapi_server, "API reads features and predictions at startup")
  Rel(data_pipeline, mlflow, "Logs experiments and registers trained models")
  Rel(fastapi_server, mlflow, "Loads model by alias at startup")
  
  UpdateLayoutConfig($c4ShapeInRow="3", $c4BoundaryInRow="1")
```

## Containers

| Container | Technology | Purpose |
|---|---|---|
| **Data Pipeline** | Python 3.13, uv | Orchestrates ingest, validation, and multi-stage transformation; sources external APIs and processes raw data through Bronze/Silver/Gold stages; logs experiments to MLflow |
| **FastAPI Server** | FastAPI, uvicorn | Exposes REST API endpoints for prediction, similarity search, underpriced cards, and card metadata queries; loads ML models from MLflow at startup |
| **Bronze Database** | DuckDB file | Immutable raw data layer: bronze_cards, bronze_prices_snapshot, bronze_meta; serves as single source of truth for validated ingest |
| **Silver Database** | DuckDB file | Cleaned, normalized, and joined data layer; denormalizes bronze data into flat schemas for analytical use |
| **Gold Database** | DuckDB file | Feature-engineered and ML-ready layer; stores gold_card_features, gold_price_features, gold_predictions, and gold_events for model serving |
| **MLflow Server** | MLflow tracking server | Central registry for experiment runs, metrics, trained models, and model aliases; enables reproducibility and model governance |

## Data Flow

1. **Data Ingestion**: Developer/Analyst triggers the Data Pipeline, which fetches card/price data from external sources (Scryfall, MTGJson, MTGGoldfish, MTGTop8).

2. **Bronze Stage**: Validated raw data is written to the Bronze Database (bronze_cards, bronze_prices_snapshot, bronze_meta) as immutable historical record.

3. **Silver Stage**: The Data Pipeline reads from Bronze, applies cleaning and normalization, and writes joined flat tables to the Silver Database (silver_cards, silver_prices_history, silver_meta_history).

4. **Gold Stage**: The Data Pipeline reads from Silver, engineers features, and writes ML-ready data to the Gold Database (gold_card_features, gold_price_features) alongside predictions and events.

5. **Model Training**: The Data Pipeline logs experiment runs, metrics, and trained models to MLflow, registering models with aliases for easy promotion.

6. **API Serving**: At startup, the FastAPI Server loads the active model from MLflow and reads Gold Database tables (gold_card_features, gold_predictions). End Users query the API for predictions (/predict), similar cards (/similar), underpriced opportunities (/underpriced), and card metadata (/cards).
