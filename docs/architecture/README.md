# Architecture Documentation

## Project Overview

`aviarium.columbarius` is a Magic: The Gathering card price prediction system that fetches card and pricing data from multiple sources (Scryfall, MTGJson, MTGGoldfish, MTGTop8), processes it through a medallion architecture (Bronze → Silver → Gold layers in DuckDB), trains a LightGBM predictive model, and exposes predictions via a FastAPI REST API with experiment tracking through MLflow.

## How to Read This Documentation

This documentation follows the **C4 Model**, a standardized approach to visualizing software architecture across four hierarchical levels:

- **C1: System Context** — Shows the system as a whole and its external interactions
- **C2: Containers** — Breaks down the system into major components and their communication
- **C3: Components** — Details the internal structure within each container
- **C4: Code** — Drills down into specific code modules and their interactions

Start at C1 for the big picture, then descend into the level of detail you need.

---

## Documentation Index

### C1: System Context
Shows how the system fits into the wider landscape and external integrations.

- [System Context Diagram](c1/system-context.md)

### C2: Containers
The major building blocks: data pipeline, ML system, API, monitoring, and their communication.

- [Containers Overview](c2/containers.md)

### C3: Components
Internal structure of each container, showing how responsibilities are divided.

| Component | Description |
|-----------|-------------|
| [Data Pipeline](c3/data-pipeline.md) | Card and pricing data fetching, validation, and ingestion |
| [ML System](c3/ml-system.md) | Feature engineering, model training, prediction, and experiment tracking |
| [API](c3/api.md) | FastAPI endpoints, request handling, response serialization |
| [Monitoring](c3/monitoring.md) | Logging, metrics collection, and health checks |

### C4: Code
Deep dives into specific modules and their internal structure.

| Module | Description |
|--------|-------------|
| [BronzeStorage](c4/bronze-storage.md) | Raw data layer implementation |
| [SilverStorage](c4/silver-storage.md) | Cleaned and validated data layer |
| [GoldStorage](c4/gold-storage.md) | Aggregated and feature-rich data layer |
| [LightGBM Model](c4/lightgbm-model.md) | Training, validation, and inference code |
| [API State & Startup](c4/api-state.md) | API initialization and state management |
| [Monitoring Modules](c4/monitoring-modules.md) | Logging and metrics implementations |

---

**For questions on specific topics**: Use Ctrl+F or your search function to find the section you need, or browse the C-level that matches your question granularity.
