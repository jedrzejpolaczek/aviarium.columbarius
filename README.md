# aviarium.columbarius

![CI](https://github.com/jpolaczek/aviarium.columbarius/actions/workflows/ci.yml/badge.svg)

**aviarium.columbarius** is a Magic: The Gathering card price prediction system. It ingests and stores raw card and pricing data from Scryfall and MTGJson, cleans and joins them in a Silver tier, and will grow to include feature engineering and an ML model for predicting card prices.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [File Structure](#file-structure)
4. [Requirements](#requirements)
5. [Setup](#setup)
6. [Configuration](#configuration)
7. [Usage](#usage)
8. [Model Training](#model-training)
9. [Results](#results)
10. [Monitoring & Scheduled Retraining](#monitoring--scheduled-retraining)
11. [API & UI](#api--ui)
12. [Data Catalog](#data-catalog)
13. [Testing](#testing)
14. [Architecture Decision Records](#architecture-decision-records)
15. [References](#references)

---

## Overview

| Property | Value |
|---|---|
| Package | `columbarius` |
| Python | ≥ 3.13 |
| Package manager | [uv](https://docs.astral.sh/uv/) |
| Storage | DuckDB |
| Validation | Pydantic v2 |
| Status | Bronze complete · Silver complete · Gold complete · ML training complete |

**Data sources:**

| Source | Endpoint | Content |
|---|---|---|
| Scryfall | `bulk-data/all-cards` | Card metadata, prices, legalities |
| MTGJson AllPrintings | `AllPrintings.json` | Every printing across all sets |
| MTGJson AllPricesToday | `AllPricesToday.json` | Current paper/MTGO prices by card UUID |

---

## Architecture

The pipeline follows a **Medallion architecture** (Bronze → Silver → Gold).

```
┌─────────────────────────────────────────────────────────────┐
│  External APIs                                              │
│  Scryfall  ·  MTGJson AllPrintings  ·  MTGJson AllPrices    │
└────────────────────┬────────────────────────────────────────┘
                     │ HTTP download
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  sources.py  (ingesting_pipeline)                           │
│  • Downloads JSON files (controlled by flag in config)      │
│  • Validates every record via Pydantic models               │
│  • Returns (records, errors) per source                     │
└────────────────────┬────────────────────────────────────────┘
                     │ list[BaseModel]
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  BRONZE  —  DuckDB  (data/bronze/cards.duckdb)              │
│  Raw, unmodified records + daily snapshot history           │
│  Config: configs/bronze_config.json                         │
└────────────────────┬────────────────────────────────────────┘
                     │ Bronze DuckDB (read-only)
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  SILVER  —  DuckDB  (data/silver/cards.duckdb)              │
│  Cleaned, joined, normalized records                        │
│  Config: configs/silver_config.json                         │
└────────────────────┬────────────────────────────────────────┘
                     │ Silver DuckDB (read-only)
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  GOLD  —  DuckDB  (data/gold/cards.duckdb)  (done)          │
│  Aggregated views ready for ML                              │
└────────────────────┬────────────────────────────────────────┘
                     │  (trained)
                     ▼
              ML price prediction model
```

**Pipeline phases:**

- `initial_pipeline` — Full load of all tiers; drops and recreates all tables. Run once on initial setup or for a full rebuild.
- `daily_pipeline` — Incremental upsert + snapshot for Bronze, incremental Silver refresh. Run once per day.

---

## File Structure

```
aviarium.columbarius/
├── scripts/
│   ├── run_pipeline.py              # ETL pipeline entry point
│   └── train_model.py               # model training entry point
├── pyproject.toml
├── configs/
│   ├── data_sources.yaml            # source URLs, local paths, download flags
│   ├── bronze_config.json           # Bronze table definitions
│   └── silver_config.json           # Silver transform config
├── data/
│   ├── raw/                         # downloaded JSON files (gitignored)
│   ├── bronze/                      # Bronze DuckDB file (gitignored)
│   ├── silver/                      # Silver DuckDB file (gitignored)
│   └── gold/                        # Gold DuckDB file (gitignored)
├── docs/
│   ├── adr/                         # Architecture Decision Records
│   └── architecture/                # C4 architecture docs + data catalog
├── notebooks/                       # Jupyter notebooks for exploration
├── src/
│   └── data/
│       ├── cards/
│       │   ├── pipelines.py         # initial_pipeline / daily_pipeline
│       │   ├── sources.py           # download, extract, validate
│       │   └── storage/
│       │       ├── base.py          # BaseStorage / TransformStorage ABCs
│       │       ├── bronze/          # BronzeStorage — raw DuckDB persistence
│       │       │   ├── config.py    #   STORAGE_CONFIG declarations
│       │       │   ├── writers.py   #   BronzeWritersMixin (write primitives)
│       │       │   └── storage.py   #   BronzeStorage orchestration class
│       │       ├── silver.py        # SilverStorage — cleaning and joining
│       │       ├── gold.py          # GoldStorage — aggregation (stub)
│       │       └── errors.py        # StorageError hierarchy
│       ├── dataclasses/
│       │   ├── mtgjson.py           # Pydantic models for MTGJson
│       │   └── scryfall.py          # Pydantic models for Scryfall
│       └── markets/
│           ├── allegro.py           # Allegro market integration (stub)
│           └── cardmarket.py        # Cardmarket integration (stub)
└── tests/
    └── data/
        ├── cards/
        │   ├── test_sources.py
        │   ├── test_pipelines.py
        │   └── storage/
        │       ├── test_base.py
        │       └── test_silver.py
        └── dataclasses/
            ├── test_mtgjson.py
            └── test_scryfall.py
```

---

## Requirements

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) (package and environment manager)

---

## Setup

```bash
# 1. Clone the repository
git clone https://github.com/jpolaczek/aviarium.columbarius
cd aviarium.columbarius

# 2. Create the virtual environment and install all dependencies
make install

# 3. Install git hooks (runs pre-push checks before every push)
make install-hooks
```

**Common commands:**

| Command | Description |
|---|---|
| `make install` | Create venv and install all dependencies |
| `make install-hooks` | Register git hooks from `scripts/` |
| `make pipeline` | Run the daily ETL pipeline |
| `make train` | Train the LightGBM model and log to MLflow |
| `make monitor` | Run the drift/MAPE check and conditionally retrain |
| `make lint` | Run `ruff check` |
| `make format` | Run `ruff format` |
| `make type-check` | Run `mypy` |
| `make test` | Run `pytest` |
| `make coverage` | Run `pytest` with coverage report (`src/` + `app/`) |
| `make check` | Run all checks (lint + format + type-check + test) |

> **Note:** `make` requires GNU Make. On Windows without Make installed, use the `uv run python -m ...` commands shown in the sections below.

---

## Configuration

Source URLs and download flags live in `configs/data_sources.yaml`:

```yaml
sources:
  - type: scryfall
    url: "https://api.scryfall.com/bulk-data/all-cards"
    path: "data/raw/scryfall_cards.json"
    flag: false          # set true to re-download from the API

  - type: mtgjson_cards
    url: "https://mtgjson.com/api/v5/AllPrintings.json"
    path: "data/raw/mtgjson_cards.json"
    flag: false

  - type: mtgjson_prices
    url: "https://mtgjson.com/api/v5/AllPricesToday.json"
    path: "data/raw/mtgjson_prices.json"
    flag: false

storage:
  - bronze_duckdb_path: "data/bronze/cards.duckdb"
```

Set `flag: true` for any source you want to download fresh from the API. Set `flag: false` to load from the existing local JSON file without hitting the network.

Per-tier table and transformation rules are in `configs/bronze_config.json` and `configs/silver_config.json`.

---

## Usage

**Initial load** (first run or full rebuild):

```python
# in scripts/run_pipeline.py — uncomment initial_pipeline:
initial_pipeline(config_path)
```

```bash
uv run python -m scripts.run_pipeline
```

Drops and recreates all Bronze and Silver tables, then writes the first snapshot rows to history tables.

**Daily update** (subsequent runs):

```bash
uv run python -m scripts.run_pipeline
```

`scripts/run_pipeline.py` calls `daily_pipeline` by default, which upserts card tables, appends one snapshot row per card to history tables, and refreshes Silver. Calling it more than once on the same day is safe — duplicate snapshots are skipped.

---

## Model Training

Training uses walk-forward cross-validation (time-series safe) and logs every run to MLflow. The full workflow from raw data to a running API:

**Step 1 — Build the Gold layer** (ETL pipeline must have run at least once):

```bash
uv run python -m scripts.run_pipeline
```

**Step 2 — Train the model:**

```bash
uv run python -m scripts.train_model
```

This runs walk-forward CV, trains a final LightGBM model on the latest snapshot, and logs everything to MLflow. On success it prints:

```
============================================================
MODEL_RUN_ID = abc123def456...
============================================================

Set in PowerShell:
  $env:MODEL_RUN_ID = "abc123def456..."

Start the API:
  uv run uvicorn app.main:app --reload
```

**Step 3 — Start the API with the trained model:**

```bash
# Linux / macOS
export MODEL_RUN_ID=abc123def456...
uv run uvicorn app.main:app --reload

# Windows PowerShell
$env:MODEL_RUN_ID = "abc123def456..."
uv run uvicorn app.main:app --reload
```

**Inspect runs in the MLflow UI:**

```bash
uv run mlflow ui
# open http://localhost:5000
```

**Optional — custom Gold DB path:**

```bash
uv run python -m scripts.train_model --db-path path/to/gold/cards.duckdb
```

> **Data requirement:** Walk-forward CV needs at least 50 days of daily snapshots (≥ 3 folds of 30-day train + 7-day validation windows). If fewer snapshots are available, the script skips CV and trains a final model directly.

---

## Results

Exploratory and confirmatory analysis behind the feature set and modelling choices is written up per phase in `notebooks/`:

| Phase | Write-up |
|---|---|
| Exploratory data analysis | [notebooks/exploratory_data_analysis/EDA_FINDINGS.md](notebooks/exploratory_data_analysis/EDA_FINDINGS.md) |
| Confirmatory data analysis | [notebooks/confirmatory_data_analysis/CDA_FINDINGS.md](notebooks/confirmatory_data_analysis/CDA_FINDINGS.md) |
| Statistical properties (stationarity, seasonality, cointegration) | [notebooks/statistical_properties/STAT_FINDINGS.md](notebooks/statistical_properties/STAT_FINDINGS.md) |
| Bayesian analysis | [notebooks/bayesian_analysis/BAYESIAN_FINDINGS.md](notebooks/bayesian_analysis/BAYESIAN_FINDINGS.md) |
| Model preparation (leakage review, validation strategy) | [notebooks/model_preparation/MODEL_PREP_FINDINGS.md](notebooks/model_preparation/MODEL_PREP_FINDINGS.md) |

**Model status:** the LightGBM pipeline trains, logs to MLflow, and serves predictions through the [API](#api--ui) using the tiered pricing strategy from [ADR-018](docs/adr/ADR-018-tier-based-model-selection.md).

> **Known gap:** [`notebooks/ml_models/ML_FINDINGS.md`](notebooks/ml_models/ML_FINDINGS.md) (baseline-vs-LightGBM comparison, SHAP importance) has not been filled in yet, and the most recently logged MLflow runs show suspicious `mae_test = 0.0` metrics that need triage before being quoted anywhere. Treat any specific accuracy number as unverified until that investigation happens — this is flagged here for follow-up investigation, not covered by this section.

---

## Monitoring & Scheduled Retraining

`scripts/check_and_retrain.py` checks for a ban/unban event or a 3-day MAPE alert (see `src/monitoring/retraining.py` (`should_retrain`)) and only retrains when one fires. Run it once a day, after the ETL pipeline:

```bash
make pipeline
make monitor
```

**Linux/macOS (cron)** — run daily at 07:00, after the pipeline:

```cron
0 7 * * * cd /path/to/aviarium.columbarius && make pipeline && make monitor >> logs/cron.log 2>&1
```

**Windows (Task Scheduler)** — create a daily trigger running:

```powershell
uv run python -m scripts.run_pipeline; if ($?) { uv run python -m scripts.check_and_retrain }
```

(Set the task's "Start in (optional)" field to the project root — relative paths like `logs/` and `data/gold/cards.duckdb` won't resolve otherwise.)

Every run writes `logs/last_check_status.json` with one of `no_retrain` / `retrained` / `error`, so the outcome can be checked without reading log files. See [docs/runbooks/model-incidents.md](docs/runbooks/model-incidents.md) for what to do with each result.

---

## API & UI

The price prediction API and its web UI run as Docker containers.

**Prerequisites:** Docker, a trained MLflow model run ID (from `mlflow ui`), and a populated Gold DuckDB (`data/gold/cards.duckdb`).

```bash
# Set the model run ID, then start both containers
export MODEL_RUN_ID=<run_id_from_mlflow>
docker compose -f docker/docker-compose.yml up --build
```

| URL | Description |
|---|---|
| `http://localhost:3000` | Web UI — search for a card and see the price prediction |
| `http://localhost:8000/docs` | Swagger UI — interactive API documentation |
| `http://localhost:8000/health` | Health check endpoint |
| `http://localhost:8000/cards` | List of all cards available for prediction |
| `http://localhost:8000/predict/{card_name}` | Price prediction for a single card |

The API starts in degraded mode if `MODEL_RUN_ID` is not set — `/health` and `/cards` still work, but `/predict` returns 503.

---

## Data Catalog

Full column schemas for all 22 DuckDB tables (7 Bronze · 6 Silver · 9 Gold), domain glossary, and cross-layer data lineage are documented in the data catalog:

**[docs/architecture/data/README.md](docs/architecture/data/README.md)**

---

## Testing

Tests are split into two invocations — see [ADR-026](docs/adr/ADR-026-isolate-mlflow-tracking-tests.md) for why.

```bash
uv run pytest --ignore=tests/ml/training/test_tracking.py
uv run pytest tests/ml/training/test_tracking.py
```

Or simply `make test`, which runs both.

Tests are in `tests/` and mirror the `src/` layout. No network access or real files are required — I/O is covered with `tmp_path` and `unittest.mock`.

**Coverage report** — full project:

```bash
uv run pytest --cov=src --cov=app --cov-report=term-missing
```

For an HTML report you can browse in a browser:

```bash
uv run pytest --cov=src --cov=app --cov-report=html
# open htmlcov/index.html
```

Scoped to a single package (faster):

```bash
uv run pytest tests/ml/ --cov=src/ml --cov-report=term-missing
```

**Coverage report** (scoped to the sources package only):

Current coverage (`src/data/cards/sources/`):

| File | Statements | Cover | Missing lines |
|---|---|---|---|
| `__init__.py` | 5 | 100% | — |
| `errors.py` | 8 | 100% | — |
| `http.py` | 38 | 100% | — |
| `extractors.py` | 97 | 96% | 162, 171, 225, 286 |
| `pipeline.py` | 152 | 99% | 306–307 |
| **Total** | **300** | **98%** | |

The remaining gaps are defensive guards for near-impossible HTML states (empty `<div>` nodes, regex matches that can't fail given the earlier CSS selector, etc.) and one error branch for an individual deck-page download failure inside `_ingest_tournament_results`.

---

## Architecture Decision Records

Design decisions are documented in `docs/adr/`:

| ADR | Decision |
|---|---|
| [ADR-001](docs/adr/ADR-001-pydantic-validation-layer.md) | Pydantic v2 as the data validation layer |
| [ADR-002](docs/adr/ADR-002-duckdb-analytical-store.md) | DuckDB as the analytical store |
| [ADR-003](docs/adr/ADR-003-medallion-architecture.md) | Medallion architecture (Bronze / Silver / Gold) |
| [ADR-004](docs/adr/ADR-004-registry-based-source-extensibility.md) | Registry-based source extensibility |
| [ADR-005](docs/adr/ADR-005-two-phase-pipeline-lifecycle.md) | Two-phase pipeline lifecycle |
| [ADR-006](docs/adr/ADR-006-records-errors-tuple-pattern.md) | Records and errors as co-returned tuples |
| [ADR-007](docs/adr/ADR-007-three-level-configuration-hierarchy.md) | Three-level configuration hierarchy |
| [ADR-008](docs/adr/ADR-008-config-driven-silver-transformations.md) | Config-driven Silver transformations |
| [ADR-009](docs/adr/ADR-009-mtgjson-priority-card-join-strategy.md) | MTGJson-priority card join strategy |
| [ADR-010](docs/adr/ADR-010-mypy-strict-mode-quality-gate.md) | mypy strict mode as a hard quality gate |
| [ADR-011](docs/adr/ADR-011-uv-package-manager.md) | uv as the Python package manager |
| [ADR-012](docs/adr/ADR-012-physical-cards-only.md) | Physical cards only — no digital formats or tix pricing |
| [ADR-013](docs/adr/ADR-013-data-source-selection.md) | Data source selection |
| [ADR-014](docs/adr/ADR-014-http-retry-exponential-backoff.md) | HTTP retry with exponential backoff for download functions |
| [ADR-015](docs/adr/ADR-015-scraping-rights-review.md) | Scraping rights review for external HTML sources |
| [ADR-016](docs/adr/ADR-016-app-src-layer-separation.md) | Separation of `app/` and `src/` layers |
| [ADR-017](docs/adr/ADR-017-lightgbm-over-xgboost-tensorflow.md) | LightGBM as the primary gradient boosting library |
| [ADR-018](docs/adr/ADR-018-tier-based-model-selection.md) | Tier-based model selection strategy |
| [ADR-019](docs/adr/ADR-019-fastapi-startup-precomputation.md) | FastAPI startup pre-computation strategy |
| [ADR-020](docs/adr/ADR-020-monitoring-and-retraining-architecture.md) | Monitoring and automated retraining architecture |
| [ADR-021](docs/adr/ADR-021-duckdb-map-type-for-legalities.md) | DuckDB MAP type for the `legalities` column |
| [ADR-022](docs/adr/ADR-022-gold-layer-tables.md) | Gold layer table design |
| [ADR-023](docs/adr/ADR-023-card-recommendation-strategy.md) | Card recommendation and underpriced detection strategy |
| [ADR-024](docs/adr/ADR-024-duckdb-compute-layer.md) | DuckDB as the compute layer for large Silver history queries |

---

## References

- [ML Project Checklist](https://threere.com/notes/machine-learning/project-checklist/) — Aurélien Géron's 8-step ML project checklist, used as a structural reference for the data and modelling pipeline.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, branching conventions, and PR guidelines.
Please read the [Code of Conduct](CODE_OF_CONDUCT.md) before contributing.

---

## License

This project is licensed under the [MIT License](LICENSE).
