# Data Catalog

Reference documentation for the aviarium.columbarius data model. Use this catalog to answer questions about what data exists, what it means, and how it flows through the pipeline — without reading source code.

| File | Purpose | When to read |
|------|---------|--------------|
| [glossary.md](glossary.md) | Domain term definitions | When a term in a column name, ADR, or code comment is unclear |
| [table-schemas.md](table-schemas.md) | Per-table column schemas for all 22 DuckDB tables across Bronze, Silver, and Gold layers | When you need to know what columns a table has, its grain, or update strategy |
| [data-lineage.md](data-lineage.md) | Cross-layer data flow: external sources → Bronze → Silver → Gold | When you need to know where a column originates or how it is transformed |

**22 tables total:** 7 Bronze · 6 Silver · 9 Gold.

See also: [Architecture overview](../README.md) (C4 model) · [ADR-003: Medallion Architecture](../../adr/ADR-003-medallion-architecture.md)
