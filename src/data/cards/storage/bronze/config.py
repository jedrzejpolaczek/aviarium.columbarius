"""Storage configuration declarations for the Bronze (raw ingestion) tier.

STORAGE_CONFIG is the single place that describes how every data source is
persisted: whether it gets a main table, which key drives upserts, and which
history snapshots to maintain. Adding a source requires only a new entry here.
"""

from dataclasses import dataclass, field


@dataclass
class SnapshotConfig:
    """Configuration for a single daily history snapshot."""

    history_table: str
    fields: list[str] | None = None


@dataclass
class SourceStorageConfig:
    """Storage configuration for one data source."""

    table: str | None
    key: str
    incremental: bool = True
    snapshots: list[SnapshotConfig] = field(default_factory=list)


STORAGE_CONFIG: dict[str, SourceStorageConfig] = {
    "scryfall": SourceStorageConfig(
        table="bronze_scryfall_cards",
        key="id",
        snapshots=[
            # bronze_scryfall_prices_history → handled by _snapshot_scryfall_prices
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
