"""BronzeStorage — the orchestration layer for Bronze-tier persistence.

Combines BronzeWritersMixin (generic DuckDB write primitives) with
BaseStorage (connection management) into a single public class. All
source-to-table routing is driven by STORAGE_CONFIG; seed_historical_prices
handles the one-time AllPrices.json backfill.

Typical usage:
    with BronzeStorage("data/bronze/cards.duckdb") as storage:
        storage.populate(results)    # initial load
        # or
        storage.daily_update(results)  # incremental daily run
"""

from collections.abc import Sequence
from datetime import date
from typing import Any

import pandas as pd
from pydantic import BaseModel

from src.data.cards.storage.base.storage import BaseStorage
from src.data.cards.storage.base.writers import DuckDBWriter
from src.data.cards.storage.bronze.config import STORAGE_CONFIG
from src.data.cards.storage.errors import StorageWriteError
from src.logger import get_logger

logger = get_logger(__name__)


def _records_to_df(records: Sequence[BaseModel]) -> pd.DataFrame:
    return pd.DataFrame([r.model_dump(mode="json") for r in records])


def _filter_prices_to_date(
    platform_prices: dict[str, Any] | None, target_date: str
) -> dict[str, Any] | None:
    if not platform_prices:
        return None
    result: dict[str, Any] = {}
    for retailer, retailer_data in platform_prices.items():
        filtered_retailer: dict[str, Any] = {}
        if "currency" in retailer_data:
            filtered_retailer["currency"] = retailer_data["currency"]
        for tx_type in ("buylist", "retail"):
            listing = retailer_data.get(tx_type) or {}
            filtered_listing: dict[str, Any] = {}
            for finish in ("foil", "normal"):
                prices = listing.get(finish) or {}
                if target_date in prices:
                    filtered_listing[finish] = {target_date: prices[target_date]}
            if filtered_listing:
                filtered_retailer[tx_type] = filtered_listing
        if any(k in filtered_retailer for k in ("buylist", "retail")):
            result[retailer] = filtered_retailer
    return result or None


class BronzeStorage(BaseStorage):
    """Persistence layer for the Bronze (raw ingestion) tier.

    Inherits generic write operations from BronzeWritersMixin and connection
    management from BaseStorage. All sources and their storage rules are
    declared in STORAGE_CONFIG — adding a source requires only a new entry
    there, no code changes.

    Usage:
        with BronzeStorage("data/bronze/cards.duckdb") as storage:
            storage.populate(results)     # initial load or full rebuild
            storage.daily_update(results) # incremental daily run

    Raises:
        StorageConnectionError: If the DuckDB connection cannot be opened.
    """

    def __init__(self, bronze_datadb_path: str) -> None:
        """Open (or create) a DuckDB database.

        Args:
            bronze_datadb_path: File path for the database, or ":memory:" for
                an in-memory database that is lost when the connection closes.

        Raises:
            StorageConnectionError: If the connection cannot be established.
        """
        self._con = self._open_connection(bronze_datadb_path, read_only=False)
        self._writer = DuckDBWriter(self._con)

    def close(self) -> None:
        """Close the underlying DuckDB connection."""
        logger.progress("Closing connection to DuckDB")
        self._con.close()

    def _full_load_table(self, records: list[BaseModel], table_name: str) -> None:
        if not records:
            logger.warning("No records to save to %r — skipping", table_name)
            return
        df = _records_to_df(records)
        self._writer.full_load(df, table_name)

    def _incremental_load(
        self, records: list[BaseModel], table_name: str, key_column: str
    ) -> None:
        if not records:
            logger.warning("No records to load into %r — skipping", table_name)
            return
        df = _records_to_df(records)
        self._writer.upsert(df, table_name, key_column)

    def _snapshot(
        self,
        records: list[BaseModel],
        key_column: str,
        history_table: str,
        fields: list[str] | None = None,
    ) -> None:
        if not records:
            logger.warning("No records to snapshot into %r — skipping", history_table)
            return

        today_iso = date.today().isoformat()
        rows = []
        for record in records:
            dump = record.model_dump(mode="json")
            data = (
                {f: dump[f] for f in fields if f in dump}
                if fields is not None
                else dump
            )
            rows.append(
                {
                    key_column: dump[key_column],
                    "snapshot_date": today_iso,
                    **data,
                }
            )

        if not rows:
            logger.warning("No data to snapshot into %r — skipping", history_table)
            return

        df = pd.DataFrame(rows)
        logger.progress("Snapshotting %d rows into %r", len(df), history_table)
        self._writer.append(df, history_table, key_column)
        logger.info(
            "Snapshotted %d rows into %r for %s", len(rows), history_table, today_iso
        )

    def seed_historical_prices(self, records: list[BaseModel]) -> None:
        """One-time seeding: explode AllPrices 90-day history into per-date rows.

        Reads MtgjsonCardPrices instances from AllPrices.json (not
        AllPricesToday.json). Each card's price dict contains up to 90
        date-keyed entries; this method expands them so that
        bronze_mtgjson_prices_history gets one row per (uuid, date). Each row
        contains only that date's prices, matching the shape of daily snapshot
        rows produced by _snapshot() from AllPricesToday.json.

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

            dates: set[str] = set()
            for platform in ("paper", "mtgo"):
                for retailer_data in (dump.get(platform) or {}).values():
                    for tx_type in ("buylist", "retail"):
                        listing = retailer_data.get(tx_type) or {}
                        dates.update((listing.get("foil") or {}).keys())
                        dates.update((listing.get("normal") or {}).keys())

            for d in dates:
                rows.append(
                    {
                        "uuid": uuid_str,
                        "snapshot_date": d,
                        "paper": _filter_prices_to_date(dump.get("paper"), d),
                        "mtgo": _filter_prices_to_date(dump.get("mtgo"), d),
                    }
                )

        if not rows:
            logger.warning("No date-keyed prices found in records — skipping seed")
            return

        DuckDBWriter(self._con).append(pd.DataFrame(rows), history_table, "uuid")
        logger.info("Seeded %d historical price rows into %r", len(rows), history_table)

    def _process_sources(
        self,
        results: dict[str, tuple[list[BaseModel], list[dict[str, object]]]],
        update: bool = False,
    ) -> None:
        """Apply the configured write strategy for every source in STORAGE_CONFIG.

        Iterates STORAGE_CONFIG and, for each source present in results, writes
        the records to the appropriate table and appends any configured snapshots.
        Errors from one source are logged and do not prevent other sources from
        being processed.

        Args:
            results: Output of ingesting_pipeline — maps source type to a
                (records, errors) tuple.
            update: When False (populate mode) full replace is always used.
                When True (daily_update mode) sources flagged incremental=True
                are upserted instead of replaced.
        """
        action = "daily_update" if update else "populate"
        for source_type, config in STORAGE_CONFIG.items():
            records, _ = results.get(source_type, ([], []))
            write_mode = (
                "incremental" if (update and config.incremental) else "full-replace"
            )
            logger.progress(
                "[%s] source=%r  records=%d  mode=%s",
                action,
                source_type,
                len(records),
                write_mode,
            )
            try:
                if config.table:
                    if update and config.incremental:
                        self._incremental_load(
                            records, config.table, key_column=config.key
                        )
                    else:
                        self._full_load_table(records, config.table)
                for snap in config.snapshots:
                    self._snapshot(
                        records,
                        key_column=config.key,
                        history_table=snap.history_table,
                        fields=snap.fields,
                    )
            except StorageWriteError as e:
                logger.error(
                    "Source %r failed during %s: %s — skipping",
                    source_type,
                    action,
                    e,
                    exc_info=True,
                )

    def populate(
        self, results: dict[str, tuple[list[BaseModel], list[dict[str, object]]]]
    ) -> None:
        """Full load of all sources into Bronze tables and an initial snapshot.

        Intended for the initial database population or a full rebuild.
        All Bronze tables are dropped and recreated via _process_sources.
        Snapshot history tables are created automatically on first call.

        After the config loop, seed_historical_prices is called to backfill
        the 90-day price history from AllPrices.json (if mtgjson_prices is
        present in results).

        Args:
            results: Output of ingesting_pipeline — maps source type to
                a (records, errors) tuple.
        """
        logger.info("Starting DuckDB populate")
        self._process_sources(results, update=False)

        prices_records, _ = results.get("mtgjson_prices", ([], []))
        try:
            self.seed_historical_prices(prices_records)
        except StorageWriteError as e:
            logger.error(
                "Historical price seed failed during populate: %s — skipping",
                e,
                exc_info=True,
            )

    def daily_update(
        self, results: dict[str, tuple[list[BaseModel], list[dict[str, object]]]]
    ) -> None:
        """Incrementally update card data and append a daily snapshot.

        Intended to be run once per day after the initial populate call.
        Sources marked incremental=True are upserted; others are fully replaced.
        Snapshot history tables accumulate one row per card per day.

        Sources and their write strategy are declared in STORAGE_CONFIG via
        _process_sources. If one source fails the others are still processed.

        Args:
            results: Output of ingesting_pipeline — maps source type to
                a (records, errors) tuple.
        """
        logger.info("Starting DuckDB update")
        self._process_sources(results, update=True)
