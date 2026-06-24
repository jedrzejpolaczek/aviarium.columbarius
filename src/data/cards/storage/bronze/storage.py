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


_MTGJSON_PRICE_MAP: dict[str, tuple[str, str, str]] = {
    "cardmarket_eur": ("cardmarket", "retail", "normal"),
    "cardmarket_eur_foil": ("cardmarket", "retail", "foil"),
    "cardmarket_buylist_eur": ("cardmarket", "buylist", "normal"),
    "tcgplayer_usd": ("tcgplayer", "retail", "normal"),
    "tcgplayer_usd_foil": ("tcgplayer", "retail", "foil"),
    "tcgplayer_buylist_usd": ("tcgplayer", "buylist", "normal"),
}


def _extract_mtgjson_scalar_prices(
    paper_dict: dict | None, target_date: str
) -> dict[str, float | None]:
    """Extract scalar FLOAT price columns from a paper price dict.

    Uses look-back semantics: selects the most recent price for each column
    where the date key is <= target_date. Appropriate for daily snapshots
    (AllPricesToday.json data has only today's key). For seeding historical
    data with multi-date dicts, use inline exact-date extraction instead
    (see seed_historical_prices).
    """
    result: dict[str, float | None] = {col: None for col in _MTGJSON_PRICE_MAP}
    if not paper_dict:
        return result
    for col, (retailer, tx_type, finish) in _MTGJSON_PRICE_MAP.items():
        prices = (
            ((paper_dict.get(retailer) or {}).get(tx_type) or {}).get(finish) or {}
        )
        candidates = {k: v for k, v in prices.items() if k <= target_date}
        result[col] = float(candidates[max(candidates)]) if candidates else None
    return result


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
        AllPricesToday.json). Each card's paper price dict contains up to 90
        date-keyed entries; this method expands them so that
        bronze_mtgjson_prices_history gets one row per (uuid, date) with scalar
        FLOAT price columns.

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
            paper = dump.get("paper") or {}

            dates: set[str] = set()
            for retailer_data in paper.values():
                for tx_type in ("buylist", "retail"):
                    listing = (retailer_data or {}).get(tx_type) or {}
                    dates.update((listing.get("foil") or {}).keys())
                    dates.update((listing.get("normal") or {}).keys())

            for d in dates:
                scalars: dict[str, float | None] = {}
                for col, (retailer, tx_type, finish) in _MTGJSON_PRICE_MAP.items():
                    prices = (
                        ((paper.get(retailer) or {}).get(tx_type) or {}).get(finish) or {}
                    )
                    val = prices.get(d)
                    scalars[col] = float(val) if val is not None else None
                rows.append({"uuid": uuid_str, "snapshot_date": d, **scalars})

        if not rows:
            logger.warning("No date-keyed prices found in records — skipping seed")
            return

        DuckDBWriter(self._con).append(pd.DataFrame(rows), history_table, "uuid")
        logger.info("Seeded %d historical price rows into %r", len(rows), history_table)

    def _snapshot_scryfall_prices(self, records: list[BaseModel]) -> None:
        """Snapshot today's Scryfall prices into bronze_scryfall_prices_history.

        Extracts scalar FLOAT price columns (eur, eur_foil, usd, usd_foil) from
        each record's prices dict. tix is excluded per ADR-012 (physical cards only).
        Null string values produce NULL float columns. Duplicate (id, snapshot_date)
        pairs are silently skipped, making the call idempotent.

        Args:
            records: Pydantic model instances with id and prices fields.
        """
        if not records:
            logger.warning("No Scryfall records to snapshot prices for — skipping")
            return

        today_iso = date.today().isoformat()
        rows = []
        for record in records:
            dump = record.model_dump(mode="json")
            prices = dump.get("prices") or {}
            rows.append(
                {
                    "id": dump["id"],
                    "snapshot_date": today_iso,
                    "eur": float(prices["eur"]) if prices.get("eur") is not None else None,
                    "eur_foil": float(prices["eur_foil"]) if prices.get("eur_foil") is not None else None,
                    "usd": float(prices["usd"]) if prices.get("usd") is not None else None,
                    "usd_foil": float(prices["usd_foil"]) if prices.get("usd_foil") is not None else None,
                }
            )

        df = pd.DataFrame(rows)
        logger.progress("Snapshotting %d Scryfall price rows", len(df))
        self._writer.append(df, "bronze_scryfall_prices_history", "id")
        logger.info("Snapshotted %d Scryfall price rows for %s", len(rows), today_iso)

    def _snapshot_mtgjson_prices(self, records: list[BaseModel]) -> None:
        """Snapshot today's MTGJson prices into bronze_mtgjson_prices_history.

        Extracts scalar FLOAT price columns from each record's paper dict using
        _extract_mtgjson_scalar_prices (look-back semantics), then appends one
        row per card with snapshot_date set to today. Duplicate (uuid,
        snapshot_date) pairs are silently skipped, making the call idempotent.

        Args:
            records: Pydantic model instances with uuid and paper fields.
        """
        if not records:
            logger.warning("No MTGJson price records to snapshot — skipping")
            return

        today_iso = date.today().isoformat()
        rows = []
        for record in records:
            dump = record.model_dump(mode="json")
            rows.append(
                {
                    "uuid": dump["uuid"],
                    "snapshot_date": today_iso,
                    **_extract_mtgjson_scalar_prices(dump.get("paper"), today_iso),
                }
            )

        df = pd.DataFrame(rows)
        logger.progress("Snapshotting %d MTGJson price rows", len(df))
        self._writer.append(df, "bronze_mtgjson_prices_history", "uuid")
        logger.info("Snapshotted %d MTGJson price rows for %s", len(rows), today_iso)

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

        After the config loop, _snapshot_scryfall_prices captures today's
        Scryfall prices as scalars, and seed_historical_prices backfills
        the 90-day MTGJson price history from AllPrices.json.

        Args:
            results: Output of ingesting_pipeline — maps source type to
                a (records, errors) tuple.
        """
        logger.info("Starting DuckDB populate")
        self._process_sources(results, update=False)

        scryfall_records, _ = results.get("scryfall", ([], []))
        try:
            self._snapshot_scryfall_prices(scryfall_records)
        except StorageWriteError as e:
            logger.error(
                "Scryfall price snapshot failed during populate: %s — skipping",
                e,
                exc_info=True,
            )

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
        Price snapshots use dedicated methods (_snapshot_scryfall_prices and
        _snapshot_mtgjson_prices) called after _process_sources.

        Args:
            results: Output of ingesting_pipeline — maps source type to
                a (records, errors) tuple.
        """
        logger.info("Starting DuckDB update")
        self._process_sources(results, update=True)

        scryfall_records, _ = results.get("scryfall", ([], []))
        self._snapshot_scryfall_prices(scryfall_records)

        mtgjson_records, _ = results.get("mtgjson_prices", ([], []))
        self._snapshot_mtgjson_prices(mtgjson_records)
