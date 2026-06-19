"""Low-level DuckDB write primitives for the Bronze tier.

Provides _filter_prices_to_date (a pure price-filtering utility),
_records_to_df (a module-level Pydantic → DataFrame helper), and
BronzeWritersMixin, which adds generic write operations to BronzeStorage:
full table replace, incremental upsert, and daily snapshot.

Write operations are delegated to DuckDBWriter from src.data.cards.storage.base.
The mixin's unique responsibility is Pydantic → DataFrame conversion and the
snapshot pre-processing logic (field selection and snapshot_date injection).
DuckDBWriter calls _serialize_objects internally, so _records_to_df only needs
to call model_dump(mode='json') — no double-serialization.
Concrete classes must set self._con.
"""

from collections.abc import Sequence
from datetime import date
from typing import Any

import duckdb
import pandas as pd
from pydantic import BaseModel

from src.data.cards.storage.base import DuckDBWriter
from src.logger import get_logger

logger = get_logger(__name__)


def _records_to_df(records: Sequence[BaseModel]) -> pd.DataFrame:
    """Convert Pydantic records to a DataFrame suitable for DuckDBWriter.

    Calls model_dump(mode='json') so UUID/date fields become plain strings.
    Object serialization (list/dict → JSON VARCHAR) is handled downstream
    by DuckDBWriter via _serialize_objects.

    Args:
        records: Sequence of Pydantic model instances.

    Returns:
        DataFrame with model fields as columns, UUID/date fields as strings.
    """
    return pd.DataFrame([r.model_dump(mode="json") for r in records])


def _filter_prices_to_date(
    platform_prices: dict[str, Any] | None, target_date: str
) -> dict[str, Any] | None:
    """Return a copy of a platform price dict containing only target_date's entries.

    Each retailer's buylist/retail listing is a dict of {finish: {date: price}}.
    Strips every date except target_date so seeded rows match the shape of
    daily snapshot rows from AllPricesToday.json. Retailers or transaction types
    with no price on target_date are omitted.

    Args:
        platform_prices: Nested price dict keyed by retailer → tx_type → finish → date.
            May be None or empty.
        target_date: ISO-8601 date string (e.g. "2026-06-12") used as the filter key.

    Returns:
        Filtered copy of platform_prices with only target_date entries preserved,
        or None if platform_prices is None/empty or no prices exist for that date.
    """
    if not platform_prices:
        return None
    result: dict[str, Any] = {}
    for retailer, retailer_data in platform_prices.items():
        filtered_retailer: dict[str, Any] = {}
        if "currency" in retailer_data:
            filtered_retailer["currency"] = retailer_data["currency"]
        for tx_type in (
            "buylist",
            "retail",
        ):  # type of store transaction (buy chepar, sell for more)
            listing = retailer_data.get(tx_type) or {}
            filtered_listing: dict[str, Any] = {}
            for finish in ("foil", "normal"):  # type of card
                prices = listing.get(finish) or {}
                if target_date in prices:
                    filtered_listing[finish] = {target_date: prices[target_date]}
            if filtered_listing:
                filtered_retailer[tx_type] = filtered_listing
        if any(k in filtered_retailer for k in ("buylist", "retail")):
            result[retailer] = filtered_retailer
    return result or None


class BronzeWritersMixin:
    """Generic DuckDB write primitives used by BronzeStorage.

    Delegates all SQL writes to DuckDBWriter; the mixin's own responsibility
    is converting Pydantic records to DataFrames (via the module-level
    _records_to_df) and building snapshot rows (field selection, snapshot_date
    injection) before delegating.  Object serialization (list/dict → JSON
    VARCHAR) is handled by DuckDBWriter.full_load/upsert/append internally —
    _records_to_df must not call _serialize_objects to avoid double-serialization.

    Concrete classes must provide:
    - self._con: duckdb.DuckDBPyConnection (set in __init__)
    """

    _con: duckdb.DuckDBPyConnection

    def _full_load_table(self, records: list[BaseModel], table_name: str) -> None:
        """Drop and recreate table_name from records (full replace).

        Converts records to a DataFrame via _records_to_df, then delegates the
        write to DuckDBWriter.full_load, which handles object serialization.

        Args:
            records: List of Pydantic model instances to write. If empty, the operation
                is skipped and a warning is logged.
            table_name: Name of the DuckDB table to create or replace.

        Returns:
            None

        Raises:
            StorageWriteError: If the DuckDB write fails.
        """
        if not records:
            logger.warning("No records to save to %r — skipping", table_name)
            return
        df = _records_to_df(records)
        DuckDBWriter(self._con).full_load(df, table_name)

    def _incremental_load(
        self, records: list[BaseModel], table_name: str, key_column: str
    ) -> None:
        """Delete-then-insert upsert keyed on key_column; creates the table on first call.

        Leaves rows not in the incoming batch untouched. Using DELETE + INSERT
        (rather than UPDATE) surfaces schema changes — new/removed columns appear
        automatically without a migration step.

        Converts records to a DataFrame via _records_to_df, then delegates the
        write to DuckDBWriter.upsert, which handles object serialization.

        Args:
            records: List of Pydantic model instances to upsert. If empty, the operation
                is skipped and a warning is logged.
            table_name: Name of the DuckDB table to upsert into.
            key_column: Column name used to identify rows for deletion before re-insertion.

        Returns:
            None

        Raises:
            StorageWriteError: If the DuckDB write fails.
        """
        if not records:
            logger.warning("No records to load into %r — skipping", table_name)
            return
        df = _records_to_df(records)
        DuckDBWriter(self._con).upsert(df, table_name, key_column)

    def _snapshot(
        self,
        records: list[BaseModel],
        key_column: str,
        history_table: str,
        fields: list[str] | None = None,
    ) -> None:
        """Append a daily snapshot of selected fields to history_table.

        Each row contains key_column, today's date, and the requested fields
        (or the full record when fields is None). Already-snapshotted
        (key, date) pairs are skipped, so multiple calls per day are safe.
        The history table is created automatically on first call.

        Pre-processing (field selection and snapshot_date injection) is performed
        here; the actual DuckDB write is delegated to DuckDBWriter.append.

        Args:
            records: List of Pydantic model instances to snapshot. If empty, the
                operation is skipped and a warning is logged.
            key_column: Column name used as the primary key in snapshot rows and for
                deduplication checks.
            history_table: Name of the DuckDB table where snapshots are appended.
            fields: Optional list of field names to include in each snapshot row.
                When None, all fields from the record are included.

        Returns:
            None

        Raises:
            StorageWriteError: If the DuckDB write fails.
        """
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
        DuckDBWriter(self._con).append(df, history_table, key_column)
        logger.info(
            "Snapshotted %d rows into %r for %s", len(rows), history_table, today_iso
        )
