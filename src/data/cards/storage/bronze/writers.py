"""Utility functions for the Bronze tier.

Provides two pure helpers used by BronzeStorage:
    _records_to_df          — Pydantic → DataFrame conversion
    _filter_prices_to_date  — per-date price dict filtering for seed/snapshot
"""

from collections.abc import Sequence
from typing import Any

import pandas as pd
from pydantic import BaseModel


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

