"""Async ingestion entry point — downloads and validates all configured card data sources."""

import asyncio
import time
from typing import Any

import httpx

from src.data.cards.sources.http import log_response
from src.data.cards.sources.scrapers import (
    _ingest_format_staples_async,
    _ingest_json_sources_async,
    _ingest_tournament_results_async,
)
from src.logger import get_logger

logger = get_logger(__name__)


async def ingesting_pipeline(
    config: dict[str, Any],
) -> dict[str, tuple[list[Any], list[dict[str, Any]]]]:
    """Download and load all configured sources into Pydantic models.

    Three source categories (JSON sources, format staples, tournament results)
    run concurrently via asyncio.gather; within each scraping category requests
    are throttled by a semaphore of 3.

    Args:
        config: Full bronze_config dict as loaded from bronze_config.json.

    Returns:
        A dict mapping source type -> (records, errors), where records is a list
        of validated Pydantic model instances and errors is a list of dicts with
        'name' and 'error' keys for each record that failed validation.
        Sources that fail entirely are logged and omitted from the returned dict.
    """
    logger.info(
        "ingesting_pipeline started — %d JSON sources, format_staples=%s, tournament_results=%s",
        len(config.get("sources", [])),
        bool(config.get("format_staples")),
        bool(config.get("tournament_results")),
    )
    t0 = time.monotonic()
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(60.0),
        follow_redirects=True,
        limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
        event_hooks={"response": [log_response]},
    ) as client:
        results_json, results_staples, results_tournaments = await asyncio.gather(
            _ingest_json_sources_async(client, config),
            _ingest_format_staples_async(client, config),
            _ingest_tournament_results_async(client, config),
        )
    merged = {**results_json, **results_staples, **results_tournaments}
    summary = ", ".join(f"{k}={len(v[0])}" for k, v in merged.items())
    logger.info(
        "ingesting_pipeline complete in %.1fs — %s", time.monotonic() - t0, summary
    )
    return merged
