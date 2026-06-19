"""HTTP download helpers with exponential-backoff retry.

Provides download_json_from_url and download_html_page — the two async download
primitives used by ingesting_pipeline. Both retry automatically on transient
HTTP errors (429, 500, 502, 503, 504) using exponential backoff; permanent
errors (401, 403, 404) are raised immediately without retrying.

Retry policy:
    Up to 5 attempts, wait 1 → 2 → 4 → 8 → 16 s (capped at 30 s).
    A WARNING is logged before each retry.
"""

import json
import logging
from pathlib import Path
from typing import Any

import httpx
from tenacity import (
    AsyncRetrying,
    before_sleep_log,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from src.data.cards.sources.errors import SourceDownloadError
from src.logger import get_logger

logger = get_logger(__name__)

# HTTP status codes that are transient and worth retrying.
# 404/401/403 are permanent — retrying them wastes time and won't help.
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def _is_retryable_http_error(exc: BaseException) -> bool:
    return (
        isinstance(exc, httpx.HTTPStatusError)
        and exc.response.status_code in _RETRYABLE_STATUS_CODES
    )


def _make_retry() -> AsyncRetrying:
    """Create a fresh AsyncRetrying instance with the shared retry policy."""
    return AsyncRetrying(
        retry=retry_if_exception(_is_retryable_http_error),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )


async def _fetch_json(client: httpx.AsyncClient, url: str) -> Any:
    """Fetch and parse JSON from *url*, retrying on transient HTTP errors.

    Kept separate from download_json_from_url so that tenacity can intercept
    HTTPStatusError before it is wrapped in SourceDownloadError.
    """
    result: Any = None
    async for attempt in _make_retry():
        with attempt:
            r = await client.get(url)
            r.raise_for_status()
            result = r.json()
    return result


async def _fetch_html(client: httpx.AsyncClient, url: str) -> str:
    """Fetch HTML text from *url*, retrying on transient HTTP errors.

    Kept separate from download_html_page so that tenacity can intercept
    HTTPStatusError before it is wrapped in SourceDownloadError.
    """
    result: str = ""
    async for attempt in _make_retry():
        with attempt:
            r = await client.get(url, headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            result = r.text
    return result


async def log_response(response: httpx.Response) -> None:
    """httpx response event hook — logs each HTTP response at PROGRESS level."""
    logger.progress(
        "HTTP %s %s — %d",
        response.request.method,
        response.request.url,
        response.status_code,
    )


async def download_json_from_url(
    client: httpx.AsyncClient, url: str, output_path: str
) -> None:
    """Download a JSON file from *url* and save it to *output_path*.

    Retries up to 5 times with exponential backoff (1s → 2s → 4s → 8s → 16s)
    on transient HTTP errors (429, 500, 502, 503, 504). Permanent errors
    (404, 401, 403) are raised immediately without retrying.

    Args:
        client:      Shared httpx.AsyncClient for the pipeline run.
        url:         The URL to download from.
        output_path: Local file path where the JSON will be saved.
    """
    logger.progress("Downloading %s → %s", url, output_path)
    try:
        data = await _fetch_json(client, url)
    except httpx.HTTPStatusError as e:
        raise SourceDownloadError(f"HTTP error downloading {url}: {e}") from e
    with open(Path(output_path), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.progress("Saved JSON to %s", output_path)


async def download_html_page(
    client: httpx.AsyncClient, url: str, output_path: str
) -> None:
    """Download an HTML page from *url* and save it to *output_path*.

    Retries up to 5 times with exponential backoff (1s → 2s → 4s → 8s → 16s)
    on transient HTTP errors (429, 500, 502, 503, 504). Permanent errors
    (404, 401, 403) are raised immediately without retrying.

    Args:
        client:      Shared httpx.AsyncClient for the pipeline run.
        url:         The URL to download from.
        output_path: Local file path where the HTML will be saved.
    """
    logger.progress("Downloading HTML %s → %s", url, output_path)
    try:
        html = await _fetch_html(client, url)
    except httpx.HTTPStatusError as e:
        raise SourceDownloadError(f"HTTP error downloading {url}: {e}") from e
    Path(output_path).write_text(html, encoding="utf-8")
