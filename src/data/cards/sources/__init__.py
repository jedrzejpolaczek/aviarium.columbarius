"""Card data sources package.

Re-exports the full public API so existing imports remain unchanged:

    from src.data.cards.sources import ingesting_pipeline  # async coroutine
    from src.data.cards.sources import SourceDownloadError

Callers must drive the event loop themselves:

    results = asyncio.run(ingesting_pipeline(config))

Internal structure:
    errors.py     — exception hierarchy (SourceError and subclasses)
    http.py       — async HTTP retry machinery and download helpers
                    (download_json_from_url, download_html_page)
    extractors.py — HTML and JSON parser functions
    registry.py   — SOURCE_REGISTRY, load_from_json, _save_to_json
    scrapers.py   — async per-source ingesters (_ingest_json_sources_async,
                    _ingest_format_staples_async, _ingest_tournament_results_async)
    pipeline.py   — orchestrator (ingesting_pipeline)
"""

from src.data.cards.sources.errors import (
    SourceDownloadError,
    SourceError,
    SourceLoadError,
    SourceNotRegisteredError,
)
from src.data.cards.sources.extractors import (
    extract_format_staples,
    extract_mtgjson_cards,
    extract_mtgjson_prices,
    extract_mtgtop8_decklist,
    extract_mtgtop8_event_decks,
    extract_mtgtop8_tournament_list,
)
from src.data.cards.sources.http import (
    _is_retryable_http_error,
    download_html_page,
    download_json_from_url,
)
from src.data.cards.sources.pipeline import ingesting_pipeline
from src.data.cards.sources.registry import (
    SOURCE_REGISTRY,
    _save_to_json,
    load_from_json,
)
from src.data.cards.sources.scrapers import (
    _ingest_format_staples_async,
    _ingest_tournament_results_async,
)

__all__ = [
    "SourceError",
    "SourceNotRegisteredError",
    "SourceDownloadError",
    "SourceLoadError",
    "extract_mtgjson_cards",
    "extract_mtgjson_prices",
    "extract_format_staples",
    "extract_mtgtop8_tournament_list",
    "extract_mtgtop8_event_decks",
    "extract_mtgtop8_decklist",
    "_is_retryable_http_error",
    "download_json_from_url",
    "download_html_page",
    "SOURCE_REGISTRY",
    "_ingest_format_staples_async",
    "_ingest_tournament_results_async",
    "_save_to_json",
    "load_from_json",
    "ingesting_pipeline",
]
