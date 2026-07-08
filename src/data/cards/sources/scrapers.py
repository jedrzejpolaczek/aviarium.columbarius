"""Async per-source ingestion functions for JSON downloads, format staples, and tournament results."""

import asyncio
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any, TypeVar

import httpx

from src.data.cards.sources.errors import (
    SourceDownloadError,
    SourceError,
    SourceNotRegisteredError,
)
from src.data.cards.sources.extractors import (
    extract_format_staples,
    extract_mtgtop8_decklist,
    extract_mtgtop8_event_decks,
    extract_mtgtop8_tournament_list,
)
from src.data.cards.sources.http import download_html_page, download_json_from_url
from src.data.cards.sources.registry import (
    SOURCE_REGISTRY,
    _save_to_json,
    load_from_json,
)
from src.logger import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


def _cleanup_html_files(paths: list[str]) -> None:
    """Delete temporary HTML files downloaded during a scrape, ignoring failures.

    missing_ok=True handles the file already being gone; the PermissionError
    guard handles Windows still holding the file open in a concurrent
    process — both ingestion paths need this, so it lives here once instead
    of being copied per scraper.
    """
    for p in paths:
        try:
            Path(p).unlink(missing_ok=True)
        except PermissionError:
            pass  # Windows: file still held by a concurrent process; ignore


async def _fetch_and_parse(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    url: str,
    html_path: str,
    html_paths: list[str],
    parse_fn: Callable[[str], T],
    error_context: str,
) -> T | None:
    """Download one HTML page under the shared semaphore, parse it, log+skip on SourceError.

    Registers html_path in html_paths (for later cleanup) before downloading.
    On success, returns parse_fn applied to the downloaded page's text. On
    SourceError (download failure), logs the error with error_context prefixed
    and returns None — callers decide the appropriate empty-result fallback.
    """
    html_paths.append(html_path)
    try:
        async with sem:
            await download_html_page(client, url, html_path)
        html_text = Path(html_path).read_text(encoding="utf-8")
        return parse_fn(html_text)
    except SourceError as e:
        logger.error("%s failed: %s — skipping", error_context, e, exc_info=True)
        return None


async def _ingest_json_sources_async(
    client: httpx.AsyncClient,
    config: dict[str, Any],
) -> dict[str, tuple[list[Any], list[dict[str, Any]]]]:
    """Download and load all JSON sources listed in config["sources"], concurrently.

    Each source is fetched independently; failures are logged and omitted from
    the returned dict without interrupting the other sources.
    """

    async def _fetch_source(
        source: dict[str, Any],
    ) -> tuple[str, tuple[list[Any], list[dict[str, Any]]]] | None:
        source_type: str = source["type"]
        url: str = source["url"]
        path: str = source["path"]
        download_flag: bool = source["flag"]
        try:
            if source_type not in SOURCE_REGISTRY:
                raise SourceNotRegisteredError(f"Unknown source type: {source_type!r}")
            model, extractor = SOURCE_REGISTRY[source_type]

            if download_flag:
                if source_type == "scryfall":
                    logger.progress("Scryfall: resolving download URI from bulk meta")
                    try:
                        r = await client.get(url)
                        r.raise_for_status()
                        url = r.json()["download_uri"]
                    except (httpx.HTTPStatusError, KeyError) as e:
                        raise SourceDownloadError(
                            f"Failed to resolve Scryfall bulk meta from {url}: {e}"
                        ) from e
                await download_json_from_url(client, url, path)

            return source_type, load_from_json(path, model, extractor)

        except Exception as e:
            logger.error(
                "Source %r failed: %s — skipping", source_type, e, exc_info=True
            )
            return None

    outcomes = await asyncio.gather(
        *[_fetch_source(source) for source in config.get("sources", [])]
    )
    return {k: v for result in outcomes if result is not None for k, v in [result]}


async def _ingest_format_staples_async(
    client: httpx.AsyncClient,
    config: dict[str, Any],
) -> dict[str, tuple[list[Any], list[dict[str, Any]]]]:
    """Scrape MTGGoldfish format-staples pages concurrently and load into FormatStaple records.

    Config key: config["format_staples"] with 'formats' (list), 'base_url'
    (template with {format} placeholder), and 'path' (destination JSON file).
    Returns an empty dict if the key is absent or base_url is empty.

    All format pages are fetched in parallel, throttled to 3 concurrent requests
    via a semaphore. Records are combined into a single JSON file and the HTML
    files are deleted. The JSON is then loaded and validated via load_from_json.

    mtggoldfish.com — scraping rights reviewed 2026-05-22 (see ADR-015):
    robots.txt: Allow: /, scraped path /format-staples/* not in Disallow list.
    ai-train=no applies to LLM training, not to this local price-prediction pipeline.
    """
    staples_cfg = config.get("format_staples", {})
    base_url = staples_cfg.get("base_url", "")
    if not staples_cfg or not base_url:
        return {}

    model, _ = SOURCE_REGISTRY["format_staples"]
    json_path: str = staples_cfg.get("path", "data/raw/format_staples.json")
    today = date.today().isoformat()
    formats: list[str] = staples_cfg.get("formats", [])
    sem = asyncio.Semaphore(3)  # polite: max 3 concurrent requests to mtggoldfish.com
    html_paths: list[str] = []

    async def _fetch_format(fmt: str) -> list[dict[str, Any]]:
        html_path = f"data/raw/format_staples_{fmt}.html"
        logger.progress("format_staples: scraping %r", fmt)
        raw = await _fetch_and_parse(
            client,
            sem,
            base_url.format(format=fmt),
            html_path,
            html_paths,
            lambda html_text: extract_format_staples(html_text, fmt=fmt),
            f"format_staples {fmt!r}",
        )
        if raw is None:
            return []
        recs: list[dict[str, Any]] = []
        for rec in raw:
            rec["snapshot_date"] = today
            recs.append(rec)
        return recs

    results = await asyncio.gather(*[_fetch_format(fmt) for fmt in formats])
    all_raw = [rec for fmt_recs in results for rec in fmt_recs]

    _save_to_json(all_raw, json_path)
    _cleanup_html_files(html_paths)

    return {"format_staples": load_from_json(json_path, model)}


async def _ingest_tournament_results_async(
    client: httpx.AsyncClient,
    config: dict[str, Any],
) -> dict[str, tuple[list[Any], list[dict[str, Any]]]]:
    """Run the 3-level mtgtop8.com scrape concurrently and load into TournamentResult records.

    Config key: config["tournament_results"] with 'formats' (list), 'format_codes'
    (dict mapping format name to mtgtop8 code), 'list_url' (template with {code}
    placeholder), 'deck_url_prefix', 'max_tournaments_per_format', and 'path'
    (destination JSON file). Returns an empty dict if the key is absent.

    Concurrency model — three levels, each fully parallelised within the level:
        Level 1 (format list pages) — gathered across all formats.
        Level 2 (event pages)       — gathered across all tournaments from all formats.
        Level 3 (deck pages)        — gathered across all decks from all events.
    All requests are throttled to 3 concurrent connections via a semaphore.

    mtgtop8.com — scraping rights reviewed 2026-05-22 (see ADR-015):
    robots.txt returns 404 (no restrictions defined); all paths allowed by default.
    """
    tournament_cfg = config.get("tournament_results", {})
    if not tournament_cfg:
        return {}

    t_model, _ = SOURCE_REGISTRY["tournament_results"]
    t_json_path: str = tournament_cfg.get("path", "data/raw/tournament_results.json")
    format_codes: dict[str, str] = tournament_cfg.get("format_codes", {})
    list_url_tpl: str = tournament_cfg.get("list_url", "")
    deck_url_prefix: str = tournament_cfg.get(
        "deck_url_prefix", "https://www.mtgtop8.com"
    )
    max_n: int = tournament_cfg.get("max_tournaments_per_format", 10)
    formats: list[str] = tournament_cfg.get("formats", [])

    t_html_paths: list[str] = []
    sem = asyncio.Semaphore(3)  # polite: max 3 concurrent requests to mtgtop8.com

    # ── Level 1: all format list pages ────────────────────────────────────────
    async def _fetch_format_list(
        fmt: str,
    ) -> tuple[str, str, list[dict[str, Any]]] | None:
        code = format_codes.get(fmt, "")
        if not code:
            logger.warning("No mtgtop8 code for format %r — skipping", fmt)
            return None
        list_path = f"data/raw/tournament_list_{fmt}.html"
        raw = await _fetch_and_parse(
            client,
            sem,
            list_url_tpl.format(code=code),
            list_path,
            t_html_paths,
            lambda list_html: extract_mtgtop8_tournament_list(list_html, fmt),
            f"Tournament list for {fmt!r}",
        )
        if raw is None:
            return None
        seen_ids: set[str] = set()
        unique: list[dict[str, Any]] = []
        for t in raw:
            if t["_event_id"] not in seen_ids:
                seen_ids.add(t["_event_id"])
                unique.append(t)
        tournaments = unique[:max_n]
        logger.info(
            "Found %d tournaments for format %r on %s",
            len(tournaments),
            fmt,
            list_url_tpl.format(code=code),
        )
        return fmt, code, tournaments

    level1 = await asyncio.gather(*[_fetch_format_list(fmt) for fmt in formats])
    valid_formats = [r for r in level1 if r is not None]

    # ── Level 2: all event pages (across all formats) ─────────────────────────
    async def _fetch_event(
        fmt: str, code: str, t: dict[str, Any]
    ) -> tuple[str, str, dict[str, Any], list[dict[str, Any]]] | None:
        event_id = t["_event_id"]
        logger.progress("[%s] fetching event — %s", fmt, t["event_name"])
        event_path = f"data/raw/tournament_event_{event_id}.html"
        decks = await _fetch_and_parse(
            client,
            sem,
            f"{deck_url_prefix}/event?e={event_id}&f={code}",
            event_path,
            t_html_paths,
            extract_mtgtop8_event_decks,
            f"Event {event_id!r}",
        )
        if decks is None:
            return None
        if len(decks) < 4:
            logger.warning(
                "[%s] event %s yielded only %d deck(s) — possible parse failure",
                fmt,
                event_id,
                len(decks),
            )
        return fmt, code, t, decks

    all_event_tasks = [
        _fetch_event(fmt, code, t)
        for fmt, code, tournaments in valid_formats
        for t in tournaments
    ]
    level2 = await asyncio.gather(*all_event_tasks)
    valid_events = [r for r in level2 if r is not None]

    # ── Level 3: all deck pages (across all events) ───────────────────────────
    async def _fetch_deck(
        fmt: str, code: str, t: dict[str, Any], deck_meta: dict[str, Any]
    ) -> list[dict[str, Any]]:
        deck_id = deck_meta["deck_id"]
        logger.progress(
            "[%s] event %s — deck: %s", fmt, t["_event_id"], deck_meta["deck_name"]
        )
        deck_path = f"data/raw/tournament_deck_{deck_id}.html"
        result = await _fetch_and_parse(
            client,
            sem,
            f"{deck_url_prefix}/event?e={t['_event_id']}&d={deck_id}&f={code}",
            deck_path,
            t_html_paths,
            lambda deck_html: extract_mtgtop8_decklist(
                deck_html,
                t["tournament_id"],
                t["tournament_date"],
                t["event_name"],
                fmt,
                deck_meta["placement"],
                deck_meta["player"],
                deck_meta["deck_name"],
            ),
            f"Deck {deck_id!r}",
        )
        return result if result is not None else []

    all_deck_tasks = [
        _fetch_deck(fmt, code, t, deck_meta)
        for fmt, code, t, deck_metas in valid_events
        for deck_meta in deck_metas
    ]
    level3 = await asyncio.gather(*all_deck_tasks)
    new_raw = [rec for deck_recs in level3 for rec in deck_recs]

    _save_to_json(new_raw, t_json_path)
    _cleanup_html_files(t_html_paths)

    result = load_from_json(t_json_path, t_model)
    logger.info(
        "Tournament results: %d records, %d errors", len(result[0]), len(result[1])
    )
    return {"tournament_results": result}
