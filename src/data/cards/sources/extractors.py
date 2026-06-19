"""HTML and JSON extractor functions for the card data sources pipeline.

Each function parses a raw source (JSON dict or HTML string) into a flat
list of record dicts ready for Pydantic validation.

MTGJson extractors:
    extract_mtgjson_cards   — flattens AllPrintings nested structure
    extract_mtgjson_prices  — flattens AllPricesToday UUID-keyed structure

MTGGoldfish extractors:
    extract_format_staples  — parses the format-staples HTML table

MTGTop8 extractors (3-level scrape):
    extract_mtgtop8_tournament_list — level 1: format page → tournament list
    extract_mtgtop8_event_decks     — level 2: event page → deck metadata
    extract_mtgtop8_decklist        — level 3: deck page → per-card rows
"""

from src.logger import get_logger
import re
from datetime import datetime
from typing import Any

from bs4 import BeautifulSoup

logger = get_logger(__name__)


def extract_mtgjson_cards(raw: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten MTGjson AllPrintings nested structure (data.<SET>.cards[]) into a flat list.

    Args:
        raw (dict): Parsed AllPrintings JSON with a 'data' key mapping set codes to set objects.

    Returns (list):
        A flat list of card dicts from all sets.
    """
    cards = []
    for set_data in raw["data"].values():
        cards.extend(set_data.get("cards", []))
    return cards


def extract_mtgjson_prices(raw: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten MTGJson AllPricesToday nested structure (data.<UUID>) into a flat list.

    The source JSON is a dict keyed by card UUID. This extractor injects the UUID
    into each price record so MtgjsonCardPrices can be validated with a uuid field.

    Args:
        raw (dict): Parsed AllPricesToday JSON with a 'data' key mapping card UUIDs to price objects.

    Returns (list):
        A flat list of price dicts, each containing a 'uuid' key.
    """
    return [
        {"uuid": uuid_str, **price_data} for uuid_str, price_data in raw["data"].items()
    ]


def extract_format_staples(html: str, fmt: str) -> list[dict[str, Any]]:
    """Parse MTGGoldfish format-staples HTML into FormatStaple records.

    Table column order (mtggoldfish.com/format-staples/{format}):
        col[0] rank        → top                (int)
        col[1] card name   → card_name          (str, from <a> tag)
        col[2] mana cost   → skipped
        col[3] % of decks  → deck_pct           (float), percentage_in_decks (int)
        col[4] # played    → played             (float)
    """
    soup = BeautifulSoup(html, "lxml")
    table = soup.select_one("table.table-staples")
    if table is None:
        logger.warning(
            "No table.table-staples found for format %r — site layout may have changed",
            fmt,
        )
        return []

    records = []
    for row in table.select("tr"):
        cols = row.select("td")
        if len(cols) < 5:
            continue

        try:
            top = int(cols[0].get_text(strip=True))
            link = cols[1].select_one("a")
            name = link.get_text(strip=True) if link else cols[1].get_text(strip=True)
            deck_pct = float(cols[3].get_text(strip=True).replace("%", ""))
            played = float(cols[4].get_text(strip=True).replace(",", ""))
        except ValueError:
            continue

        records.append(
            {
                "id": f"{name}__{fmt}",
                "card_name": name,
                "format": fmt,
                "deck_pct": deck_pct,
                "percentage_in_decks": int(deck_pct),
                "played": played,
                "top": top,
            }
        )

    return records


def _parse_mtgtop8_date(date_str: str) -> str:
    """Parse a mtgtop8.com date string to ISO format YYYY-MM-DD.

    Tries each known date format in order until one succeeds. mtgtop8.com is
    inconsistent — different pages use DD/MM/YY, MM/DD/YY, or abbreviated month
    variants, so multiple formats must be attempted.

    Args:
        date_str: Raw date string scraped from the page (e.g. "24/05/26", "May 24 26").

    Returns:
        ISO-formatted date string "YYYY-MM-DD".

    Raises:
        ValueError: If none of the known formats match date_str.
    """
    for fmt in ("%d/%m/%y", "%m/%d/%y", "%b %d %y", "%d %b %y"):
        try:
            return datetime.strptime(date_str.strip(), fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    raise ValueError(f"Cannot parse mtgtop8 date: {date_str!r}")


def extract_mtgtop8_tournament_list(html: str, fmt: str) -> list[dict[str, Any]]:
    """Parse a mtgtop8.com format page into tournament metadata records.

    Scrapes /format?f={code} — the table of recent top-8 events for one format.
    Each record contains enough information to construct the event and deck URLs
    needed for the second and third scraping levels.

    The page renders each tournament as a <tr class="hover_tr"> inside
    <table class="Stable">. Each row has four <td> columns:
        [0] thumbnail icon (empty)
        [1] event name with <a href="event?e={id}&f={code}"> link
        [2] location / venue (empty for MTGO events)
        [3] date string (DD/MM/YY)

    NOTE: CSS selectors are based on the observed mtgtop8.com HTML structure and
    may need adjustment if the site changes its layout.

    Args:
        html: Raw HTML of the format list page.
        fmt:  Canonical format name (e.g. "modern").

    Returns:
        List of dicts with keys: tournament_id, tournament_date, event_name,
        format, _event_id (raw numeric ID used for URL construction).
    """
    soup = BeautifulSoup(html, "lxml")
    records = []

    for row in soup.select("table.Stable tr.hover_tr"):
        cols = row.select("td")
        if not cols:
            continue

        link = cols[1].select_one("a[href*='event?e=']")
        if not link:
            continue

        href = str(link.get("href") or "")
        match = re.search(r"e=(\d+)", href)
        if not match:
            continue

        event_id = match.group(1)
        event_name = link.get_text(strip=True)
        date_text = cols[-1].get_text(strip=True)

        try:
            tournament_date = _parse_mtgtop8_date(date_text)
        except ValueError:
            logger.warning(
                "Skipping event %r — unparseable date %r", event_id, date_text
            )
            continue

        records.append(
            {
                "tournament_id": f"mtgtop8_{event_id}",
                "tournament_date": tournament_date,
                "event_name": event_name,
                "format": fmt,
                "_event_id": event_id,
            }
        )

    return records


def extract_mtgtop8_event_decks(html: str) -> list[dict[str, Any]]:
    """Parse a mtgtop8.com event page into deck metadata records.

    Scrapes /event?e={id}&f={code} — the top-8 results page for one tournament.
    Returns player name, deck archetype, placement, and the deck ID needed to
    fetch the full decklist in the third scraping level.

    NOTE: CSS selectors are based on the expected mtgtop8.com HTML structure and
    may need adjustment if the site changes its layout.

    Args:
        html: Raw HTML of the event results page.

    Returns:
        Up to 8 dicts with keys: deck_id, player, deck_name, placement.
    """
    soup = BeautifulSoup(html, "lxml")
    decks: list[dict[str, Any]] = []

    for row in soup.select("div.chosen_tr, div.hover_tr"):
        link = row.select_one("a[href*='&d=']")
        if not link:
            continue

        href = str(link.get("href") or "")
        match = re.search(r"d=(\d+)", href)
        if not match:
            continue

        player_el = row.select_one(".G11")
        deck_name_el = row.select_one(".S14") or link

        decks.append(
            {
                "deck_id": match.group(1),
                "player": player_el.get_text(strip=True) if player_el else "",
                "deck_name": deck_name_el.get_text(strip=True),
                "placement": len(decks) + 1,
            }
        )

        if len(decks) == 8:
            break

    return decks


def extract_mtgtop8_decklist(
    html: str,
    tournament_id: str,
    tournament_date: str,
    event_name: str,
    fmt: str,
    placement: int,
    player: str,
    deck_name: str,
) -> list[dict[str, Any]]:
    """Parse a mtgtop8.com deck page into per-card rows.

    Scrapes /event?e={id}&d={deck_id}&f={code} — the full decklist for one
    top-8 finisher. Each card in the main deck and sideboard becomes one row.

    Every card row is rendered as a <div class="deck_line hover_tr" id="md{ref}">
    for main-deck entries and id="sb{ref}"> for sideboard entries. The id prefix
    is the canonical sideboard marker — there is no separate sideboard CSS class.

    NOTE: CSS selectors are based on the observed mtgtop8.com HTML structure and
    may need adjustment if the site changes its layout.

    Args:
        html:             Raw HTML of the deck page.
        tournament_id:    Synthetic tournament ID (e.g. "mtgtop8_12345").
        tournament_date:  ISO date string "YYYY-MM-DD".
        event_name:       Name of the tournament event.
        fmt:              Canonical format name (e.g. "modern").
        placement:        Finishing position 1–8.
        player:           Player name.
        deck_name:        Deck archetype name.

    Returns:
        List of card-row dicts, one per (card_name, is_sideboard) occurrence.
    """
    soup = BeautifulSoup(html, "lxml")
    records: list[dict[str, Any]] = []

    for el in soup.select("div.deck_line"):
        text = el.get_text().strip()
        if not text:
            continue

        is_sideboard = str(el.get("id") or "").startswith("sb")

        match = re.match(r"^(\d+)\s+(.+)$", text)
        if not match:
            continue

        copies = int(match.group(1))
        card_name = match.group(2).strip()

        records.append(
            {
                "id": f"{tournament_id}__{card_name}__{is_sideboard}",
                "tournament_id": tournament_id,
                "tournament_date": tournament_date,
                "format": fmt,
                "event_name": event_name,
                "placement": placement,
                "player": player,
                "deck_name": deck_name,
                "card_name": card_name,
                "copies": copies,
                "is_sideboard": is_sideboard,
            }
        )

    return records
