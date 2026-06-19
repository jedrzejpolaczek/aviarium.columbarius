"""Pydantic model for mtgtop8.com tournament result records."""

from pydantic import BaseModel


class TournamentResult(BaseModel):
    """One card's appearance in a top-8 tournament decklist.

    Scraped from mtgtop8.com — tournament list page then individual deck pages.
    Primary key is `id` — a composite of tournament_id, card_name, and
    is_sideboard so the same card in the same tournament produces two distinct
    rows when it appears in both main deck and sideboard.
    """

    id: str  # "{tournament_id}__{card_name}__{is_sideboard}"
    tournament_id: str  # "mtgtop8_{event_id}"
    tournament_date: str  # "YYYY-MM-DD"
    format: str  # "modern" | "legacy" | "vintage" | "standard" | "pioneer"
    event_name: str
    placement: int  # 1–8
    player: str
    deck_name: str  # archetype name
    card_name: str  # face name as listed in decklist
    copies: int
    is_sideboard: bool
