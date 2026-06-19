"""Pydantic model for MTGGoldfish format staples records."""

from pydantic import BaseModel


class FormatStaple(BaseModel):
    """One card's presence in a specific format's staple list.

    Scraped from mtggoldfish.com/format-staples/{format}.
    Primary key is `id` — a composite of card_name and format so that
    the same card appearing in multiple formats produces distinct rows.
    """

    id: str
    card_name: str
    format: str
    deck_pct: float
    percentage_in_decks: int
    played: float
    top: int
