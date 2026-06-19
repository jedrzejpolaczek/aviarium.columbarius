"""Pydantic models for Scryfall bulk-data card objects.

Covers the Scryfall all-cards bulk export (ScryfallCard). Nested structures
such as prices, legalities, image_uris, and card_faces are modelled as
sub-models so they can be stored as DuckDB STRUCT columns and queried with
dot notation (e.g. prices.usd, legalities.standard).

All fields are Optional unless Scryfall guarantees their presence. Fields not
needed downstream are omitted; unknown fields are silently dropped.
"""

from datetime import date
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class ScryfallPrices(BaseModel):
    """
    Daily prices for a Scryfall card across currencies and finishes. All values are nullable — a missing price means no market data is available.

    All data information are from Scryfall API docummentation: https://scryfall.com/docs/api/cards#card-face-objects
    """

    usd: Optional[float] = Field(None, description="Current nonfoil price in USD.")
    usd_foil: Optional[float] = Field(None, description="Current foil price in USD.")
    usd_etched: Optional[float] = Field(
        None, description="Current etched foil price in USD."
    )
    eur: Optional[float] = Field(None, description="Current nonfoil price in EUR.")
    eur_foil: Optional[float] = Field(None, description="Current foil price in EUR.")
    eur_etched: Optional[float] = Field(
        None, description="Current etched foil price in EUR."
    )
    tix: Optional[float] = Field(None, description="Current price in MTGO tickets.")

    @field_validator("*", mode="before")
    @classmethod
    def parse_price(cls, v: object) -> float | None:
        return float(v) if v is not None else None  # type: ignore[arg-type]


class ScryfallPreview(BaseModel):
    """
    Spoiler preview metadata for a card that was previewed before its set's official release.

    All data information are from Scryfall API docummentation: https://scryfall.com/docs/api/cards#card-face-objects
    """

    previewed_at: Optional[date] = Field(
        None, description="The date this card was previewed."
    )
    source_uri: Optional[str] = Field(
        None, description="A link to the preview for this card."
    )
    source: Optional[str] = Field(
        None, description="The name of the source that previewed this card."
    )


class ScryfallRelatedCard(BaseModel):
    """
    A card closely related to another card — tokens it creates, meld parts, or combo pieces. Appears in ScryfallCard.all_parts.

    All data information are from Scryfall API docummentation: https://scryfall.com/docs/api/cards#card-face-objects
    """

    id: UUID = Field(description="A unique ID for this card in Scryfall's database.")
    object: str = Field(
        description="A content type for this object, always related_card."
    )
    component: str = Field(
        description="A field explaining what role this card plays in this relationship, one of token, meld_part, meld_result, or combo_piece."
    )
    name: str = Field(description="The name of this particular related card.")
    type_line: str = Field(description="The type line of this card.")
    uri: str = Field(
        description="A URI where you can retrieve a full object describing this card on Scryfall's API."
    )


class ScryfallCardFace(BaseModel):
    """
    One face of a multifaced card (transform, modal DFC, split, flip, etc.). Present in ScryfallCard.card_faces when layout is not normal.

    All data information are from Scryfall API docummentation: https://scryfall.com/docs/api/cards#card-face-objects
    """

    artist: Optional[str] = Field(
        None,
        description="The name of the illustrator of this card face. Newly spoiled cards may not have this field yet.",
    )
    artist_id: Optional[UUID] = Field(
        None,
        description="The ID of the illustrator of this card face. Newly spoiled cards may not have this field yet.",
    )
    cmc: Optional[float] = Field(
        None,
        description="The mana value of this particular face, if the card is reversible.",
    )
    color_indicator: Optional[list[str]] = Field(
        None, description="The colors in this face's color indicator, if any."
    )
    colors: Optional[list[str]] = Field(
        None,
        description="This face's colors, if the game defines colors for the individual face of this card.",
    )
    defense: Optional[str] = Field(None, description="This face's defense, if any.")
    flavor_text: Optional[str] = Field(
        None, description="The flavor text printed on this face, if any."
    )
    illustration_id: Optional[UUID] = Field(
        None,
        description="A unique identifier for the card face artwork that remains consistent across reprints. Newly spoiled cards may not have this field yet.",
    )
    image_uris: Optional[dict[str, str]] = Field(
        None,
        description="An object providing URIs to imagery for this face, if this is a double-sided card. If this card is not double-sided, image_uris will be part of the parent object instead.",
    )
    layout: Optional[str] = Field(
        None, description="The layout of this card face, if the card is reversible."
    )
    loyalty: Optional[str] = Field(None, description="This face's loyalty, if any.")
    mana_cost: str = Field(
        description="The mana cost for this face. This value will be any empty string if the cost is absent."
    )
    name: str = Field(description="The name of this particular face.")
    object: str = Field(description="A content type for this object, always card_face.")
    oracle_id: Optional[UUID] = Field(
        None,
        description="The Oracle ID of this particular face, if the card is reversible.",
    )
    oracle_text: Optional[str] = Field(
        None, description="The Oracle text for this face, if any."
    )
    power: Optional[str] = Field(
        None,
        description="This face's power, if any. Note that some cards have powers that are not numeric, such as *.",
    )
    printed_name: Optional[str] = Field(
        None, description="The localized name printed on this face, if any."
    )
    printed_text: Optional[str] = Field(
        None, description="The localized text printed on this face, if any."
    )
    printed_type_line: Optional[str] = Field(
        None, description="The localized type line printed on this face, if any."
    )
    toughness: Optional[str] = Field(None, description="This face's toughness, if any.")
    type_line: Optional[str] = Field(
        None,
        description="The type line of this particular face, if the card is reversible.",
    )
    watermark: Optional[str] = Field(
        None, description="The watermark on this particular card face, if any."
    )


class ScryfallCard(BaseModel):
    """
    A single Magic: The Gathering card printing as returned by the Scryfall API. One record per unique print (set + collector number + language), not per oracle card identity.

    All data information are from Scryfall API docummentation: https://scryfall.com/docs/api/cards#card-face-objects
    """

    # --- Core fields ---
    arena_id: Optional[int] = Field(
        None,
        description="This card's Arena ID, if any. A large percentage of cards are not available on Arena and do not have this ID.",
    )
    id: UUID = Field(description="A unique ID for this card in Scryfall's database.")
    lang: str = Field(description="A language code for this printing.")
    mtgo_id: Optional[int] = Field(
        None,
        description="This card's Magic Online ID (also known as the Catalog ID), if any. A large percentage of cards are not available on Magic Online and do not have this ID.",
    )
    mtgo_foil_id: Optional[int] = Field(
        None,
        description="This card's foil Magic Online ID (also known as the Catalog ID), if any. A large percentage of cards are not available on Magic Online and do not have this ID.",
    )
    multiverse_ids: Optional[list[int]] = Field(
        None,
        description="This card's multiverse IDs on Gatherer, if any, as an array of integers. Note that Scryfall includes many promo cards, tokens, and other esoteric objects that do not have these identifiers.",
    )
    resource_id: Optional[str] = Field(
        None, description="This card's Resource ID on Gatherer, if any."
    )
    tcgplayer_id: Optional[int] = Field(
        None,
        description="This card's ID on TCGplayer's API, also known as the productId.",
    )
    tcgplayer_etched_id: Optional[int] = Field(
        None,
        description="This card's ID on TCGplayer's API, for its etched version if that version is a separate product.",
    )
    cardmarket_id: Optional[int] = Field(
        None,
        description="This card's ID on Cardmarket's API, also known as the idProduct.",
    )
    object: str = Field(
        description="A content type for thisoracle_id object, always card."
    )
    layout: str = Field(description="A code for this card's layout.")
    oracle_id: Optional[UUID] = Field(
        None,
        description="A unique ID for this card's oracle identity. Consistent across reprinted card editions and unique among different cards with the same name. Always present except for the reversible_card layout where oracle_id will be found on each face instead.",
    )
    prints_search_uri: str = Field(
        description="A link to where you can begin paginating all re/prints for this card on Scryfall's API."
    )
    rulings_uri: str = Field(
        description="A link to this card's rulings list on Scryfall's API."
    )
    scryfall_uri: str = Field(
        description="A link to this card's permapage on Scryfall's website."
    )
    uri: str = Field(description="A link to this card object on Scryfall's API.")

    # --- Gameplay fields ---
    all_parts: Optional[list[ScryfallRelatedCard]] = Field(
        None,
        description="If this card is closely related to other cards, this property will be an array with Related Card Objects.",
    )
    card_faces: Optional[list[ScryfallCardFace]] = Field(
        None, description="An array of Card Face objects, if this card is multifaced."
    )
    cmc: Optional[float] = Field(
        default=None,
        description="The card's mana value. Absent for reversible_card layout — use card_faces[*].cmc instead.",
    )
    color_identity: list[str] = Field(description="This card's color identity.")
    color_indicator: Optional[list[str]] = Field(
        None,
        description="The colors in this card's color indicator, if any. A null value for this field indicates the card does not have one.",
    )
    colors: Optional[list[str]] = Field(
        None,
        description="This card's colors, if the overall card has colors defined by the rules. Otherwise the colors will be on the card_faces objects.",
    )
    defense: Optional[str] = Field(None, description="This card's defense, if any.")
    edhrec_rank: Optional[int] = Field(
        None,
        description="This card's overall rank/popularity on EDHREC. Not all cards are ranked.",
    )
    game_changer: Optional[bool] = Field(
        None, description="True if this card is on the Commander Game Changer list."
    )
    hand_modifier: Optional[str] = Field(
        None,
        description="This card's hand modifier, if it is a Vanguard card. This value will contain a delta, such as -1.",
    )
    keywords: list[str] = Field(
        description="An array of keywords that this card uses, such as 'Flying' and 'Cumulative upkeep'."
    )
    legalities: dict[str, str] = Field(
        description="An object describing the legality of this card across play formats. Possible legalities are legal, not_legal, restricted, and banned."
    )
    life_modifier: Optional[str] = Field(
        None,
        description="This card's life modifier, if it is a Vanguard card. This value will contain a delta, such as +2.",
    )
    loyalty: Optional[str] = Field(
        None,
        description="This card's loyalty, if any. Note that some cards have loyalties that are not numeric, such as X.",
    )
    mana_cost: Optional[str] = Field(
        None,
        description="The mana cost for this card. This value will be any empty string if the cost is absent. Multi-faced cards will report this value in card faces.",
    )
    name: str = Field(
        description="The name of this card. If this card has multiple faces, this field will contain both names separated by //."
    )
    oracle_text: Optional[str] = Field(
        None, description="The Oracle text for this card, if any."
    )
    penny_rank: Optional[int] = Field(
        None,
        description="This card's rank/popularity on Penny Dreadful. Not all cards are ranked.",
    )
    power: Optional[str] = Field(
        None,
        description="This card's power, if any. Note that some cards have powers that are not numeric, such as *.",
    )
    produced_mana: Optional[list[str]] = Field(
        None, description="Colors of mana that this card could produce."
    )
    reserved: bool = Field(description="True if this card is on the Reserved List.")
    toughness: Optional[str] = Field(
        None,
        description="This card's toughness, if any. Note that some cards have toughnesses that are not numeric, such as *.",
    )
    type_line: Optional[str] = Field(
        default=None,
        description="The type line of this card. Absent for reversible_card layout — use card_faces[*].type_line instead.",
    )

    # --- Print fields ---
    artist: Optional[str] = Field(
        None,
        description="The name of the illustrator of this card. Newly spoiled cards may not have this field yet.",
    )
    artist_ids: Optional[list[UUID]] = Field(
        None,
        description="The IDs of the artists that illustrated this card. Newly spoiled cards may not have this field yet.",
    )
    attraction_lights: Optional[list[int]] = Field(
        None, description="The lit Unfinity attraction lights on this card, if any."
    )
    booster: bool = Field(description="Whether this card is found in boosters.")
    border_color: str = Field(
        description="This card's border color: black, white, borderless, yellow, silver, or gold."
    )
    card_back_id: Optional[UUID] = Field(
        None,
        description="The Scryfall ID for the card back design present on this card.",
    )
    collector_number: str = Field(
        description="This card's collector number. Note that collector numbers can contain non-numeric characters, such as letters or a star symbol."
    )
    content_warning: Optional[bool] = Field(
        None,
        description="True if you should consider avoiding use of this print downstream.",
    )
    digital: bool = Field(
        description="True if this card was only released in a video game."
    )
    finishes: list[str] = Field(
        description="An array of computer-readable flags that indicate if this card can come in foil, nonfoil, or etched finishes."
    )
    flavor_name: Optional[str] = Field(
        None,
        description="The just-for-fun name printed on the card (such as for Godzilla series cards).",
    )
    flavor_text: Optional[str] = Field(None, description="The flavor text, if any.")
    frame_effects: Optional[list[str]] = Field(
        None, description="This card's frame effects, if any."
    )
    frame: str = Field(description="This card's frame layout.")
    full_art: bool = Field(
        description="True if this card's artwork is larger than normal."
    )
    games: list[str] = Field(
        description="A list of games that this card print is available in: paper, arena, mtgo, astral, and/or sega."
    )
    highres_image: bool = Field(
        description="True if this card's imagery is high resolution."
    )
    illustration_id: Optional[UUID] = Field(
        None,
        description="A unique identifier for the card artwork that remains consistent across reprints. Newly spoiled cards may not have this field yet.",
    )
    image_status: str = Field(
        description="A computer-readable indicator for the state of this card's image, one of missing, placeholder, lowres, or highres_scan."
    )
    image_uris: Optional[dict[str, str]] = Field(
        None, description="An object listing available imagery for this card."
    )
    oversized: bool = Field(description="True if this card is oversized.")
    prices: ScryfallPrices = Field(
        description="An object containing daily price information for this card, including usd, usd_foil, usd_etched, eur, eur_foil, eur_etched, and tix prices, as strings."
    )
    printed_name: Optional[str] = Field(
        None, description="The localized name printed on this card, if any."
    )
    printed_text: Optional[str] = Field(
        None, description="The localized text printed on this card, if any."
    )
    printed_type_line: Optional[str] = Field(
        None, description="The localized type line printed on this card, if any."
    )
    promo: bool = Field(description="True if this card is a promotional print.")
    promo_types: Optional[list[str]] = Field(
        None,
        description="An array of strings describing what categories of promo cards this card falls into.",
    )
    purchase_uris: Optional[dict[str, str]] = Field(
        None,
        description="An object providing URIs to this card's listing on major marketplaces. Omitted if the card is unpurchaseable.",
    )
    rarity: str = Field(
        description="This card's rarity. One of common, uncommon, rare, special, mythic, or bonus."
    )
    related_uris: dict[str, str] = Field(
        description="An object providing URIs to this card's listing on other Magic: The Gathering online resources."
    )
    released_at: date = Field(description="The date this card was first released.")
    reprint: bool = Field(description="True if this card is a reprint.")
    scryfall_set_uri: str = Field(
        description="A link to this card's set on Scryfall's website."
    )
    set_name: str = Field(description="This card's full set name.")
    set_search_uri: str = Field(
        description="A link to where you can begin paginating this card's set on the Scryfall API."
    )
    set_type: str = Field(description="The type of set this printing is in.")
    set_uri: str = Field(
        description="A link to this card's set object on Scryfall's API."
    )
    set: str = Field(description="This card's set code.")
    set_id: UUID = Field(description="This card's Set object UUID.")
    story_spotlight: bool = Field(description="True if this card is a Story Spotlight.")
    textless: bool = Field(description="True if the card is printed without text.")
    variation: bool = Field(
        description="Whether this card is a variation of another printing."
    )
    variation_of: Optional[UUID] = Field(
        None, description="The printing ID of the printing this card is a variation of."
    )
    security_stamp: Optional[str] = Field(
        None,
        description="The security stamp on this card, if any. One of oval, triangle, acorn, circle, arena, or heart.",
    )
    watermark: Optional[str] = Field(None, description="This card's watermark, if any.")
    preview: Optional[ScryfallPreview] = Field(
        None,
        description="Preview information for this card, including source, source URI, and preview date.",
    )
