"""Pydantic models for MTGJson card and price data.

Covers three MTGJson bulk-data endpoints:
    MtgjsonSet         — set-level metadata from AllPrintings
    MtgjsonCard        — individual printing from AllPrintings (cards[])
    MtgjsonCardPrices  — per-card price data from AllPricesToday (keyed by UUID)

All fields are Optional unless MTGJson guarantees their presence. Fields not
needed downstream are omitted; the models use extra="ignore" so unknown fields
are silently dropped rather than raising a validation error.
"""

from datetime import date as date_time
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# AllPricesToday models
# ---------------------------------------------------------------------------


class MtgjsonPriceListing(BaseModel):
    """Foil/nonfoil price history at a single retailer for a single transaction type.

    Each finish maps to a date-keyed dict of prices, e.g. {"2026-04-29": 4.89}.
    An empty listing (both None) means no prices are available for that transaction type.

    All data information are from MTGjson docummentation: https://mtgjson.com/
    """

    foil: Optional[dict[str, float]] = Field(
        None,
        description="Foil price history keyed by date (YYYY-MM-DD). None if unavailable.",
    )
    normal: Optional[dict[str, float]] = Field(
        None,
        description="Nonfoil price history keyed by date (YYYY-MM-DD). None if unavailable.",
    )


class MtgjsonRetailerPrices(BaseModel):
    """Buylist and retail prices at a single retailer.

    All data information are from MTGjson docummentation: https://mtgjson.com/
    """

    buylist: Optional[MtgjsonPriceListing] = Field(
        None,
        description="Prices the retailer pays when buying cards (buylist / trade-in).",
    )
    retail: Optional[MtgjsonPriceListing] = Field(
        None, description="Prices the retailer charges when selling cards."
    )
    currency: Optional[str] = Field(
        None, description="Currency code for this retailer's prices (e.g. USD, EUR)."
    )


class MtgjsonCardPrices(BaseModel):
    """All prices for a single card UUID from AllPricesToday.json. Keyed by game platform, then retailer.

    All data information are from MTGjson docummentation: https://mtgjson.com/
    """

    uuid: UUID = Field(
        description="The MTGJSON UUID of the card this price record belongs to. Injected from the AllPricesToday.json dict key during extraction."
    )
    paper: Optional[dict[str, MtgjsonRetailerPrices]] = Field(
        None,
        description="Paper prices per retailer (cardkingdom, cardmarket, tcgplayer, etc.).",
    )
    mtgo: Optional[dict[str, MtgjsonRetailerPrices]] = Field(
        None, description="MTGO prices per retailer (cardhoarder)."
    )


# ---------------------------------------------------------------------------
# AllPrintings nested models
# ---------------------------------------------------------------------------


class MtgjsonIdentifiers(BaseModel):
    """Cross-site identifiers linking this card to Scryfall, TCGPlayer, Card Kingdom, MTGO, and other platforms.

    All data information are from MTGjson docummentation: https://mtgjson.com/
    """

    scryfall_id: Optional[UUID] = Field(
        None, alias="scryfallId", description="The Scryfall UUID for this printing."
    )
    scryfall_oracle_id: Optional[UUID] = Field(
        None,
        alias="scryfallOracleId",
        description="The Scryfall oracle UUID, shared across all printings of the same oracle card.",
    )
    scryfall_illustration_id: Optional[UUID] = Field(
        None,
        alias="scryfallIllustrationId",
        description="The Scryfall illustration UUID, shared across printings with the same artwork.",
    )
    scryfall_card_back_id: Optional[UUID] = Field(
        None,
        alias="scryfallCardBackId",
        description="The Scryfall UUID of the card back design.",
    )
    mtgo_id: Optional[str] = Field(
        None, alias="mtgoId", description="The Magic Online nonfoil catalog ID."
    )
    mtgo_foil_id: Optional[str] = Field(
        None, alias="mtgoFoilId", description="The Magic Online foil catalog ID."
    )
    multiverse_id: Optional[str] = Field(
        None, alias="multiverseId", description="The Gatherer multiverse ID."
    )
    tcgplayer_product_id: Optional[str] = Field(
        None, alias="tcgplayerProductId", description="The TCGPlayer product ID."
    )
    card_kingdom_id: Optional[str] = Field(
        None, alias="cardKingdomId", description="The Card Kingdom nonfoil product ID."
    )
    card_kingdom_foil_id: Optional[str] = Field(
        None, alias="cardKingdomFoilId", description="The Card Kingdom foil product ID."
    )
    mcm_id: Optional[str] = Field(
        None, alias="mcmId", description="The Cardmarket (MCM) product ID."
    )
    mcm_meta_id: Optional[str] = Field(
        None, alias="mcmMetaId", description="The Cardmarket meta product ID."
    )
    cardsphere_id: Optional[str] = Field(
        None, alias="cardsphereId", description="The Cardsphere nonfoil product ID."
    )
    cardsphere_foil_id: Optional[str] = Field(
        None, alias="cardsphereFoilId", description="The Cardsphere foil product ID."
    )
    deckbox_id: Optional[str] = Field(
        None, alias="deckboxId", description="The Deckbox product ID."
    )
    mtgjson_v4_id: Optional[UUID] = Field(
        None,
        alias="mtgjsonV4Id",
        description="The MTGJSON v4 UUID for backwards compatibility.",
    )
    mtgjson_foil_version_id: Optional[UUID] = Field(
        None,
        alias="mtgjsonFoilVersionId",
        description="The MTGJSON UUID for the foil version of this card.",
    )

    model_config = {"populate_by_name": True}


class MtgjsonLeadershipSkills(BaseModel):
    """Which Commander-variant formats this card is eligible to lead as a commander/leader.

    All data information are from MTGjson docummentation: https://mtgjson.com/
    """

    brawl: bool = Field(
        description="True if this card can be used as a Brawl commander."
    )
    commander: bool = Field(description="True if this card can be used as a Commander.")
    oathbreaker: bool = Field(
        description="True if this card can be used as an Oathbreaker."
    )


class MtgjsonRuling(BaseModel):
    """A single official ruling or clarification for a card, as published by Wizards of the Coast.

    All data information are from MTGjson docummentation: https://mtgjson.com/
    """

    date: date_time = Field(description="The date the ruling was issued.")
    text: str = Field(description="The ruling text.")


class MtgjsonForeignData(BaseModel):
    """Localized card data for a non-English language printing of the same card.

    All data information are from MTGjson docummentation: https://mtgjson.com/
    """

    flavor_text: Optional[str] = Field(
        None,
        alias="flavorText",
        description="The flavor text in this language, if any.",
    )
    language: str = Field(description="The language name, e.g. 'French', 'Japanese'.")
    multiverse_id: Optional[int] = Field(
        None,
        alias="multiverseId",
        description="The Gatherer multiverse ID for this language printing.",
    )
    name: str = Field(description="The card name in this language.")
    text: Optional[str] = Field(
        None, description="The oracle text in this language, if any."
    )
    type: Optional[str] = Field(
        None, description="The type line in this language, if any."
    )
    uuid: Optional[UUID] = Field(
        None, description="The MTGJSON UUID for this language printing."
    )

    model_config = {"populate_by_name": True}


# ---------------------------------------------------------------------------
# AllPrintings — Set
# ---------------------------------------------------------------------------


class MtgjsonSet(BaseModel):
    """Metadata for a single Magic: The Gathering set as found in AllPrintings.json under data.<SET_CODE>.

    All data information are from MTGjson docummentation: https://mtgjson.com/
    """

    base_set_size: int = Field(
        alias="baseSetSize",
        description="The number of cards in the set's base print run, excluding variants.",
    )
    block: Optional[str] = Field(
        None, description="The block this set belongs to, if any."
    )
    code: str = Field(description="The set code, e.g. 'BLB', '10E'.")
    is_foil_only: bool = Field(
        alias="isFoilOnly", description="True if all cards in this set are foil-only."
    )
    is_online_only: bool = Field(
        alias="isOnlineOnly",
        description="True if this set was released exclusively on MTGO or Arena.",
    )
    keyrune_code: str = Field(
        alias="keyruneCode",
        description="The Keyrune font code for this set's expansion symbol.",
    )
    languages: list[str] = Field(description="Languages this set was printed in.")
    mcm_id: Optional[int] = Field(
        None, alias="mcmId", description="The Cardmarket product ID for this set."
    )
    mcm_name: Optional[str] = Field(
        None, alias="mcmName", description="The Cardmarket name for this set."
    )
    mtgo_code: Optional[str] = Field(
        None,
        alias="mtgoCode",
        description="The MTGO set code, if different from the paper code.",
    )
    name: str = Field(description="The full set name, e.g. 'Bloomburrow'.")
    release_date: date_time = Field(
        alias="releaseDate", description="The official release date of this set."
    )
    tcgplayer_group_id: Optional[int] = Field(
        None,
        alias="tcgplayerGroupId",
        description="The TCGPlayer group ID for this set.",
    )
    token_set_code: Optional[str] = Field(
        None,
        alias="tokenSetCode",
        description="The set code for this set's associated token sheet.",
    )
    total_set_size: int = Field(
        alias="totalSetSize",
        description="The total number of cards in the set including all variants.",
    )
    translations: dict[str, Optional[str]] = Field(
        description="Set name translations keyed by language name."
    )
    type: str = Field(
        description="The set type: core, expansion, masters, draft_innovation, funny, memorabilia, etc."
    )

    model_config = {"populate_by_name": True}


# ---------------------------------------------------------------------------
# AllPrintings — Card
# ---------------------------------------------------------------------------


class MtgjsonCard(BaseModel):
    """A single Magic: The Gathering card printing as found in AllPrintings.json under data.<SET_CODE>.cards. One record per unique print within a set.

    All data information are from MTGjson docummentation: https://mtgjson.com/
    """

    # --- Identity ---
    uuid: UUID = Field(
        description="The MTGJSON UUID for this card printing. Unique per print. Used as the key in AllPricesToday.json."
    )
    name: str = Field(
        description="The full card name. Multi-faced cards use ' // ' as separator."
    )
    ascii_name: Optional[str] = Field(
        None,
        alias="asciiName",
        description="ASCII-safe version of the card name, for cards with non-ASCII characters.",
    )
    set_code: str = Field(
        alias="setCode",
        description="The set code this card belongs to, e.g. '10E', 'BLB'.",
    )
    number: str = Field(
        description="The collector number within the set. May contain non-numeric characters."
    )
    language: str = Field(
        description="The language of this printing, e.g. 'English', 'Japanese'."
    )

    # --- Oracle / Gameplay ---
    layout: str = Field(
        description="The card layout: normal, split, flip, transform, meld, leveler, saga, adventure, etc."
    )
    mana_cost: Optional[str] = Field(
        None,
        alias="manaCost",
        description="The mana cost string, e.g. '{3}{U}{U}{U}'. Absent for cards with no mana cost.",
    )
    mana_value: float = Field(
        alias="manaValue", description="The numeric mana value (converted mana cost)."
    )
    converted_mana_cost: float = Field(
        alias="convertedManaCost",
        description="Deprecated alias for manaValue, retained for backwards compatibility.",
    )
    type: str = Field(
        description="The full type line as printed, e.g. 'Legendary Creature — Wizard'."
    )
    types: list[str] = Field(
        description="Card types parsed into an array, e.g. ['Legendary', 'Creature']."
    )
    supertypes: list[str] = Field(
        description="Supertypes parsed into an array, e.g. ['Legendary', 'Basic', 'Snow']."
    )
    subtypes: list[str] = Field(
        description="Subtypes parsed into an array, e.g. ['Wizard', 'Forest']."
    )
    text: Optional[str] = Field(
        None, description="The current oracle text of the card."
    )
    original_text: Optional[str] = Field(
        None,
        alias="originalText",
        description="The original printed text before any oracle errata.",
    )
    original_type: Optional[str] = Field(
        None,
        alias="originalType",
        description="The original printed type line before any oracle errata.",
    )
    colors: list[str] = Field(
        description="The colors of the card: W, U, B, R, G. Empty list for colorless."
    )
    color_identity: list[str] = Field(
        alias="colorIdentity",
        description="The Commander color identity, including mana symbols in card text.",
    )
    color_indicator: Optional[list[str]] = Field(
        None,
        alias="colorIndicator",
        description="Colors shown in the color indicator dot, for cards whose color is not derived from mana cost.",
    )
    keywords: Optional[list[str]] = Field(
        None, description="Oracle keywords on this card, e.g. ['Flying', 'Trample']."
    )
    produced_mana: Optional[list[str]] = Field(
        None, alias="producedMana", description="Mana colors this card can produce."
    )
    power: Optional[str] = Field(
        None,
        description="The creature's power. String because it can be '*', '1+*', etc.",
    )
    toughness: Optional[str] = Field(
        None, description="The creature's toughness. String because it can be '*'."
    )
    loyalty: Optional[str] = Field(
        None, description="The planeswalker's starting loyalty."
    )
    defense: Optional[str] = Field(None, description="The battle card's defense value.")
    hand: Optional[str] = Field(
        None, description="The Vanguard card's hand size modifier, e.g. '+1'."
    )
    life: Optional[str] = Field(
        None, description="The Vanguard card's life total modifier, e.g. '+2'."
    )
    legalities: dict[str, str] = Field(
        description="Format legality map. Values are 'Legal', 'Banned', 'Restricted', or 'Not Legal'."
    )
    leadership_skills: Optional[MtgjsonLeadershipSkills] = Field(
        None,
        alias="leadershipSkills",
        description="Which Commander-variant formats this card is eligible to lead.",
    )
    rulings: Optional[list[MtgjsonRuling]] = Field(
        None,
        description="Official rulings embedded directly in the record (unlike Scryfall which only provides a URI).",
    )

    # --- Print properties ---
    rarity: str = Field(
        description="The card's rarity: common, uncommon, rare, mythic, special, or bonus."
    )
    border_color: str = Field(
        alias="borderColor",
        description="The border color: black, white, borderless, gold, or silver.",
    )
    frame_version: str = Field(
        alias="frameVersion",
        description="The frame era: 1993, 1997, 2003, 2015, or future.",
    )
    frame_effects: Optional[list[str]] = Field(
        None,
        alias="frameEffects",
        description="Special frame treatments: legendary, miracle, extendedart, companion, etched, snow, lesson, etc.",
    )
    finishes: list[str] = Field(
        description="Available finishes: nonfoil, foil, etched."
    )
    availability: list[str] = Field(
        description="Platforms where this card is available: paper, mtgo, arena."
    )
    booster_types: Optional[list[str]] = Field(
        None,
        alias="boosterTypes",
        description="Which booster sheet types include this card.",
    )
    watermark: Optional[str] = Field(
        None, description="The watermark on this card, if any."
    )
    security_stamp: Optional[str] = Field(
        None,
        alias="securityStamp",
        description="The security stamp type: oval, triangle, acorn, circle, arena, or heart.",
    )
    flavor_text: Optional[str] = Field(
        None,
        alias="flavorText",
        description="The flavor text as printed on this specific version.",
    )
    flavor_name: Optional[str] = Field(
        None,
        alias="flavorName",
        description="The alternate flavor name printed on showcase or crossover cards.",
    )
    artist: Optional[str] = Field(
        None, description="The illustrator name(s) as credited on the card."
    )
    artist_ids: Optional[list[UUID]] = Field(
        None,
        alias="artistIds",
        description="MTGJSON UUIDs for the illustrators of this card.",
    )
    attraction_lights: Optional[list[int]] = Field(
        None,
        alias="attractionLights",
        description="Lit attraction light numbers for Unfinity attraction cards.",
    )
    printed_name: Optional[str] = Field(
        None,
        alias="printedName",
        description="The localized name as printed, for non-English cards.",
    )
    printed_text: Optional[str] = Field(
        None,
        alias="printedText",
        description="The localized text as printed, for non-English cards.",
    )
    printed_type: Optional[str] = Field(
        None,
        alias="printedType",
        description="The localized type line as printed, for non-English cards.",
    )
    face_name: Optional[str] = Field(
        None,
        alias="faceName",
        description="The name of this specific face for multi-faced cards.",
    )
    face_flavor_name: Optional[str] = Field(
        None,
        alias="faceFlavorName",
        description="The flavor name of this face for multi-faced showcase cards.",
    )
    face_mana_value: Optional[float] = Field(
        None,
        alias="faceManaValue",
        description="The mana value of this face for reversible cards.",
    )
    face_converted_mana_cost: Optional[float] = Field(
        None,
        alias="faceConvertedManaCost",
        description="Deprecated alias for faceManaValue.",
    )
    face_printed_name: Optional[str] = Field(
        None,
        alias="facePrintedName",
        description="The localized printed name of this face.",
    )
    side: Optional[str] = Field(
        None,
        description="Which side of a double-faced card this is: 'a' (front) or 'b' (back).",
    )
    other_face_ids: Optional[list[UUID]] = Field(
        None,
        alias="otherFaceIds",
        description="MTGJSON UUIDs of the other faces of this card.",
    )
    card_parts: Optional[list[str]] = Field(
        None,
        alias="cardParts",
        description="Names of the parts of split or adventure cards.",
    )
    duel_deck: Optional[str] = Field(
        None,
        alias="duelDeck",
        description="Which deck of a Duel Deck product this card is from: 'a' or 'b'.",
    )
    signature: Optional[str] = Field(
        None, description="The signature on this card, for autographed promo cards."
    )
    subsets: Optional[list[str]] = Field(
        None, description="Subset codes this card belongs to within its set."
    )

    # --- Boolean flags ---
    is_reprint: Optional[bool] = Field(
        None,
        alias="isReprint",
        description="True if this card has been printed in a previous set.",
    )
    is_reserved: Optional[bool] = Field(
        None,
        alias="isReserved",
        description="True if this card is on the Reserved List.",
    )
    is_promo: Optional[bool] = Field(
        None, alias="isPromo", description="True if this is a promotional printing."
    )
    is_full_art: Optional[bool] = Field(
        None,
        alias="isFullArt",
        description="True if this card has a full-art treatment.",
    )
    is_textless: Optional[bool] = Field(
        None,
        alias="isTextless",
        description="True if this card is printed without rules text.",
    )
    is_oversized: Optional[bool] = Field(
        None, alias="isOversized", description="True if this is an oversized card."
    )
    is_online_only: Optional[bool] = Field(
        None,
        alias="isOnlineOnly",
        description="True if this card was released exclusively in a digital format.",
    )
    is_alternative: Optional[bool] = Field(
        None,
        alias="isAlternative",
        description="True if this is an alternative version of a card in the same set.",
    )
    is_funny: Optional[bool] = Field(
        None,
        alias="isFunny",
        description="True if this card is from an Un-set or acorn product.",
    )
    is_story_spotlight: Optional[bool] = Field(
        None,
        alias="isStorySpotlight",
        description="True if this card is a Story Spotlight card.",
    )
    is_timeshifted: Optional[bool] = Field(
        None,
        alias="isTimeshifted",
        description="True if this card is timeshifted from a different era.",
    )
    is_rebalanced: Optional[bool] = Field(
        None,
        alias="isRebalanced",
        description="True if this card has been rebalanced for MTG Arena's Alchemy format.",
    )
    is_game_changer: Optional[bool] = Field(
        None,
        alias="isGameChanger",
        description="True if this card is designated a format-defining Game Changer.",
    )
    has_alternative_deck_limit: Optional[bool] = Field(
        None,
        alias="hasAlternativeDeckLimit",
        description="True if this card has a non-standard deck limit.",
    )
    has_content_warning: Optional[bool] = Field(
        None,
        alias="hasContentWarning",
        description="True if this card has been flagged with a content warning.",
    )
    promo_types: Optional[list[str]] = Field(
        None,
        alias="promoTypes",
        description="Categories of promo treatment: prerelease, datestamped, buyabox, fnm, etc.",
    )

    # --- Print history ---
    printings: Optional[list[str]] = Field(
        None, description="All set codes in which this oracle card has been printed."
    )
    original_release_date: Optional[str] = Field(
        None,
        alias="originalReleaseDate",
        description="Original release date override for some promos.",
    )
    variations: Optional[list[UUID]] = Field(
        None,
        description="UUIDs of other printings that are variations of this card within the same set.",
    )
    original_printings: Optional[list[str]] = Field(
        None,
        alias="originalPrintings",
        description="Set codes of the original printings before this card was rebalanced.",
    )
    rebalanced_printings: Optional[list[str]] = Field(
        None,
        alias="rebalancedPrintings",
        description="Set codes of the Alchemy-rebalanced versions of this card.",
    )

    # --- Cross-references ---
    identifiers: MtgjsonIdentifiers = Field(
        description="Cross-site identifiers linking this card to Scryfall, TCGPlayer, Card Kingdom, MTGO, and other platforms."
    )
    foreign_data: Optional[list[MtgjsonForeignData]] = Field(
        None,
        alias="foreignData",
        description="Localized data for non-English printings of this card.",
    )
    related_cards: Optional[dict[str, Any]] = Field(
        None,
        alias="relatedCards",
        description="Cards related by game mechanic (tokens, meld pairs, etc.).",
    )
    source_products: Optional[dict[str, Any]] = Field(
        None,
        alias="sourceProducts",
        description="Products (precon decks, bundles) that contain this card.",
    )
    sku_ids: Optional[dict[str, Any]] = Field(
        None,
        alias="skuIds",
        description="SKU IDs per finish for inventory and marketplace tracking.",
    )
    purchase_urls: Optional[dict[str, str]] = Field(
        None,
        alias="purchaseUrls",
        description="Purchase links on Card Kingdom, Cardmarket, and TCGPlayer.",
    )

    # --- EDHREC ---
    edhrec_rank: Optional[int] = Field(
        None,
        alias="edhrecRank",
        description="Popularity rank on EDHREC. Lower is more popular.",
    )
    edhrec_saltiness: Optional[float] = Field(
        None,
        alias="edhrecSaltiness",
        description="How controversial this card is among Commander players (0.0–1.0+). Higher means more disliked.",
    )

    model_config = {"populate_by_name": True}
