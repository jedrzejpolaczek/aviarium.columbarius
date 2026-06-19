from datetime import date
from uuid import UUID

import pytest
from pydantic import ValidationError

from src.data.dataclasses.scryfall import (
    ScryfallCard,
    ScryfallCardFace,
    ScryfallPreview,
    ScryfallPrices,
    ScryfallRelatedCard,
)

# ---------------------------------------------------------------------------
# Shared UUIDs used across tests
# ---------------------------------------------------------------------------

CARD_ID = "00000000-0000-0000-0000-000000000001"
SET_ID = "00000000-0000-0000-0000-000000000002"
ORACLE_ID = "00000000-0000-0000-0000-000000000003"
RELATED_ID = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"


def _minimal_card() -> dict:
    """Minimum required fields for a valid ScryfallCard."""
    return {
        "id": CARD_ID,
        "lang": "en",
        "object": "card",
        "layout": "normal",
        "oracle_id": ORACLE_ID,
        "prints_search_uri": "https://api.scryfall.com/cards/search",
        "rulings_uri": "https://api.scryfall.com/cards/ruling",
        "scryfall_uri": "https://scryfall.com/card/abc/1",
        "uri": "https://api.scryfall.com/cards/00000000-0000-0000-0000-000000000001",
        "cmc": 3.0,
        "color_identity": ["U"],
        "keywords": [],
        "legalities": {"standard": "legal", "modern": "legal"},
        "name": "Test Wizard",
        "reserved": False,
        "type_line": "Creature — Wizard",
        "booster": True,
        "border_color": "black",
        "collector_number": "42",
        "digital": False,
        "finishes": ["nonfoil", "foil"],
        "frame": "2015",
        "full_art": False,
        "games": ["paper"],
        "highres_image": True,
        "image_status": "highres_scan",
        "oversized": False,
        "prices": {"usd": "1.00", "usd_foil": "2.50"},
        "promo": False,
        "rarity": "uncommon",
        "related_uris": {"gatherer": "https://gatherer.wizards.com/"},
        "released_at": "2023-01-01",
        "reprint": False,
        "scryfall_set_uri": "https://scryfall.com/sets/abc",
        "set_name": "Test Set",
        "set_search_uri": "https://api.scryfall.com/cards/search?q=set%3Aabc",
        "set_type": "expansion",
        "set_uri": "https://api.scryfall.com/sets/abc",
        "set": "abc",
        "set_id": SET_ID,
        "story_spotlight": False,
        "textless": False,
        "variation": False,
    }


# ---------------------------------------------------------------------------
# ScryfallPrices
# ---------------------------------------------------------------------------


class TestScryfallPrices:
    def test_empty_is_valid(self):
        prices = ScryfallPrices()
        assert prices.usd is None
        assert prices.usd_foil is None
        assert prices.tix is None

    def test_string_prices_coerced_to_float(self):
        prices = ScryfallPrices(usd="1.23", eur="0.80", tix="0.05")
        assert prices.usd == 1.23
        assert prices.eur == 0.80
        assert prices.tix == 0.05

    def test_zero_price_string_accepted(self):
        prices = ScryfallPrices(usd="0.00")
        assert prices.usd == 0.0

    def test_numeric_values_accepted(self):
        prices = ScryfallPrices(usd=2.5, usd_foil=5.0)
        assert prices.usd == 2.5
        assert prices.usd_foil == 5.0

    def test_none_values_stay_none(self):
        prices = ScryfallPrices(usd=None, usd_etched=None)
        assert prices.usd is None
        assert prices.usd_etched is None

    def test_all_fields_populated(self):
        prices = ScryfallPrices(
            usd="1.00",
            usd_foil="2.00",
            usd_etched="3.00",
            eur="0.80",
            eur_foil="1.60",
            eur_etched="2.40",
            tix="0.05",
        )
        assert prices.usd == 1.00
        assert prices.usd_foil == 2.00
        assert prices.usd_etched == 3.00
        assert prices.eur == 0.80
        assert prices.eur_foil == 1.60
        assert prices.eur_etched == 2.40
        assert prices.tix == 0.05

    def test_non_numeric_string_raises(self):
        with pytest.raises(ValidationError):
            ScryfallPrices(usd="not_a_price")

    def test_empty_string_raises(self):
        with pytest.raises(ValidationError):
            ScryfallPrices(usd="")


# ---------------------------------------------------------------------------
# ScryfallPreview
# ---------------------------------------------------------------------------


class TestScryfallPreview:
    def test_empty_is_valid(self):
        preview = ScryfallPreview()
        assert preview.previewed_at is None
        assert preview.source is None
        assert preview.source_uri is None

    def test_all_fields(self):
        preview = ScryfallPreview(
            previewed_at="2023-06-15",
            source="Weekly MTG",
            source_uri="https://example.com/preview",
        )
        assert preview.previewed_at == date(2023, 6, 15)
        assert preview.source == "Weekly MTG"
        assert preview.source_uri == "https://example.com/preview"

    def test_date_string_parsed(self):
        preview = ScryfallPreview(previewed_at="2022-09-01")
        assert isinstance(preview.previewed_at, date)
        assert preview.previewed_at == date(2022, 9, 1)


# ---------------------------------------------------------------------------
# ScryfallRelatedCard
# ---------------------------------------------------------------------------


class TestScryfallRelatedCard:
    def _valid_data(self) -> dict:
        return {
            "id": RELATED_ID,
            "object": "related_card",
            "component": "token",
            "name": "Goblin Token",
            "type_line": "Token Creature — Goblin",
            "uri": "https://api.scryfall.com/cards/tok",
        }

    def test_valid(self):
        card = ScryfallRelatedCard(**self._valid_data())
        assert card.name == "Goblin Token"
        assert card.component == "token"

    def test_id_is_uuid_object(self):
        card = ScryfallRelatedCard(**self._valid_data())
        assert isinstance(card.id, UUID)
        assert str(card.id) == RELATED_ID

    @pytest.mark.parametrize(
        "missing_field", ["id", "object", "component", "name", "type_line", "uri"]
    )
    def test_missing_required_field_raises(self, missing_field):
        data = self._valid_data()
        del data[missing_field]
        with pytest.raises(ValidationError):
            ScryfallRelatedCard(**data)

    def test_all_component_types(self):
        for component in ("token", "meld_part", "meld_result", "combo_piece"):
            card = ScryfallRelatedCard(**{**self._valid_data(), "component": component})
            assert card.component == component


# ---------------------------------------------------------------------------
# ScryfallCardFace
# ---------------------------------------------------------------------------


class TestScryfallCardFace:
    def _minimal(self) -> dict:
        return {"mana_cost": "{1}{U}", "name": "Front Face", "object": "card_face"}

    def test_minimal_valid(self):
        face = ScryfallCardFace(**self._minimal())
        assert face.name == "Front Face"
        assert face.mana_cost == "{1}{U}"

    def test_empty_mana_cost_accepted(self):
        face = ScryfallCardFace(**{**self._minimal(), "mana_cost": ""})
        assert face.mana_cost == ""

    def test_optional_fields_default_to_none(self):
        face = ScryfallCardFace(**self._minimal())
        assert face.artist is None
        assert face.colors is None
        assert face.color_indicator is None
        assert face.oracle_text is None
        assert face.power is None
        assert face.toughness is None
        assert face.loyalty is None
        assert face.image_uris is None

    def test_with_optional_fields(self):
        face = ScryfallCardFace(
            **{
                **self._minimal(),
                "colors": ["U", "R"],
                "power": "2",
                "toughness": "*",
                "oracle_text": "Flying",
                "artist": "Jane Smith",
            }
        )
        assert face.colors == ["U", "R"]
        assert face.power == "2"
        assert face.toughness == "*"
        assert face.artist == "Jane Smith"

    def test_non_numeric_power_accepted(self):
        face = ScryfallCardFace(**{**self._minimal(), "power": "*", "toughness": "1+*"})
        assert face.power == "*"
        assert face.toughness == "1+*"

    @pytest.mark.parametrize("missing_field", ["mana_cost", "name", "object"])
    def test_missing_required_field_raises(self, missing_field):
        data = self._minimal()
        del data[missing_field]
        with pytest.raises(ValidationError):
            ScryfallCardFace(**data)


# ---------------------------------------------------------------------------
# ScryfallCard
# ---------------------------------------------------------------------------


class TestScryfallCard:
    def test_minimal_valid(self):
        card = ScryfallCard.model_validate(_minimal_card())
        assert card.name == "Test Wizard"
        assert card.lang == "en"

    def test_id_is_uuid(self):
        card = ScryfallCard.model_validate(_minimal_card())
        assert isinstance(card.id, UUID)
        assert str(card.id) == CARD_ID

    def test_set_id_is_uuid(self):
        card = ScryfallCard.model_validate(_minimal_card())
        assert isinstance(card.set_id, UUID)

    def test_released_at_parsed_as_date(self):
        card = ScryfallCard.model_validate(_minimal_card())
        assert isinstance(card.released_at, date)
        assert card.released_at == date(2023, 1, 1)

    def test_prices_validated_and_coerced(self):
        card = ScryfallCard.model_validate(_minimal_card())
        assert isinstance(card.prices, ScryfallPrices)
        assert card.prices.usd == 1.00
        assert card.prices.usd_foil == 2.50

    def test_optional_fields_default_to_none(self):
        card = ScryfallCard.model_validate(_minimal_card())
        assert card.arena_id is None
        assert card.card_faces is None
        assert card.all_parts is None
        assert card.preview is None
        assert card.colors is None
        assert card.oracle_text is None
        assert card.flavor_text is None
        assert card.power is None
        assert card.toughness is None

    def test_unknown_fields_ignored(self):
        data = {**_minimal_card(), "totally_unknown_field": "ignored"}
        card = ScryfallCard.model_validate(data)
        assert card.name == "Test Wizard"

    @pytest.mark.parametrize(
        "missing_field",
        [
            "id",
            "lang",
            "name",
            "object",
            "layout",
            "color_identity",
            "keywords",
            "legalities",
            "reserved",
            "booster",
            "border_color",
            "collector_number",
            "digital",
            "finishes",
            "frame",
            "full_art",
            "games",
            "highres_image",
            "image_status",
            "oversized",
            "prices",
            "promo",
            "rarity",
            "related_uris",
            "released_at",
            "reprint",
            "set",
            "set_id",
            "set_name",
            "set_type",
            "story_spotlight",
            "textless",
            "variation",
        ],
    )
    def test_missing_required_field_raises(self, missing_field):
        data = _minimal_card()
        del data[missing_field]
        with pytest.raises(ValidationError):
            ScryfallCard.model_validate(data)

    def test_with_all_parts(self):
        data = {
            **_minimal_card(),
            "all_parts": [
                {
                    "id": RELATED_ID,
                    "object": "related_card",
                    "component": "token",
                    "name": "Goblin Token",
                    "type_line": "Token Creature — Goblin",
                    "uri": "https://api.scryfall.com/cards/tok",
                }
            ],
        }
        card = ScryfallCard.model_validate(data)
        assert card.all_parts is not None
        assert len(card.all_parts) == 1
        assert card.all_parts[0].name == "Goblin Token"
        assert isinstance(card.all_parts[0], ScryfallRelatedCard)

    def test_with_card_faces(self):
        data = {
            **_minimal_card(),
            "layout": "transform",
            "card_faces": [
                {"mana_cost": "{2}{U}", "name": "Day Side", "object": "card_face"},
                {"mana_cost": "", "name": "Night Side", "object": "card_face"},
            ],
        }
        card = ScryfallCard.model_validate(data)
        assert card.card_faces is not None
        assert len(card.card_faces) == 2
        assert card.card_faces[0].name == "Day Side"
        assert card.card_faces[1].mana_cost == ""
        assert all(isinstance(f, ScryfallCardFace) for f in card.card_faces)

    def test_with_preview(self):
        data = {
            **_minimal_card(),
            "preview": {
                "previewed_at": "2022-08-18",
                "source": "Card Preview Show",
                "source_uri": "https://example.com/preview",
            },
        }
        card = ScryfallCard.model_validate(data)
        assert isinstance(card.preview, ScryfallPreview)
        assert card.preview.previewed_at == date(2022, 8, 18)
        assert card.preview.source == "Card Preview Show"

    def test_zero_cmc_accepted(self):
        data = {**_minimal_card(), "cmc": 0.0, "type_line": "Land"}
        card = ScryfallCard.model_validate(data)
        assert card.cmc == 0.0

    def test_fractional_cmc_accepted(self):
        data = {**_minimal_card(), "cmc": 0.5}
        card = ScryfallCard.model_validate(data)
        assert card.cmc == 0.5

    def test_oracle_id_optional_for_reversible_layout(self):
        data = {**_minimal_card(), "layout": "reversible_card"}
        del data["oracle_id"]
        card = ScryfallCard.model_validate(data)
        assert card.oracle_id is None

    def test_cmc_and_type_line_optional_for_reversible_card(self):
        data = {**_minimal_card(), "layout": "reversible_card"}
        del data["cmc"]
        del data["type_line"]
        card = ScryfallCard.model_validate(data)
        assert card.cmc is None
        assert card.type_line is None

    def test_with_optional_platform_ids(self):
        data = {
            **_minimal_card(),
            "arena_id": 12345,
            "mtgo_id": 67890,
            "tcgplayer_id": 11111,
            "cardmarket_id": 22222,
        }
        card = ScryfallCard.model_validate(data)
        assert card.arena_id == 12345
        assert card.mtgo_id == 67890
        assert card.tcgplayer_id == 11111
        assert card.cardmarket_id == 22222

    def test_multicolor_identity(self):
        data = {**_minimal_card(), "color_identity": ["U", "R", "G"]}
        card = ScryfallCard.model_validate(data)
        assert card.color_identity == ["U", "R", "G"]

    def test_colorless_card(self):
        data = {**_minimal_card(), "color_identity": [], "colors": []}
        card = ScryfallCard.model_validate(data)
        assert card.color_identity == []
        assert card.colors == []
