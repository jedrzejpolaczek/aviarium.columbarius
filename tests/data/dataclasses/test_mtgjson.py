"""Unit tests for src/data/dataclasses/mtgjson.py Pydantic models."""

from datetime import date
from uuid import UUID

import pytest
from pydantic import ValidationError

from src.data.dataclasses.mtgjson import (
    MtgjsonCard,
    MtgjsonCardPrices,
    MtgjsonForeignData,
    MtgjsonIdentifiers,
    MtgjsonLeadershipSkills,
    MtgjsonPriceListing,
    MtgjsonRetailerPrices,
    MtgjsonRuling,
    MtgjsonSet,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

SAMPLE_UUID = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"

MINIMAL_CARD: dict = {
    "uuid": SAMPLE_UUID,
    "name": "Lightning Bolt",
    "setCode": "LEB",
    "number": "161",
    "language": "English",
    "layout": "normal",
    "manaValue": 1.0,
    "convertedManaCost": 1.0,
    "type": "Instant",
    "types": ["Instant"],
    "supertypes": [],
    "subtypes": [],
    "colors": ["R"],
    "colorIdentity": ["R"],
    "legalities": {"commander": "Legal", "legacy": "Legal"},
    "rarity": "common",
    "borderColor": "black",
    "frameVersion": "2015",
    "finishes": ["nonfoil"],
    "availability": ["paper"],
    "identifiers": {},
}

MINIMAL_SET: dict = {
    "baseSetSize": 302,
    "code": "LEB",
    "isFoilOnly": False,
    "isOnlineOnly": False,
    "keyruneCode": "LEB",
    "languages": ["English"],
    "name": "Limited Edition Beta",
    "releaseDate": "1993-10-04",
    "totalSetSize": 302,
    "translations": {"French": "Édition Limitée Beta"},
    "type": "core",
}


# ---------------------------------------------------------------------------
# MtgjsonPriceListing
# ---------------------------------------------------------------------------


class TestMtgjsonPriceListing:
    def test_valid_both_fields(self):
        p = MtgjsonPriceListing(foil={"2026-04-29": 4.89}, normal={"2026-04-29": 2.50})
        assert p.foil == {"2026-04-29": 4.89}
        assert p.normal == {"2026-04-29": 2.50}

    def test_all_optional_defaults_to_none(self):
        p = MtgjsonPriceListing()
        assert p.foil is None
        assert p.normal is None

    def test_only_foil(self):
        p = MtgjsonPriceListing(foil={"2026-04-29": 9.99})
        assert p.foil == {"2026-04-29": 9.99}
        assert p.normal is None

    def test_only_normal(self):
        p = MtgjsonPriceListing(normal={"2026-04-29": 1.00})
        assert p.foil is None
        assert p.normal == {"2026-04-29": 1.00}


# ---------------------------------------------------------------------------
# MtgjsonRetailerPrices
# ---------------------------------------------------------------------------


class TestMtgjsonRetailerPrices:
    def test_valid_full(self):
        data = {
            "buylist": {"foil": {"2026-04-29": 3.00}},
            "retail": {"normal": {"2026-04-29": 5.00}},
            "currency": "USD",
        }
        r = MtgjsonRetailerPrices.model_validate(data)
        assert r.currency == "USD"
        assert r.buylist is not None
        assert r.buylist.foil == {"2026-04-29": 3.00}
        assert r.retail is not None
        assert r.retail.normal == {"2026-04-29": 5.00}

    def test_all_optional(self):
        r = MtgjsonRetailerPrices()
        assert r.buylist is None
        assert r.retail is None
        assert r.currency is None


# ---------------------------------------------------------------------------
# MtgjsonCardPrices
# ---------------------------------------------------------------------------


class TestMtgjsonCardPrices:
    def test_valid_uuid_coercion(self):
        p = MtgjsonCardPrices(uuid=SAMPLE_UUID)
        assert isinstance(p.uuid, UUID)
        assert str(p.uuid) == SAMPLE_UUID

    def test_paper_and_mtgo_optional(self):
        p = MtgjsonCardPrices(uuid=SAMPLE_UUID)
        assert p.paper is None
        assert p.mtgo is None

    def test_nested_paper_prices(self):
        data = {
            "uuid": SAMPLE_UUID,
            "paper": {
                "cardkingdom": {
                    "buylist": {"normal": {"2026-04-29": 1.50}},
                    "retail": {"normal": {"2026-04-29": 3.00}},
                    "currency": "USD",
                }
            },
        }
        p = MtgjsonCardPrices.model_validate(data)
        assert p.paper is not None
        assert "cardkingdom" in p.paper
        assert p.paper["cardkingdom"].currency == "USD"

    def test_missing_uuid_fails(self):
        with pytest.raises(ValidationError):
            MtgjsonCardPrices.model_validate({})


# ---------------------------------------------------------------------------
# MtgjsonIdentifiers
# ---------------------------------------------------------------------------


class TestMtgjsonIdentifiers:
    def test_all_optional_empty_dict(self):
        i = MtgjsonIdentifiers.model_validate({})
        assert i.scryfall_id is None
        assert i.mtgo_id is None

    def test_camel_case_alias(self):
        i = MtgjsonIdentifiers.model_validate({"scryfallId": SAMPLE_UUID})
        assert isinstance(i.scryfall_id, UUID)
        assert str(i.scryfall_id) == SAMPLE_UUID

    def test_snake_case_via_populate_by_name(self):
        i = MtgjsonIdentifiers.model_validate({"scryfall_id": SAMPLE_UUID})
        assert isinstance(i.scryfall_id, UUID)

    def test_string_uuid_coerced(self):
        i = MtgjsonIdentifiers.model_validate({"scryfallOracleId": SAMPLE_UUID})
        assert isinstance(i.scryfall_oracle_id, UUID)

    def test_string_fields_remain_strings(self):
        i = MtgjsonIdentifiers.model_validate({"tcgplayerProductId": "12345"})
        assert i.tcgplayer_product_id == "12345"

    def test_extra_fields_ignored(self):
        i = MtgjsonIdentifiers.model_validate({"unknownField": "value"})
        assert not hasattr(i, "unknownField")


# ---------------------------------------------------------------------------
# MtgjsonLeadershipSkills
# ---------------------------------------------------------------------------


class TestMtgjsonLeadershipSkills:
    def test_valid(self):
        ls = MtgjsonLeadershipSkills(brawl=False, commander=True, oathbreaker=False)
        assert ls.commander is True
        assert ls.brawl is False

    def test_missing_field_fails(self):
        with pytest.raises(ValidationError):
            MtgjsonLeadershipSkills.model_validate({"brawl": True, "commander": True})


# ---------------------------------------------------------------------------
# MtgjsonRuling
# ---------------------------------------------------------------------------


class TestMtgjsonRuling:
    def test_valid(self):
        r = MtgjsonRuling(date="2004-10-04", text="This card does something.")
        assert isinstance(r.date, date)
        assert r.date == date(2004, 10, 4)
        assert r.text == "This card does something."

    def test_missing_text_fails(self):
        with pytest.raises(ValidationError):
            MtgjsonRuling.model_validate({"date": "2004-10-04"})

    def test_missing_date_fails(self):
        with pytest.raises(ValidationError):
            MtgjsonRuling.model_validate({"text": "Some ruling."})


# ---------------------------------------------------------------------------
# MtgjsonForeignData
# ---------------------------------------------------------------------------


class TestMtgjsonForeignData:
    def test_valid_minimal(self):
        fd = MtgjsonForeignData.model_validate({"language": "French", "name": "Éclair"})
        assert fd.language == "French"
        assert fd.name == "Éclair"
        assert fd.flavor_text is None

    def test_camel_case_alias(self):
        fd = MtgjsonForeignData.model_validate(
            {
                "language": "Japanese",
                "name": "稲妻",
                "flavorText": "速さこそが力だ。",
            }
        )
        assert fd.flavor_text == "速さこそが力だ。"

    def test_missing_language_fails(self):
        with pytest.raises(ValidationError):
            MtgjsonForeignData.model_validate({"name": "Éclair"})

    def test_missing_name_fails(self):
        with pytest.raises(ValidationError):
            MtgjsonForeignData.model_validate({"language": "French"})


# ---------------------------------------------------------------------------
# MtgjsonSet
# ---------------------------------------------------------------------------


class TestMtgjsonSet:
    def test_valid_minimal(self):
        s = MtgjsonSet.model_validate(MINIMAL_SET)
        assert s.code == "LEB"
        assert s.name == "Limited Edition Beta"
        assert s.base_set_size == 302

    def test_camel_case_aliases(self):
        s = MtgjsonSet.model_validate(MINIMAL_SET)
        assert s.is_foil_only is False
        assert s.is_online_only is False
        assert s.keyrune_code == "LEB"
        assert s.total_set_size == 302

    def test_release_date_coercion(self):
        s = MtgjsonSet.model_validate(MINIMAL_SET)
        assert isinstance(s.release_date, date)
        assert s.release_date == date(1993, 10, 4)

    def test_optional_fields_default_none(self):
        s = MtgjsonSet.model_validate(MINIMAL_SET)
        assert s.block is None
        assert s.mcm_id is None
        assert s.mtgo_code is None

    def test_missing_required_field_fails(self):
        incomplete = {k: v for k, v in MINIMAL_SET.items() if k != "code"}
        with pytest.raises(ValidationError):
            MtgjsonSet.model_validate(incomplete)


# ---------------------------------------------------------------------------
# MtgjsonCard
# ---------------------------------------------------------------------------


class TestMtgjsonCard:
    def test_valid_minimal(self):
        card = MtgjsonCard.model_validate(MINIMAL_CARD)
        assert card.name == "Lightning Bolt"
        assert card.rarity == "common"

    def test_uuid_coercion(self):
        card = MtgjsonCard.model_validate(MINIMAL_CARD)
        assert isinstance(card.uuid, UUID)
        assert str(card.uuid) == SAMPLE_UUID

    def test_camel_case_aliases(self):
        card = MtgjsonCard.model_validate(MINIMAL_CARD)
        assert card.set_code == "LEB"
        assert card.mana_value == 1.0
        assert card.color_identity == ["R"]
        assert card.border_color == "black"
        assert card.frame_version == "2015"

    def test_optional_fields_default_none(self):
        card = MtgjsonCard.model_validate(MINIMAL_CARD)
        assert card.mana_cost is None
        assert card.text is None
        assert card.power is None
        assert card.toughness is None
        assert card.flavor_text is None
        assert card.keywords is None
        assert card.rulings is None
        assert card.edhrec_rank is None

    def test_extra_fields_ignored(self):
        data = {**MINIMAL_CARD, "unknownFutureField": "someValue"}
        card = MtgjsonCard.model_validate(data)
        assert not hasattr(card, "unknownFutureField")

    def test_nested_identifiers(self):
        data = {
            **MINIMAL_CARD,
            "identifiers": {"scryfallId": SAMPLE_UUID, "tcgplayerProductId": "99"},
        }
        card = MtgjsonCard.model_validate(data)
        assert isinstance(card.identifiers.scryfall_id, UUID)
        assert card.identifiers.tcgplayer_product_id == "99"

    def test_nested_rulings(self):
        data = {
            **MINIMAL_CARD,
            "rulings": [{"date": "2004-10-04", "text": "Does not target."}],
        }
        card = MtgjsonCard.model_validate(data)
        assert card.rulings is not None
        assert len(card.rulings) == 1
        assert card.rulings[0].text == "Does not target."
        assert isinstance(card.rulings[0].date, date)

    def test_missing_required_field_fails(self):
        incomplete = {k: v for k, v in MINIMAL_CARD.items() if k != "name"}
        with pytest.raises(ValidationError):
            MtgjsonCard.model_validate(incomplete)

    def test_legalities_dict(self):
        card = MtgjsonCard.model_validate(MINIMAL_CARD)
        assert card.legalities["commander"] == "Legal"
