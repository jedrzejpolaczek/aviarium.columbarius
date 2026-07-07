"""Unit tests for src/data/cards/sources/registry.py."""

import json

from pydantic import BaseModel

from src.data.cards.sources.registry import (
    SOURCE_REGISTRY,
    _save_to_json,
    load_from_json,
)
from src.data.dataclasses.format_staples import FormatStaple
from src.data.dataclasses.mtgjson import MtgjsonCard, MtgjsonCardPrices
from src.data.dataclasses.scryfall import ScryfallCard
from src.data.dataclasses.tournament import TournamentResult


class TestSourceRegistry:
    def test_contains_expected_source_types(self):
        assert set(SOURCE_REGISTRY) == {
            "scryfall",
            "mtgjson_cards",
            "mtgjson_prices",
            "format_staples",
            "tournament_results",
        }

    def test_entries_are_model_extractor_pairs(self):
        for model, extractor in SOURCE_REGISTRY.values():
            assert isinstance(model, type) and issubclass(model, BaseModel)
            assert callable(extractor)

    def test_scryfall_maps_to_scryfall_card_with_identity_extractor(self):
        model, extractor = SOURCE_REGISTRY["scryfall"]
        assert model is ScryfallCard
        raw = [{"id": "1"}]
        assert extractor(raw) == raw

    def test_mtgjson_cards_maps_to_mtgjson_card_model(self):
        model, _ = SOURCE_REGISTRY["mtgjson_cards"]
        assert model is MtgjsonCard

    def test_mtgjson_prices_maps_to_mtgjson_card_prices_model(self):
        model, _ = SOURCE_REGISTRY["mtgjson_prices"]
        assert model is MtgjsonCardPrices

    def test_format_staples_maps_to_format_staple_with_noop_extractor(self):
        model, extractor = SOURCE_REGISTRY["format_staples"]
        assert model is FormatStaple
        assert extractor({"anything": "goes"}) == []

    def test_tournament_results_maps_to_tournament_result_with_noop_extractor(self):
        model, extractor = SOURCE_REGISTRY["tournament_results"]
        assert model is TournamentResult
        assert extractor({"anything": "goes"}) == []

    def test_unknown_source_type_not_registered(self):
        # registry.py itself is a plain dict lookup — it does not raise on a
        # missing key. Callers (see scrapers.py:50-51) are responsible for
        # raising SourceNotRegisteredError when a key is absent.
        assert "not_a_real_source" not in SOURCE_REGISTRY


# Broader load_from_json/_save_to_json behaviour (valid-file loads, invalid
# records going to errors instead of raising, custom extractors, missing-file
# and invalid-JSON error paths, parent-directory creation, overwrite-in-place)
# is already covered in tests/data/cards/sources/test_pipeline.py's
# TestSaveToJson/TestLoadFromJson. Only scenarios not already exercised there
# are added here, to avoid two files needing an update for one behaviour change.
class TestSaveToJson:
    def test_non_json_serialisable_values_are_stringified(self, tmp_path):
        out = tmp_path / "out.json"

        _save_to_json([{"created": object()}], str(out))

        loaded = json.loads(out.read_text(encoding="utf-8"))
        assert isinstance(loaded[0]["created"], str)


class TestLoadFromJson:
    def test_empty_list_returns_no_records_and_no_errors(self, tmp_path):
        input_file = tmp_path / "empty.json"
        input_file.write_text("[]", encoding="utf-8")

        records, errors = load_from_json(str(input_file), FormatStaple)

        assert records == []
        assert errors == []
