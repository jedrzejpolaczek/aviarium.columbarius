"""Unit tests for src/data/cards/sources/registry.py."""

import json

import pytest
from pydantic import BaseModel, ValidationError

from src.data.cards.sources.errors import SourceLoadError
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


class TestSaveToJson:
    def test_writes_records_to_file(self, tmp_path):
        out = tmp_path / "out.json"
        records = [{"name": "Black Lotus"}, {"name": "Mox Pearl"}]

        _save_to_json(records, str(out))

        assert json.loads(out.read_text(encoding="utf-8")) == records

    def test_creates_missing_parent_directories(self, tmp_path):
        out = tmp_path / "nested" / "dir" / "out.json"

        _save_to_json([{"name": "Island"}], str(out))

        assert out.exists()
        assert json.loads(out.read_text(encoding="utf-8")) == [{"name": "Island"}]

    def test_overwrites_existing_content(self, tmp_path):
        out = tmp_path / "out.json"
        out.write_text(json.dumps([{"name": "Old"}]), encoding="utf-8")

        _save_to_json([{"name": "New"}], str(out))

        assert json.loads(out.read_text(encoding="utf-8")) == [{"name": "New"}]

    def test_non_json_serialisable_values_are_stringified(self, tmp_path):
        out = tmp_path / "out.json"

        _save_to_json([{"created": object()}], str(out))

        loaded = json.loads(out.read_text(encoding="utf-8"))
        assert isinstance(loaded[0]["created"], str)

    def test_os_error_raises_source_load_error(self, tmp_path):
        # Writing to a path that is itself a directory cannot succeed —
        # opening it for write raises an OSError subclass on every platform.
        directory_as_path = tmp_path / "a_directory"
        directory_as_path.mkdir()

        with pytest.raises(SourceLoadError, match="Failed to write"):
            _save_to_json([{"name": "X"}], str(directory_as_path))


class TestLoadFromJson:
    def test_loads_valid_records(self, tmp_path):
        input_file = tmp_path / "staples.json"
        raw = [
            {
                "id": "Lightning Bolt__modern",
                "card_name": "Lightning Bolt",
                "format": "modern",
                "deck_pct": 95.5,
                "percentage_in_decks": 95,
                "played": 3.8,
                "top": 1,
            },
            {
                "id": "Thoughtseize__modern",
                "card_name": "Thoughtseize",
                "format": "modern",
                "deck_pct": 80.0,
                "percentage_in_decks": 80,
                "played": 2.1,
                "top": 2,
            },
        ]
        input_file.write_text(json.dumps(raw), encoding="utf-8")

        records, errors = load_from_json(str(input_file), FormatStaple)

        assert errors == []
        assert len(records) == 2
        assert all(isinstance(r, FormatStaple) for r in records)
        assert {r.card_name for r in records} == {"Lightning Bolt", "Thoughtseize"}

    def test_malformed_entries_are_collected_as_errors_not_raised(self, tmp_path):
        input_file = tmp_path / "staples.json"
        raw = [
            {
                "id": "Lightning Bolt__modern",
                "card_name": "Lightning Bolt",
                "format": "modern",
                "deck_pct": 95.5,
                "percentage_in_decks": 95,
                "played": 3.8,
                "top": 1,
            },
            {"name": "Broken Entry"},  # missing every required FormatStaple field
        ]
        input_file.write_text(json.dumps(raw), encoding="utf-8")

        records, errors = load_from_json(str(input_file), FormatStaple)

        assert len(records) == 1
        assert records[0].card_name == "Lightning Bolt"
        assert len(errors) == 1
        assert errors[0]["name"] == "Broken Entry"
        assert isinstance(errors[0]["error"], ValidationError)

    def test_uses_custom_extractor_before_validation(self, tmp_path):
        input_file = tmp_path / "nested.json"
        raw = {"data": {"LEB": {"cards": [{"name": "Black Lotus"}]}}}
        input_file.write_text(json.dumps(raw), encoding="utf-8")

        class NameOnly(BaseModel):
            name: str

        def flatten(parsed):
            return [c for s in parsed["data"].values() for c in s["cards"]]

        records, errors = load_from_json(str(input_file), NameOnly, extractor=flatten)

        assert errors == []
        assert len(records) == 1
        assert records[0].name == "Black Lotus"

    def test_missing_file_raises_source_load_error(self, tmp_path):
        missing = tmp_path / "does_not_exist.json"

        with pytest.raises(SourceLoadError, match="File not found"):
            load_from_json(str(missing), FormatStaple)

    def test_invalid_json_raises_source_load_error(self, tmp_path):
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("{not valid json", encoding="utf-8")

        with pytest.raises(SourceLoadError, match="Invalid JSON"):
            load_from_json(str(bad_file), FormatStaple)

    def test_empty_list_returns_no_records_and_no_errors(self, tmp_path):
        input_file = tmp_path / "empty.json"
        input_file.write_text("[]", encoding="utf-8")

        records, errors = load_from_json(str(input_file), FormatStaple)

        assert records == []
        assert errors == []
