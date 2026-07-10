"""Unit tests for src/data/dataclasses/format_staples.py Pydantic model."""

import pytest
from pydantic import ValidationError

from src.data.dataclasses.format_staples import FormatStaple

SAMPLE_FORMAT_STAPLE = {
    "id": "abc-123",
    "card_name": "Lightning Bolt",
    "format": "modern",
    "deck_pct": 12.5,
    "percentage_in_decks": 12,
    "played": 340.0,
    "top": 3,
}


class TestFormatStaple:
    def test_valid_record(self):
        rec = FormatStaple(**SAMPLE_FORMAT_STAPLE)
        assert rec.id == "abc-123"
        assert rec.card_name == "Lightning Bolt"
        assert rec.format == "modern"
        assert rec.deck_pct == 12.5
        assert rec.percentage_in_decks == 12
        assert rec.played == 340.0
        assert rec.top == 3

    def test_missing_required_field_fails(self):
        data = {k: v for k, v in SAMPLE_FORMAT_STAPLE.items() if k != "card_name"}
        with pytest.raises(ValidationError):
            FormatStaple(**data)

    def test_wrong_type_for_deck_pct_fails(self):
        data = {**SAMPLE_FORMAT_STAPLE, "deck_pct": "not-a-number"}
        with pytest.raises(ValidationError):
            FormatStaple(**data)

    def test_wrong_type_for_percentage_in_decks_fails(self):
        data = {**SAMPLE_FORMAT_STAPLE, "percentage_in_decks": "not-an-int"}
        with pytest.raises(ValidationError):
            FormatStaple(**data)

    def test_wrong_type_for_top_fails(self):
        data = {**SAMPLE_FORMAT_STAPLE, "top": "not-an-int"}
        with pytest.raises(ValidationError):
            FormatStaple(**data)
