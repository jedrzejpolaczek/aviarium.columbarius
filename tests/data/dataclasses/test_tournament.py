"""Unit tests for src/data/dataclasses/tournament.py Pydantic model."""

import pytest
from pydantic import ValidationError

from src.data.dataclasses.tournament import TournamentResult

SAMPLE_TOURNAMENT_RESULT = {
    "id": "evt-111-d-1",
    "tournament_id": "111",
    "tournament_date": "2026-06-01",
    "format": "modern",
    "event_name": "Modern Open",
    "placement": 1,
    "player": "Player A",
    "deck_name": "Burn",
    "card_name": "Lightning Bolt",
    "copies": 4,
    "is_sideboard": False,
}


class TestTournamentResult:
    def test_valid_record(self):
        rec = TournamentResult(**SAMPLE_TOURNAMENT_RESULT)
        assert rec.id == "evt-111-d-1"
        assert rec.tournament_id == "111"
        assert rec.tournament_date == "2026-06-01"
        assert rec.format == "modern"
        assert rec.event_name == "Modern Open"
        assert rec.placement == 1
        assert rec.player == "Player A"
        assert rec.deck_name == "Burn"
        assert rec.card_name == "Lightning Bolt"
        assert rec.copies == 4
        assert rec.is_sideboard is False

    def test_missing_required_field_fails(self):
        data = {k: v for k, v in SAMPLE_TOURNAMENT_RESULT.items() if k != "player"}
        with pytest.raises(ValidationError):
            TournamentResult(**data)

    def test_wrong_type_for_copies_fails(self):
        data = {**SAMPLE_TOURNAMENT_RESULT, "copies": "four"}
        with pytest.raises(ValidationError):
            TournamentResult(**data)

    def test_wrong_type_for_placement_fails(self):
        data = {**SAMPLE_TOURNAMENT_RESULT, "placement": "first"}
        with pytest.raises(ValidationError):
            TournamentResult(**data)

    def test_wrong_type_for_is_sideboard_fails(self):
        data = {**SAMPLE_TOURNAMENT_RESULT, "is_sideboard": "not-a-bool"}
        with pytest.raises(ValidationError):
            TournamentResult(**data)
