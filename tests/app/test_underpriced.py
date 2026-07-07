"""Unit tests for GET /underpriced endpoint.

Test data (from conftest.py):
    "Lightning Bolt"  eur=1.5    log_return=0.5  predicted≈3.12  confidence≈2.08 → underpriced
    "Dark Ritual"     eur=0.3    log_return=-0.1 predicted≈0.27  confidence≈0.90 → not underpriced
    "Black Lotus"     eur=1500.0 log_return=0.0  Tier 3 → never flagged

Expected result: only "Lightning Bolt" is returned (Tier 1, confidence > 1.3).
"""

import math

import pandas as pd
from fastapi.testclient import TestClient

from app.routers.underpriced import _to_underpriced_cards


# ---------------------------------------------------------------------------
# HTTP status codes
# ---------------------------------------------------------------------------


def test_underpriced_returns_200(test_client: TestClient) -> None:
    response = test_client.get("/underpriced/")
    assert response.status_code == 200


def test_underpriced_no_model_returns_503(test_client_no_model: TestClient) -> None:
    response = test_client_no_model.get("/underpriced/")
    assert response.status_code == 503


# ---------------------------------------------------------------------------
# Response body structure
# ---------------------------------------------------------------------------


def test_underpriced_has_cards_key(test_client: TestClient) -> None:
    data = test_client.get("/underpriced/").json()
    assert "cards" in data


def test_underpriced_has_generated_at(test_client: TestClient) -> None:
    data = test_client.get("/underpriced/").json()
    assert "generated_at" in data


def test_underpriced_has_model_run_id(test_client: TestClient) -> None:
    data = test_client.get("/underpriced/").json()
    assert "model_run_id" in data


def test_underpriced_cards_is_list(test_client: TestClient) -> None:
    data = test_client.get("/underpriced/").json()
    assert isinstance(data["cards"], list)


def test_underpriced_each_card_has_name(test_client: TestClient) -> None:
    data = test_client.get("/underpriced/").json()
    for card in data["cards"]:
        assert "name" in card


def test_underpriced_each_card_has_uuid(test_client: TestClient) -> None:
    data = test_client.get("/underpriced/").json()
    for card in data["cards"]:
        assert "uuid" in card


def test_underpriced_each_card_has_actual_price(test_client: TestClient) -> None:
    data = test_client.get("/underpriced/").json()
    for card in data["cards"]:
        assert "actual_price" in card


def test_underpriced_each_card_has_predicted_price(test_client: TestClient) -> None:
    data = test_client.get("/underpriced/").json()
    for card in data["cards"]:
        assert "predicted_price" in card


def test_underpriced_each_card_has_confidence(test_client: TestClient) -> None:
    data = test_client.get("/underpriced/").json()
    for card in data["cards"]:
        assert "confidence" in card


def test_underpriced_each_card_has_tier(test_client: TestClient) -> None:
    data = test_client.get("/underpriced/").json()
    for card in data["cards"]:
        assert "tier" in card


def test_underpriced_each_card_has_reason(test_client: TestClient) -> None:
    data = test_client.get("/underpriced/").json()
    for card in data["cards"]:
        assert "reason" in card


# ---------------------------------------------------------------------------
# Value correctness
# ---------------------------------------------------------------------------


def test_underpriced_returns_only_lightning_bolt(test_client: TestClient) -> None:
    # Only Lightning Bolt has confidence > 1.3 in the test fixture
    data = test_client.get("/underpriced/").json()
    names = [c["name"] for c in data["cards"]]
    assert names == ["Lightning Bolt"]


def test_underpriced_lightning_bolt_actual_price(test_client: TestClient) -> None:
    data = test_client.get("/underpriced/").json()
    assert abs(data["cards"][0]["actual_price"] - 1.5) < 1e-6


def test_underpriced_lightning_bolt_predicted_price_formula(
    test_client: TestClient,
) -> None:
    # predicted = expm1(log1p(1.5) + 0.5)
    expected = math.expm1(math.log1p(1.5) + 0.5)
    data = test_client.get("/underpriced/").json()
    assert abs(data["cards"][0]["predicted_price"] - expected) < 1e-4


def test_underpriced_lightning_bolt_tier_is_1(test_client: TestClient) -> None:
    data = test_client.get("/underpriced/").json()
    assert data["cards"][0]["tier"] == 1


def test_underpriced_lightning_bolt_confidence_above_threshold(
    test_client: TestClient,
) -> None:
    data = test_client.get("/underpriced/").json()
    assert data["cards"][0]["confidence"] > 1.3


def test_underpriced_reason_contains_ml_predicts(test_client: TestClient) -> None:
    data = test_client.get("/underpriced/").json()
    assert "ML predicts" in data["cards"][0]["reason"]


def test_underpriced_model_run_id_matches_fixture(test_client: TestClient) -> None:
    data = test_client.get("/underpriced/").json()
    assert data["model_run_id"] == "test-run-123"


def test_underpriced_dark_ritual_not_in_results(test_client: TestClient) -> None:
    data = test_client.get("/underpriced/").json()
    names = [c["name"] for c in data["cards"]]
    assert "Dark Ritual" not in names


def test_underpriced_black_lotus_not_in_results(test_client: TestClient) -> None:
    # Tier 3 is never flagged
    data = test_client.get("/underpriced/").json()
    names = [c["name"] for c in data["cards"]]
    assert "Black Lotus" not in names


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------


def test_underpriced_tier_filter_tier1_returns_results(test_client: TestClient) -> None:
    data = test_client.get("/underpriced/?tier=1").json()
    assert len(data["cards"]) >= 1


def test_underpriced_tier_filter_tier2_returns_empty(test_client: TestClient) -> None:
    # No Tier 2 cards in test fixture
    data = test_client.get("/underpriced/?tier=2").json()
    assert data["cards"] == []


def test_underpriced_tier_filter_tier3_returns_empty(test_client: TestClient) -> None:
    # Tier 3 never flagged
    data = test_client.get("/underpriced/?tier=3").json()
    assert data["cards"] == []


def test_underpriced_high_min_confidence_filters_out_all(
    test_client: TestClient,
) -> None:
    # Lightning Bolt confidence ≈ 2.08; requiring 3.0 excludes it
    data = test_client.get("/underpriced/?min_confidence=3.0").json()
    assert data["cards"] == []


def test_underpriced_low_min_confidence_includes_lightning_bolt(
    test_client: TestClient,
) -> None:
    data = test_client.get("/underpriced/?min_confidence=1.1").json()
    names = [c["name"] for c in data["cards"]]
    assert "Lightning Bolt" in names


def test_underpriced_min_confidence_below_1_returns_422(
    test_client: TestClient,
) -> None:
    response = test_client.get("/underpriced/?min_confidence=0.5")
    assert response.status_code == 422


def test_underpriced_invalid_tier_returns_422(test_client: TestClient) -> None:
    response = test_client.get("/underpriced/?tier=4")
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# _to_underpriced_cards (direct unit test, no TestClient/HTTP layer)
# ---------------------------------------------------------------------------


def test_to_underpriced_cards_maps_fields_correctly() -> None:
    flagged = pd.DataFrame(
        {
            "name": ["Lightning Bolt"],
            "uuid": ["abc-123"],
            "eur": [1.5],
            "predicted_eur": [3.12],
            "confidence": [2.08],
            "tier": [1],
            "reason": ["ML predicts a 108% increase"],
        }
    )

    cards = _to_underpriced_cards(flagged)

    assert len(cards) == 1
    card = cards[0]
    assert card.name == "Lightning Bolt"
    assert card.uuid == "abc-123"
    assert card.actual_price == 1.5
    assert card.predicted_price == 3.12
    assert card.confidence == 2.08
    assert card.tier == 1
    assert card.reason == "ML predicts a 108% increase"
