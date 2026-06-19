"""Unit tests for GET /similar/{card_name} endpoint.

The mock similarity index (from conftest.py) always returns:
    [{"name": "Dark Ritual", ...}, {"name": "Black Lotus", ...}]
regardless of which card is queried.
"""

from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# HTTP status codes
# ---------------------------------------------------------------------------


def test_similar_known_card_returns_200(test_client: TestClient) -> None:
    response = test_client.get("/similar/Lightning%20Bolt")
    assert response.status_code == 200


def test_similar_unknown_card_returns_404(test_client: TestClient) -> None:
    response = test_client.get("/similar/Nonexistent%20Card")
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# Response body structure
# ---------------------------------------------------------------------------


def test_similar_has_card_name(test_client: TestClient) -> None:
    data = test_client.get("/similar/Lightning%20Bolt").json()
    assert "card_name" in data


def test_similar_has_similar_cards(test_client: TestClient) -> None:
    data = test_client.get("/similar/Lightning%20Bolt").json()
    assert "similar_cards" in data


def test_similar_similar_cards_is_list(test_client: TestClient) -> None:
    data = test_client.get("/similar/Lightning%20Bolt").json()
    assert isinstance(data["similar_cards"], list)


def test_similar_each_card_has_name(test_client: TestClient) -> None:
    data = test_client.get("/similar/Lightning%20Bolt").json()
    for card in data["similar_cards"]:
        assert "name" in card


def test_similar_each_card_has_uuid(test_client: TestClient) -> None:
    data = test_client.get("/similar/Lightning%20Bolt").json()
    for card in data["similar_cards"]:
        assert "uuid" in card


def test_similar_each_card_has_similarity_score(test_client: TestClient) -> None:
    data = test_client.get("/similar/Lightning%20Bolt").json()
    for card in data["similar_cards"]:
        assert "similarity_score" in card


def test_similar_each_card_has_current_price_key(test_client: TestClient) -> None:
    data = test_client.get("/similar/Lightning%20Bolt").json()
    for card in data["similar_cards"]:
        assert "current_price" in card


# ---------------------------------------------------------------------------
# Value correctness
# ---------------------------------------------------------------------------


def test_similar_card_name_matches_request(test_client: TestClient) -> None:
    data = test_client.get("/similar/Lightning%20Bolt").json()
    assert data["card_name"] == "Lightning Bolt"


def test_similar_returns_two_cards_by_default(test_client: TestClient) -> None:
    # Mock returns exactly 2 cards; default n=10 shows all of them
    data = test_client.get("/similar/Lightning%20Bolt").json()
    assert len(data["similar_cards"]) == 2


def test_similar_first_result_name_correct(test_client: TestClient) -> None:
    data = test_client.get("/similar/Lightning%20Bolt").json()
    assert data["similar_cards"][0]["name"] == "Dark Ritual"


def test_similar_similarity_scores_are_float(test_client: TestClient) -> None:
    data = test_client.get("/similar/Lightning%20Bolt").json()
    for card in data["similar_cards"]:
        assert isinstance(card["similarity_score"], float)


def test_similar_similarity_scores_between_0_and_1(test_client: TestClient) -> None:
    data = test_client.get("/similar/Lightning%20Bolt").json()
    for card in data["similar_cards"]:
        assert 0.0 <= card["similarity_score"] <= 1.0


# ---------------------------------------------------------------------------
# n parameter
# ---------------------------------------------------------------------------


def test_similar_n_parameter_limits_results(test_client: TestClient) -> None:
    data = test_client.get("/similar/Lightning%20Bolt?n=1").json()
    assert len(data["similar_cards"]) == 1


def test_similar_n_parameter_returns_all_when_n_exceeds_available(
    test_client: TestClient,
) -> None:
    # Mock returns 2 cards; n=50 should return all 2
    data = test_client.get("/similar/Lightning%20Bolt?n=50").json()
    assert len(data["similar_cards"]) == 2


def test_similar_n_zero_returns_422(test_client: TestClient) -> None:
    response = test_client.get("/similar/Lightning%20Bolt?n=0")
    assert response.status_code == 422


def test_similar_n_above_50_returns_422(test_client: TestClient) -> None:
    response = test_client.get("/similar/Lightning%20Bolt?n=51")
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# 503 — no similarity index
# ---------------------------------------------------------------------------


def test_similar_no_index_returns_503(test_client_no_similarity: TestClient) -> None:
    response = test_client_no_similarity.get("/similar/Lightning%20Bolt")
    assert response.status_code == 503


def test_similar_no_index_detail_message(test_client_no_similarity: TestClient) -> None:
    data = test_client_no_similarity.get("/similar/Lightning%20Bolt").json()
    assert "detail" in data


# ---------------------------------------------------------------------------
# 404 — find_similar raises ValueError
# ---------------------------------------------------------------------------


def test_similar_value_error_returns_404(
    test_client_similarity_error: TestClient,
) -> None:
    response = test_client_similarity_error.get("/similar/Lightning%20Bolt")
    assert response.status_code == 404


def test_similar_value_error_detail_is_string(
    test_client_similarity_error: TestClient,
) -> None:
    data = test_client_similarity_error.get("/similar/Lightning%20Bolt").json()
    assert isinstance(data["detail"], str)
