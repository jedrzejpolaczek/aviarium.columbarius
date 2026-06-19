"""Unit tests for GET /predict/{card_name} endpoint.

Test data (from conftest.py):
    "Lightning Bolt"  eur=1.5   log_return=0.5  → Tier 1, predicted ≈ 3.12
    "Dark Ritual"     eur=0.3   log_return=0.5  → Tier 1, predicted ≈ 0.62
    "Black Lotus"     eur=1500  log_return=0.5  → Tier 3, predicted_price=null
"""

import math

from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# HTTP status codes
# ---------------------------------------------------------------------------


def test_predict_known_card_returns_200(test_client: TestClient) -> None:
    response = test_client.get("/predict/Lightning%20Bolt")
    assert response.status_code == 200


def test_predict_unknown_card_returns_404(test_client: TestClient) -> None:
    response = test_client.get("/predict/Nonexistent%20Card")
    assert response.status_code == 404


def test_predict_no_model_returns_503(test_client_no_model: TestClient) -> None:
    response = test_client_no_model.get("/predict/Lightning%20Bolt")
    assert response.status_code == 503


# ---------------------------------------------------------------------------
# Response body structure
# ---------------------------------------------------------------------------


def test_predict_has_card_name(test_client: TestClient) -> None:
    data = test_client.get("/predict/Lightning%20Bolt").json()
    assert "card_name" in data


def test_predict_has_current_price(test_client: TestClient) -> None:
    data = test_client.get("/predict/Lightning%20Bolt").json()
    assert "current_price" in data


def test_predict_has_predicted_price(test_client: TestClient) -> None:
    data = test_client.get("/predict/Lightning%20Bolt").json()
    assert "predicted_price" in data


def test_predict_has_log_return_7d(test_client: TestClient) -> None:
    data = test_client.get("/predict/Lightning%20Bolt").json()
    assert "log_return_7d" in data


def test_predict_has_tier(test_client: TestClient) -> None:
    data = test_client.get("/predict/Lightning%20Bolt").json()
    assert "tier" in data


def test_predict_has_model_run_id(test_client: TestClient) -> None:
    data = test_client.get("/predict/Lightning%20Bolt").json()
    assert "model_run_id" in data


# ---------------------------------------------------------------------------
# Value correctness — Tier 1 card ("Lightning Bolt", eur=1.5)
# ---------------------------------------------------------------------------


def test_predict_card_name_matches_request(test_client: TestClient) -> None:
    data = test_client.get("/predict/Lightning%20Bolt").json()
    assert data["card_name"] == "Lightning Bolt"


def test_predict_current_price_correct(test_client: TestClient) -> None:
    data = test_client.get("/predict/Lightning%20Bolt").json()
    assert abs(data["current_price"] - 1.5) < 1e-6


def test_predict_tier1_assignment(test_client: TestClient) -> None:
    data = test_client.get("/predict/Lightning%20Bolt").json()
    assert data["tier"] == 1


def test_predict_predicted_price_is_float_for_tier1(test_client: TestClient) -> None:
    data = test_client.get("/predict/Lightning%20Bolt").json()
    assert isinstance(data["predicted_price"], float)


def test_predict_predicted_price_positive(test_client: TestClient) -> None:
    data = test_client.get("/predict/Lightning%20Bolt").json()
    assert data["predicted_price"] > 0


def test_predict_log_return_matches_mock(test_client: TestClient) -> None:
    # Mock always returns _LOG_RETURNS[0] = 0.5
    data = test_client.get("/predict/Lightning%20Bolt").json()
    assert abs(data["log_return_7d"] - 0.5) < 1e-6


def test_predict_predicted_price_formula_correct(test_client: TestClient) -> None:
    # predicted = expm1(log1p(1.5) + 0.5)
    expected = math.expm1(math.log1p(1.5) + 0.5)
    data = test_client.get("/predict/Lightning%20Bolt").json()
    assert abs(data["predicted_price"] - expected) < 1e-4


def test_predict_model_run_id_matches_fixture(test_client: TestClient) -> None:
    data = test_client.get("/predict/Lightning%20Bolt").json()
    assert data["model_run_id"] == "test-run-123"


# ---------------------------------------------------------------------------
# Tier 3 card ("Black Lotus", eur=1500)
# ---------------------------------------------------------------------------


def test_predict_tier3_returns_200(test_client: TestClient) -> None:
    response = test_client.get("/predict/Black%20Lotus")
    assert response.status_code == 200


def test_predict_tier3_assignment(test_client: TestClient) -> None:
    data = test_client.get("/predict/Black%20Lotus").json()
    assert data["tier"] == 3


def test_predict_tier3_predicted_price_is_null(test_client: TestClient) -> None:
    data = test_client.get("/predict/Black%20Lotus").json()
    assert data["predicted_price"] is None


def test_predict_tier3_log_return_is_null(test_client: TestClient) -> None:
    data = test_client.get("/predict/Black%20Lotus").json()
    assert data["log_return_7d"] is None


def test_predict_tier3_current_price_present(test_client: TestClient) -> None:
    data = test_client.get("/predict/Black%20Lotus").json()
    assert abs(data["current_price"] - 1500.0) < 1e-6


# ---------------------------------------------------------------------------
# UUID endpoint — GET /predict/uuid/{uuid}
# ---------------------------------------------------------------------------


def test_predict_by_uuid_known_returns_200(test_client: TestClient) -> None:
    response = test_client.get("/predict/uuid/uuid_lb")
    assert response.status_code == 200


def test_predict_by_uuid_unknown_returns_404(test_client: TestClient) -> None:
    response = test_client.get("/predict/uuid/no_such_uuid")
    assert response.status_code == 404


def test_predict_by_uuid_no_model_returns_503(test_client_no_model: TestClient) -> None:
    response = test_client_no_model.get("/predict/uuid/uuid_lb")
    assert response.status_code == 503


def test_predict_by_uuid_returns_correct_card_name(test_client: TestClient) -> None:
    data = test_client.get("/predict/uuid/uuid_lb").json()
    assert data["card_name"] == "Lightning Bolt"


def test_predict_by_uuid_returns_correct_price(test_client: TestClient) -> None:
    data = test_client.get("/predict/uuid/uuid_lb").json()
    assert abs(data["current_price"] - 1.5) < 1e-6


def test_predict_by_uuid_tier3_predicted_price_null(test_client: TestClient) -> None:
    data = test_client.get("/predict/uuid/uuid_bl").json()
    assert data["predicted_price"] is None


def test_predict_by_uuid_response_has_model_run_id(test_client: TestClient) -> None:
    data = test_client.get("/predict/uuid/uuid_lb").json()
    assert data["model_run_id"] == "test-run-123"
