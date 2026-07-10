"""Unit tests for GET /health endpoint."""

from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Response structure
# ---------------------------------------------------------------------------


def test_health_returns_200(test_client: TestClient) -> None:
    response = test_client.get("/health")
    assert response.status_code == 200


def test_health_returns_json(test_client: TestClient) -> None:
    response = test_client.get("/health")
    assert response.headers["content-type"].startswith("application/json")


def test_health_has_status_key(test_client: TestClient) -> None:
    data = test_client.get("/health").json()
    assert "status" in data


def test_health_has_model_loaded_key(test_client: TestClient) -> None:
    data = test_client.get("/health").json()
    assert "model_loaded" in data


def test_health_has_db_connected_key(test_client: TestClient) -> None:
    data = test_client.get("/health").json()
    assert "db_connected" in data


# ---------------------------------------------------------------------------
# Status values — model and DB present (test_client fixture)
# ---------------------------------------------------------------------------


def test_health_status_ok_when_model_and_db_loaded(test_client: TestClient) -> None:
    data = test_client.get("/health").json()
    assert data["status"] == "ok"


def test_health_model_loaded_true(test_client: TestClient) -> None:
    data = test_client.get("/health").json()
    assert data["model_loaded"] is True


def test_health_db_connected_true(test_client: TestClient) -> None:
    data = test_client.get("/health").json()
    assert data["db_connected"] is True


# ---------------------------------------------------------------------------
# Degraded mode — no model (test_client_no_model fixture)
# ---------------------------------------------------------------------------


def test_health_status_degraded_when_no_model(test_client_no_model: TestClient) -> None:
    data = test_client_no_model.get("/health").json()
    assert data["status"] == "degraded"


def test_health_model_loaded_false_when_no_model(
    test_client_no_model: TestClient,
) -> None:
    data = test_client_no_model.get("/health").json()
    assert data["model_loaded"] is False


def test_health_db_connected_true_even_without_model(
    test_client_no_model: TestClient,
) -> None:
    data = test_client_no_model.get("/health").json()
    assert data["db_connected"] is True


def test_health_degraded_returns_503(test_client_no_model: TestClient) -> None:
    response = test_client_no_model.get("/health")
    assert response.status_code == 503


def test_health_features_loaded_true_by_default(test_client: TestClient) -> None:
    data = test_client.get("/health").json()
    assert data["features_loaded"] is True


def test_health_features_loaded_false_when_features_unavailable(
    test_client_no_features: TestClient,
) -> None:
    data = test_client_no_features.get("/health").json()
    assert data["features_loaded"] is False
