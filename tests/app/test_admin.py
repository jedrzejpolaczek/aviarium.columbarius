"""Unit tests for app/routers/admin.py."""

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routers import admin


@pytest.fixture
def app_with_admin_router(monkeypatch):
    app = FastAPI()
    app.include_router(admin.router)
    app.state.model = None
    app.state.model_run_id = ""
    return app


def test_reload_model_returns_503_when_admin_token_not_configured(
    app_with_admin_router, monkeypatch
):
    monkeypatch.delenv("ADMIN_TOKEN", raising=False)
    client = TestClient(app_with_admin_router)

    response = client.post(
        "/admin/reload-model",
        json={"model_run_id": "abc123"},
        headers={"X-Admin-Token": "whatever"},
    )

    assert response.status_code == 503


def test_reload_model_returns_403_when_token_wrong(app_with_admin_router, monkeypatch):
    monkeypatch.setenv("ADMIN_TOKEN", "correct-token")
    client = TestClient(app_with_admin_router)

    response = client.post(
        "/admin/reload-model",
        json={"model_run_id": "abc123"},
        headers={"X-Admin-Token": "wrong-token"},
    )

    assert response.status_code == 403


def test_reload_model_swaps_app_state_on_success(app_with_admin_router, monkeypatch):
    monkeypatch.setenv("ADMIN_TOKEN", "correct-token")
    mock_model = MagicMock()
    monkeypatch.setattr(admin, "load_model_from_mlflow", lambda run_id: mock_model)
    client = TestClient(app_with_admin_router)

    response = client.post(
        "/admin/reload-model",
        json={"model_run_id": "new-run-456"},
        headers={"X-Admin-Token": "correct-token"},
    )

    assert response.status_code == 200
    assert response.json() == {"status": "reloaded", "model_run_id": "new-run-456"}
    assert app_with_admin_router.state.model is mock_model
    assert app_with_admin_router.state.model_run_id == "new-run-456"


def test_reload_model_returns_502_when_mlflow_load_fails(
    app_with_admin_router, monkeypatch
):
    monkeypatch.setenv("ADMIN_TOKEN", "correct-token")

    def _raise(run_id):
        raise RuntimeError("run not found in registry")

    monkeypatch.setattr(admin, "load_model_from_mlflow", _raise)
    client = TestClient(app_with_admin_router)

    response = client.post(
        "/admin/reload-model",
        json={"model_run_id": "missing-run"},
        headers={"X-Admin-Token": "correct-token"},
    )

    assert response.status_code == 502
    assert app_with_admin_router.state.model is None  # unchanged on failure


def test_reload_model_returns_422_when_model_run_id_missing(
    app_with_admin_router, monkeypatch
):
    monkeypatch.setenv("ADMIN_TOKEN", "correct-token")
    client = TestClient(app_with_admin_router)

    response = client.post(
        "/admin/reload-model",
        json={},
        headers={"X-Admin-Token": "correct-token"},
    )

    assert response.status_code == 422
