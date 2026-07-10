"""Unit tests for app/main.py's global exception handler."""

from unittest.mock import MagicMock

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from app.main import register_exception_handlers


def _build_app_with_broken_route() -> FastAPI:
    app = FastAPI()
    register_exception_handlers(app)

    @app.get("/boom")
    def boom() -> None:
        raise ValueError("unexpected internal failure")

    @app.get("/not-found")
    def not_found() -> None:
        raise HTTPException(404, detail="Card 'X' not found.")

    return app


def test_unhandled_exception_returns_500_with_structured_body(monkeypatch):
    monkeypatch.setattr("app.main.send_alert", MagicMock())
    app = _build_app_with_broken_route()

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/boom")

    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error."}


def test_unhandled_exception_sends_an_alert(monkeypatch):
    mock_send_alert = MagicMock()
    monkeypatch.setattr("app.main.send_alert", mock_send_alert)
    app = _build_app_with_broken_route()

    with TestClient(app, raise_server_exceptions=False) as client:
        client.get("/boom")

    mock_send_alert.assert_called_once()
    subject, message = mock_send_alert.call_args.args
    assert subject == "Unhandled API exception"
    assert "unexpected internal failure" in message
    assert "/boom" in message


def test_existing_http_exceptions_are_not_affected_by_the_handler(monkeypatch):
    monkeypatch.setattr("app.main.send_alert", MagicMock())
    app = _build_app_with_broken_route()

    with TestClient(app) as client:
        response = client.get("/not-found")

    assert response.status_code == 404
    assert response.json() == {"detail": "Card 'X' not found."}
