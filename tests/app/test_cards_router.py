import pandas as pd
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routers.cards import router


def _client(rows: list[dict]) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.state.X_all = pd.DataFrame(rows)
    return TestClient(app)


def test_returns_entries_sorted_by_name_then_set_code():
    client = _client(
        [
            {
                "uuid": "c",
                "name": "Lightning Bolt",
                "set_code": "M11",
                "rarity": "common",
                "eur": 1.0,
            },
            {
                "uuid": "a",
                "name": "Black Lotus",
                "set_code": "LEA",
                "rarity": "rare",
                "eur": 5000.0,
            },
            {
                "uuid": "b",
                "name": "Lightning Bolt",
                "set_code": "M10",
                "rarity": "common",
                "eur": 1.5,
            },
        ]
    )
    response = client.get("/cards")
    assert response.status_code == 200
    cards = response.json()["cards"]
    assert [c["uuid"] for c in cards] == ["a", "b", "c"]


def test_deduplicates_by_uuid():
    client = _client(
        [
            {
                "uuid": "x",
                "name": "Force of Will",
                "set_code": "ALL",
                "rarity": "uncommon",
                "eur": 80.0,
            },
            {
                "uuid": "x",
                "name": "Force of Will",
                "set_code": "ALL",
                "rarity": "uncommon",
                "eur": 80.0,
            },
        ]
    )
    response = client.get("/cards")
    assert response.status_code == 200
    assert len(response.json()["cards"]) == 1


def test_null_eur_becomes_none():
    client = _client(
        [
            {
                "uuid": "z",
                "name": "Some Card",
                "set_code": "TST",
                "rarity": "rare",
                "eur": float("nan"),
            },
        ]
    )
    response = client.get("/cards")
    assert response.status_code == 200
    assert response.json()["cards"][0]["eur"] is None


def test_empty_dataset_returns_empty_list():
    client = _client([])
    response = client.get("/cards")
    assert response.status_code == 200
    assert response.json() == {"cards": []}
