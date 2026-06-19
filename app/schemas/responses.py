"""
Schematy Pydantic definiujace struktury responsow API.

CO ROBIĆ:
Te klasy sa juz gotowe — nie trzeba nic implementowac.
FastAPI uzywa ich do:
- Walidacji danych wyjsciowych (sprawdza typy)
- Generowania Swagger UI (/docs)
- Automatycznej dokumentacji OpenAPI

TEN SAM WZORZEC co src/data/dataclasses/ (ScryfallCard itp.):
Pydantic BaseModel z typami i opcjonalnymi polami.

predicted_price: float | None:
    None dla Tier 3 (> 1000 EUR) — brak predykcji ML, zbyt malo danych.
"""

from datetime import date
from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    card_name: str
    current_price: float | None = Field(description="Aktualna cena w EUR")
    predicted_price: float | None = Field(
        description="Przewidywana cena za 7 dni. None dla Tier 3."
    )
    log_return_7d: float | None = Field(
        description="Przewidywana zmiana log1p w ciagu 7 dni"
    )
    tier: int = Field(
        description="Tier cenowy: 1 (<100 EUR), 2 (100-1000 EUR), 3 (>1000 EUR)"
    )
    model_run_id: str = Field(description="MLflow run_id modelu ktory dal te predykcje")


class SimilarCard(BaseModel):
    name: str
    uuid: str
    current_price: float | None
    similarity_score: float = Field(
        description="Wspolczynnik podobienstwa (1.0=identyczna, 0.0=brak)"
    )


class SimilarCardsResponse(BaseModel):
    card_name: str
    similar_cards: list[SimilarCard]


class UnderpricedCard(BaseModel):
    name: str
    uuid: str
    actual_price: float
    predicted_price: float
    confidence: float = Field(description="Stosunek predicted/actual (np. 1.44 = +44%)")
    tier: int
    reason: str


class UnderpricedResponse(BaseModel):
    cards: list[UnderpricedCard]
    generated_at: date
    model_run_id: str


class CardEntry(BaseModel):
    uuid: str
    name: str
    set_code: str
    rarity: str
    eur: float | None


class CardsResponse(BaseModel):
    cards: list[CardEntry]
