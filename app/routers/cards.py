import pandas as pd
from fastapi import APIRouter, Depends

from app.dependencies import RequestFeatures, get_request_features
from app.schemas.responses import CardEntry, CardsResponse

router = APIRouter(prefix="/cards", tags=["cards"])


@router.get("", response_model=CardsResponse)
def list_cards(
    features: RequestFeatures = Depends(get_request_features),
) -> CardsResponse:
    X_all = features.X_all
    if X_all.empty:
        return CardsResponse(cards=[])
    cols = (
        X_all[["uuid", "name", "set_code", "rarity", "eur"]]
        .dropna(subset=["uuid", "name"])
        .drop_duplicates(subset=["uuid"])
        .sort_values(["name", "set_code"])
    )
    entries = [
        CardEntry(
            uuid=str(row.uuid),
            name=str(row.name),
            set_code=str(row.set_code) if pd.notna(row.set_code) else "",
            rarity=str(row.rarity) if pd.notna(row.rarity) else "",
            eur=float(row.eur) if pd.notna(row.eur) else None,  # type: ignore[arg-type]
        )
        for row in cols.itertuples()
    ]
    return CardsResponse(cards=entries)
