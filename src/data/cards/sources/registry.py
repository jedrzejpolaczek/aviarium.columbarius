"""Source registry and JSON file I/O — schema lookup, serialisation, and validation."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from src.data.cards.sources.errors import SourceLoadError
from src.data.cards.sources.extractors import (
    extract_mtgjson_cards,
    extract_mtgjson_prices,
)
from src.data.dataclasses.format_staples import FormatStaple
from src.data.dataclasses.mtgjson import MtgjsonCard, MtgjsonCardPrices
from src.data.dataclasses.scryfall import ScryfallCard
from src.data.dataclasses.tournament import TournamentResult
from src.logger import get_logger

T = TypeVar("T", bound=BaseModel)
logger = get_logger(__name__)


# Dictionary to differentiate ways of handling incoming JSON/HTML files.
SOURCE_REGISTRY: dict[str, tuple[type[BaseModel], Callable[[Any], list[Any]]]] = {
    "scryfall": (ScryfallCard, lambda raw: raw),
    "mtgjson_cards": (MtgjsonCard, extract_mtgjson_cards),
    "mtgjson_prices": (MtgjsonCardPrices, extract_mtgjson_prices),
    "format_staples": (FormatStaple, lambda _: []),
    # extractor unused — ingesting_pipeline handles the multi-level scrape directly
    "tournament_results": (TournamentResult, lambda _: []),
}


def _save_to_json(records: list[dict[str, Any]], path: str) -> None:
    """Write a list of dicts to a JSON file, overwriting any existing content.

    Parent directories are created if they do not exist.

    Args:
        records: Raw record dicts to serialise.
        path:    Destination file path.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        with out.open("w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, default=str)
    except OSError as e:
        raise SourceLoadError(f"Failed to write {path}: {e}") from e
    logger.info("Saved %d records to %s", len(records), path)


def load_from_json(
    input_file_path: str,
    model: type[T],
    extractor: Callable[[Any], list[Any]] = lambda raw: raw,
) -> tuple[list[T], list[dict[str, Any]]]:
    """Load and validate records from a JSON file into Pydantic models.

    Args:
        input_file_path: Path to the local JSON file to load.
        model: Pydantic model class to validate each record against.
        extractor: Callable that transforms the raw parsed JSON into a flat
            list of records before validation. Defaults to identity, which
            expects the file to already contain a flat list at the top level.

    Returns:
        A tuple of (records, errors) where records is a list of validated
        model instances and errors is a list of dicts with 'name' and 'error'
        keys for each record that failed validation.
    """
    logger.progress("Loading %s from %s", model.__name__, input_file_path)

    try:
        with open(Path(input_file_path), encoding="utf-8") as f:
            raw = json.load(f)
    except FileNotFoundError:
        raise SourceLoadError(f"File not found: {input_file_path}") from None
    except json.JSONDecodeError as e:
        raise SourceLoadError(f"Invalid JSON in {input_file_path}: {e}") from e

    records, errors = [], []
    for entry in extractor(raw):
        try:
            records.append(model.model_validate(entry))
        except ValidationError as e:
            errors.append({"name": entry.get("name"), "error": e})

    if errors:
        logger.warning(
            "Loaded %d records, %d failed validation", len(records), len(errors)
        )
        for err in errors[:5]:
            logger.debug("Validation failure — name=%r: %s", err["name"], err["error"])
    else:
        logger.info("Loaded %d records, 0 failed validation", len(records))
    return records, errors
