"""Shared JSON-file-loading primitive.

Used by src/data/cards/pipelines.py, src/data/cards/sources/registry.py, and
src/data/cards/storage/silver/storage.py, which each independently
implemented the same open+parse+translate-errors control flow before this
was extracted. Each caller keeps its own domain exception type and message
text — only the control flow is centralized. See ADR-030.
"""

import json
from pathlib import Path
from typing import Any


def load_json_file(
    path: str,
    *,
    not_found_error: type[Exception],
    not_found_message: str,
    invalid_json_error: type[Exception],
    invalid_json_message: str,
) -> Any:
    """Read and parse a JSON file, translating failures into caller-specified exceptions.

    Args:
        path: Path to the JSON file.
        not_found_error: Exception type to raise if the file does not exist.
        not_found_message: Message for that exception.
        invalid_json_error: Exception type to raise if the file is not valid JSON.
        invalid_json_message: Message prefix for that exception (the original
            json.JSONDecodeError is appended after a colon).

    Returns:
        The parsed JSON value (typically a dict).
    """
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise not_found_error(not_found_message) from None
    except json.JSONDecodeError as e:
        raise invalid_json_error(f"{invalid_json_message}: {e}") from e
