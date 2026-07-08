"""Tests for the shared load_json_file primitive."""

import pytest

from src.data.json_files import load_json_file


class _NotFound(Exception):
    pass


class _Invalid(Exception):
    pass


def test_loads_valid_json(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"a": 1}', encoding="utf-8")

    result = load_json_file(
        str(path),
        not_found_error=_NotFound,
        not_found_message="not found",
        invalid_json_error=_Invalid,
        invalid_json_message="invalid",
    )

    assert result == {"a": 1}


def test_raises_caller_specified_error_when_file_missing(tmp_path):
    missing = tmp_path / "does_not_exist.json"

    with pytest.raises(_NotFound, match="custom not-found message"):
        load_json_file(
            str(missing),
            not_found_error=_NotFound,
            not_found_message="custom not-found message",
            invalid_json_error=_Invalid,
            invalid_json_message="invalid",
        )


def test_raises_caller_specified_error_when_json_is_malformed(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not valid json", encoding="utf-8")

    with pytest.raises(_Invalid, match="custom invalid-json message"):
        load_json_file(
            str(path),
            not_found_error=_NotFound,
            not_found_message="not found",
            invalid_json_error=_Invalid,
            invalid_json_message="custom invalid-json message",
        )
