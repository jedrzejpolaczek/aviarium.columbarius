import json
import logging
from pathlib import Path
from unittest.mock import patch

import duckdb
import pandas as pd
import pytest

from src.data.cards.storage.silver import SilverStorage, SilverTransforms
from src.data.cards.storage.silver.card_join import SilverCardJoin
from src.data.cards.storage.silver.cleaning import (
    _clean_booleans,
    _clean_lists,
    _clean_numerics,
    _clean_strings,
    _drop_columns,
    _filter_rows,
    _parse_json_columns,
    _rename_columns,
)
from src.data.cards.storage.silver.persistence import SilverWriter
from src.data.cards.storage.errors import StorageWriteError

MINIMAL_CONFIG = {
    "language_map": {"en": "English", "es": "Spanish"},
    "legality_map": {"Legal": "legal", "Not Legal": "not_legal"},
    "supertypes": ["Legendary", "Basic"],
    "card_types": ["Creature", "Instant"],
    "sources": {},
}


def _make_storage(tmp_path):
    """Return a SilverStorage backed by a temp bronze file and in-memory silver."""
    config_path = tmp_path / "silver_config.json"
    config_path.write_text(json.dumps(MINIMAL_CONFIG))
    bronze_path = str(tmp_path / "bronze.duckdb")
    # DuckDB requires the file to exist before opening it read-only.
    duckdb.connect(bronze_path).close()
    return SilverStorage(bronze_path, ":memory:", str(config_path))


@pytest.fixture
def storage(tmp_path):
    s = _make_storage(tmp_path)
    yield s
    s.close()


@pytest.fixture
def transforms() -> SilverTransforms:
    return SilverTransforms(
        language_map={"en": "English", "es": "Spanish"},
        legality_map={"Legal": "legal", "Not Legal": "not_legal"},
        supertypes=["Legendary", "Basic"],
        card_types=["Creature", "Instant"],
    )


@pytest.fixture
def card_join() -> SilverCardJoin:
    return SilverCardJoin(language_map={"en": "English", "es": "Spanish"})


# ---------------------------------------------------------------------------
# _filter_rows
# ---------------------------------------------------------------------------


class TestFilterRows:
    def test_drops_matching_rows_and_resets_index(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        issues: list = []
        result = _filter_rows(df, {"a": 2}, issues)
        assert list(result["a"]) == [1, 3]
        assert result.index.tolist() == [0, 1]

    def test_no_matching_rows_does_not_add_issue(self):
        df = pd.DataFrame({"a": [1, 2]})
        issues: list = []
        _filter_rows(df, {"a": 99}, issues)
        assert issues == []

    def test_dropped_rows_adds_issue_with_count(self):
        df = pd.DataFrame({"flag": [True, False, True]})
        issues: list = []
        _filter_rows(df, {"flag": True}, issues)
        assert any(
            i["issue"] == "rows_dropped" and i["count"] == 2 and i["value"] is True
            for i in issues
        )

    def test_missing_column_adds_issue(self):
        df = pd.DataFrame({"a": [1]})
        issues: list = []
        _filter_rows(df, {"missing": True}, issues)
        assert any(i["issue"] == "column_not_found" for i in issues)

    def test_drops_matching_rows_with_list_value(self):
        df = pd.DataFrame(
            {"layout": ["token", "normal", "double_faced_token", "transform"]}
        )
        issues: list = []
        result = _filter_rows(df, {"layout": ["token", "double_faced_token"]}, issues)
        assert list(result["layout"]) == ["normal", "transform"]
        assert result.index.tolist() == [0, 1]
        assert any(
            i["issue"] == "rows_dropped"
            and i["count"] == 2
            and i["value"] == ["token", "double_faced_token"]
            for i in issues
        )

    def test_list_value_no_match_does_not_add_issue(self):
        df = pd.DataFrame({"layout": ["normal", "transform"]})
        issues: list = []
        _filter_rows(df, {"layout": ["token", "emblem"]}, issues)
        assert issues == []

    def test_empty_list_value_drops_nothing(self):
        df = pd.DataFrame({"layout": ["normal", "transform"]})
        issues: list = []
        result = _filter_rows(df, {"layout": []}, issues)
        assert list(result["layout"]) == ["normal", "transform"]
        assert issues == []


# ---------------------------------------------------------------------------
# _drop_columns
# ---------------------------------------------------------------------------


class TestDropColumns:
    def test_drops_listed_columns(self):
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        issues: list = []
        result = _drop_columns(df, ["a", "c"], issues)
        assert list(result.columns) == ["b"]

    def test_missing_column_adds_issue_but_does_not_raise(self):
        df = pd.DataFrame({"a": [1]})
        issues: list = []
        result = _drop_columns(df, ["a", "missing"], issues)
        assert "a" not in result.columns
        assert any(i["issue"] == "column_not_found" for i in issues)


# ---------------------------------------------------------------------------
# Dropped columns — near-100% null in bronze (config-level regression tests)
# ---------------------------------------------------------------------------

_EMPTY_SOURCE_CONFIG: dict = {
    "drop_entries": {},
    "drop_columns": [],
    "json_columns": [],
    "string_ops": {},
    "numeric_columns": [],
    "list_operations": {},
    "bool_columns": [],
    "rename_columns": {},
}


class TestScryfallDroppedColumns:
    """Verify that Scryfall columns dropped for being near-100% null are absent
    from the transform output.  These columns were confirmed useless via EDA:
    - variation_of / life_modifier / hand_modifier: Vanguard card-type only
    - content_warning: always false after digital/oversized cards are filtered
    - tcgplayer_etched_id: etched-foil niche, 99.8% null
    - arena_id / mtgo_id / mtgo_foil_id: digital platforms, out of scope
    """

    _DROPPED = [
        "variation_of",
        "life_modifier",
        "hand_modifier",
        "content_warning",
        "tcgplayer_etched_id",
        "arena_id",
        "mtgo_id",
        "mtgo_foil_id",
    ]

    def _config(self):
        return {**_EMPTY_SOURCE_CONFIG, "drop_columns": self._DROPPED}

    def test_dropped_columns_absent_after_transform(self, transforms):
        df = pd.DataFrame({col: [None] for col in self._DROPPED} | {"name": ["A"]})
        result, _ = transforms.transform(df, self._config())
        for col in self._DROPPED:
            assert col not in result.columns, f"{col!r} should have been dropped"

    def test_non_dropped_columns_preserved(self, transforms):
        df = pd.DataFrame({col: [None] for col in self._DROPPED} | {"name": ["A"]})
        result, _ = transforms.transform(df, self._config())
        assert "name" in result.columns


class TestMtgjsonDroppedColumns:
    """Verify that MTGJson columns dropped for being near-100% null are absent
    from the transform output.  These columns were confirmed useless via EDA:
    - signature / face_printed_name / face_flavor_name: 100% null
    - hand / life: Vanguard card-type only
    - leadership_skills: 90% null, low analytical value
    - original_release_date: 98% null, not used downstream
    """

    _DROPPED = [
        "signature",
        "face_printed_name",
        "face_flavor_name",
        "hand",
        "life",
        "leadership_skills",
        "original_release_date",
    ]

    def _config(self):
        return {**_EMPTY_SOURCE_CONFIG, "drop_columns": self._DROPPED}

    def test_dropped_columns_absent_after_transform(self, transforms):
        df = pd.DataFrame({col: [None] for col in self._DROPPED} | {"uuid": ["u1"]})
        result, _ = transforms.transform(df, self._config())
        for col in self._DROPPED:
            assert col not in result.columns, f"{col!r} should have been dropped"

    def test_non_dropped_columns_preserved(self, transforms):
        df = pd.DataFrame({col: [None] for col in self._DROPPED} | {"uuid": ["u1"]})
        result, _ = transforms.transform(df, self._config())
        assert "uuid" in result.columns


# ---------------------------------------------------------------------------
# _parse_json_columns
# ---------------------------------------------------------------------------


class TestParseJsonColumns:
    def test_parses_json_strings_to_python_objects(self):
        df = pd.DataFrame({"col": ["[1, 2]", '{"key": "val"}']})
        issues: list = []
        result = _parse_json_columns(df, ["col"], issues)
        assert result["col"].iloc[0] == [1, 2]
        assert result["col"].iloc[1] == {"key": "val"}

    def test_leaves_non_strings_unchanged(self):
        df = pd.DataFrame({"col": [[1, 2], None]})
        issues: list = []
        result = _parse_json_columns(df, ["col"], issues)
        assert result["col"].iloc[0] == [1, 2]

    def test_invalid_json_leaves_value_and_adds_issue(self):
        df = pd.DataFrame({"col": ["not json"]})
        issues: list = []
        result = _parse_json_columns(df, ["col"], issues)
        assert result["col"].iloc[0] == "not json"
        assert any(i["issue"] == "json_parse_failed" for i in issues)

    def test_skips_missing_column_silently(self):
        df = pd.DataFrame({"a": [1]})
        issues: list = []
        result = _parse_json_columns(df, ["missing"], issues)
        assert list(result.columns) == ["a"]


# ---------------------------------------------------------------------------
# _clean_strings
# ---------------------------------------------------------------------------


class TestCleanStrings:
    def test_strip_removes_whitespace(self):
        df = pd.DataFrame({"name": ["  hello  ", " world "]})
        issues: list = []
        result = _clean_strings(df, {"name": ["strip"]}, issues)
        assert list(result["name"]) == ["hello", "world"]

    def test_upper(self):
        df = pd.DataFrame({"code": ["abc"]})
        issues: list = []
        result = _clean_strings(df, {"code": ["upper"]}, issues)
        assert result["code"].iloc[0] == "ABC"

    def test_lower(self):
        df = pd.DataFrame({"code": ["ABC"]})
        issues: list = []
        result = _clean_strings(df, {"code": ["lower"]}, issues)
        assert result["code"].iloc[0] == "abc"

    def test_title(self):
        df = pd.DataFrame({"name": ["hello world"]})
        issues: list = []
        result = _clean_strings(df, {"name": ["title"]}, issues)
        assert result["name"].iloc[0] == "Hello World"

    def test_replace_sentinel_replaces_underscore_with_none(self):
        df = pd.DataFrame({"val": ["_", "real", "_"]})
        issues: list = []
        result = _clean_strings(df, {"val": ["replace_sentinel"]}, issues)
        assert pd.isna(result["val"].iloc[0])
        assert result["val"].iloc[1] == "real"
        assert any(
            i["issue"] == "sentinel_replaced" and i["count"] == 2 for i in issues
        )

    def test_missing_column_adds_issue(self):
        df = pd.DataFrame({"a": [1]})
        issues: list = []
        _clean_strings(df, {"missing": ["strip"]}, issues)
        assert any(i["issue"] == "column_not_found" for i in issues)


# ---------------------------------------------------------------------------
# _clean_numerics
# ---------------------------------------------------------------------------


class TestCleanNumerics:
    def test_coerces_string_numbers_to_float(self):
        df = pd.DataFrame({"val": ["1.5", "2", "3.0"]})
        issues: list = []
        result = _clean_numerics(df, ["val"], issues)
        assert list(result["val"]) == [1.5, 2.0, 3.0]

    def test_unparseable_value_becomes_nan_and_adds_issue(self):
        df = pd.DataFrame({"val": ["1", "bad", "3"]})
        issues: list = []
        result = _clean_numerics(df, ["val"], issues)
        assert pd.isna(result["val"].iloc[1])
        assert any(i["issue"] == "unparseable_values" for i in issues)

    def test_skips_missing_column_silently(self):
        df = pd.DataFrame({"a": [1]})
        issues: list = []
        result = _clean_numerics(df, ["missing"], issues)
        assert list(result.columns) == ["a"]


# ---------------------------------------------------------------------------
# _clean_lists
# ---------------------------------------------------------------------------


class TestCleanLists:
    def test_fill_empty_replaces_none_with_empty_list(self):
        df = pd.DataFrame({"tags": [None, ["a"], None]})
        issues: list = []
        result = _clean_lists(df, {"tags": ["fill_empty"]}, issues)
        assert result["tags"].iloc[0] == []
        assert result["tags"].iloc[1] == ["a"]

    def test_null_fill_adds_issue_with_count(self):
        df = pd.DataFrame({"tags": [None, None]})
        issues: list = []
        _clean_lists(df, {"tags": ["fill_empty"]}, issues)
        assert any(
            i["issue"] == "nulls_filled_with_empty_list" and i["count"] == 2
            for i in issues
        )

    def test_upper_applies_to_items(self):
        df = pd.DataFrame({"tags": [["white", "blue"]]})
        issues: list = []
        result = _clean_lists(df, {"tags": ["upper"]}, issues)
        assert result["tags"].iloc[0] == ["WHITE", "BLUE"]

    def test_lower_applies_to_items(self):
        df = pd.DataFrame({"tags": [["WHITE", "BLUE"]]})
        issues: list = []
        result = _clean_lists(df, {"tags": ["lower"]}, issues)
        assert result["tags"].iloc[0] == ["white", "blue"]

    def test_title_applies_to_items(self):
        df = pd.DataFrame({"tags": [["hello world"]]})
        issues: list = []
        result = _clean_lists(df, {"tags": ["title"]}, issues)
        assert result["tags"].iloc[0] == ["Hello World"]


# ---------------------------------------------------------------------------
# _clean_booleans
# ---------------------------------------------------------------------------


class TestCleanBooleans:
    def test_fills_none_with_false(self):
        df = pd.DataFrame({"flag": [None, True, None]})
        issues: list = []
        result = _clean_booleans(df, ["flag"], issues)
        assert list(result["flag"]) == [False, True, False]

    def test_casts_column_to_bool_dtype(self):
        df = pd.DataFrame({"flag": [1, 0, None]})
        issues: list = []
        result = _clean_booleans(df, ["flag"], issues)
        assert result["flag"].dtype == bool

    def test_null_fill_adds_issue(self):
        df = pd.DataFrame({"flag": [None]})
        issues: list = []
        _clean_booleans(df, ["flag"], issues)
        assert any(i["issue"] == "nulls_filled_with_false" for i in issues)


# ---------------------------------------------------------------------------
# SilverTransforms._normalize_values
# ---------------------------------------------------------------------------


class TestNormalizeValues:
    def test_expands_known_language_codes(self, transforms):
        df = pd.DataFrame({"language": ["en", "es"]})
        issues: list = []
        result = transforms._normalize_values(df, issues)
        assert list(result["language"]) == ["English", "Spanish"]

    def test_unknown_language_code_left_unchanged_and_adds_issue(self, transforms):
        df = pd.DataFrame({"language": ["xx"]})
        issues: list = []
        result = transforms._normalize_values(df, issues)
        assert result["language"].iloc[0] == "xx"
        assert any(i["issue"] == "unknown_language_codes" for i in issues)

    def test_already_expanded_language_not_re_mapped(self, transforms):
        df = pd.DataFrame({"language": ["English"]})
        issues: list = []
        result = transforms._normalize_values(df, issues)
        assert result["language"].iloc[0] == "English"

    def test_normalizes_legality_values(self, transforms):
        df = pd.DataFrame(
            {"legalities": [{"standard": "Legal", "modern": "Not Legal"}]}
        )
        issues: list = []
        result = transforms._normalize_values(df, issues)
        leg = result["legalities"].iloc[0]
        assert leg["standard"] == "legal"
        assert leg["modern"] == "not_legal"

    def test_uses_lang_column_when_language_absent(self, transforms):
        df = pd.DataFrame({"lang": ["en"]})
        issues: list = []
        result = transforms._normalize_values(df, issues)
        assert result["lang"].iloc[0] == "English"


# ---------------------------------------------------------------------------
# SilverTransforms._parse_type_line
# ---------------------------------------------------------------------------


class TestParseTypeLine:
    def test_splits_supertypes_types_and_subtypes(self, transforms):
        supers, types, subs = transforms._parse_type_line(
            "Legendary Creature — Human Wizard"
        )
        assert supers == ["Legendary"]
        assert types == ["Creature"]
        assert subs == ["Human", "Wizard"]

    def test_no_em_dash_returns_empty_subtypes(self, transforms):
        supers, types, subs = transforms._parse_type_line("Basic Instant")
        assert supers == ["Basic"]
        assert types == ["Instant"]
        assert subs == []

    def test_none_returns_three_empty_lists(self, transforms):
        assert transforms._parse_type_line(None) == ([], [], [])

    def test_empty_string_returns_three_empty_lists(self, transforms):
        assert transforms._parse_type_line("") == ([], [], [])

    def test_unknown_words_not_included_in_supertypes_or_types(self, transforms):
        supers, types, subs = transforms._parse_type_line("Unknown — Foo")
        assert supers == []
        assert types == []
        assert subs == ["Foo"]


# ---------------------------------------------------------------------------
# SilverTransforms._add_computed_columns
# ---------------------------------------------------------------------------


class TestAddComputedColumns:
    def test_errata_true_when_text_differs_from_original(self, transforms):
        df = pd.DataFrame(
            {
                "text": ["new text", "same"],
                "original_text": ["old text", "same"],
            }
        )
        issues: list = []
        result = transforms._add_computed_columns(df, issues)
        assert list(result["errata"]) == [True, False]

    def test_errata_issue_added_when_errata_cards_exist(self, transforms):
        df = pd.DataFrame({"text": ["new"], "original_text": ["old"]})
        issues: list = []
        transforms._add_computed_columns(df, issues)
        assert any(i["issue"] == "cards_with_errata" for i in issues)

    def test_original_type_parsed_into_three_columns(self, transforms):
        df = pd.DataFrame({"original_type": ["Legendary Creature — Elf"]})
        issues: list = []
        result = transforms._add_computed_columns(df, issues)
        assert result["original_supertypes"].iloc[0] == ["Legendary"]
        assert result["original_types"].iloc[0] == ["Creature"]
        assert result["original_subtypes"].iloc[0] == ["Elf"]

    def test_ascii_name_null_filled_from_name(self, transforms):
        df = pd.DataFrame({"ascii_name": [None, "Foo"], "name": ["Bar", "Baz"]})
        issues: list = []
        result = transforms._add_computed_columns(df, issues)
        assert result["ascii_name"].iloc[0] == "Bar"
        assert result["ascii_name"].iloc[1] == "Foo"

    def test_ascii_name_fill_adds_issue(self, transforms):
        df = pd.DataFrame({"ascii_name": [None], "name": ["Bar"]})
        issues: list = []
        transforms._add_computed_columns(df, issues)
        assert any(i["issue"] == "nulls_filled_with_name" for i in issues)


# ---------------------------------------------------------------------------
# _rename_columns
# ---------------------------------------------------------------------------


class TestRenameColumns:
    def test_renames_present_columns(self):
        df = pd.DataFrame({"old": [1], "other": [2]})
        issues: list = []
        result = _rename_columns(df, {"old": "new"}, issues)
        assert "new" in result.columns
        assert "old" not in result.columns

    def test_missing_column_adds_issue_and_skips_silently(self):
        df = pd.DataFrame({"a": [1]})
        issues: list = []
        result = _rename_columns(df, {"a": "x", "missing": "y"}, issues)
        assert "x" in result.columns
        assert any(i["issue"] == "columns_not_found" for i in issues)


# ---------------------------------------------------------------------------
# SilverCardJoin
# ---------------------------------------------------------------------------


class TestSilverCardJoin:
    def test_joins_mtgjson_and_scryfall_on_scryfall_id(self, card_join):
        mtgjson = pd.DataFrame(
            {
                "uuid": ["u1", "u2"],
                "identifiers": [{"scryfall_id": "s1"}, {"scryfall_id": "s2"}],
                "name": ["CardA", "CardB"],
            }
        )
        scryfall = pd.DataFrame(
            {
                "id": ["s1", "s2"],
                "image_uris": ["img1", "img2"],
            }
        )
        result = card_join.join(mtgjson, scryfall)
        assert len(result) == 2
        assert "image_uris" in result.columns
        assert "id" not in result.columns

    def test_missing_source_returns_empty_dataframe(self, storage):
        # Tests the orchestration-level guard in SilverStorage._join_cards
        # which checks for both required sources before delegating to SilverCardJoin.
        result = storage._join_cards({"mtgjson_cards": pd.DataFrame()})
        assert result.empty

    def test_scryfall_columns_duplicated_in_mtgjson_are_excluded(self, card_join):
        mtgjson = pd.DataFrame(
            {
                "uuid": ["u1"],
                "identifiers": [{"scryfall_id": "s1"}],
                "name": ["CardA"],
            }
        )
        scryfall = pd.DataFrame(
            {
                "id": ["s1"],
                "name": ["CardA"],
                "extra": ["val"],
            }
        )
        result = card_join.join(mtgjson, scryfall)
        assert result.columns.tolist().count("name") == 1

    def test_unmatched_mtgjson_rows_kept_with_null_scryfall_columns(self, card_join):
        mtgjson = pd.DataFrame(
            {
                "uuid": ["u1"],
                "identifiers": [{"scryfall_id": "no-match"}],
                "name": ["CardA"],
            }
        )
        scryfall = pd.DataFrame({"id": ["s1"], "image_uris": ["img1"]})
        result = card_join.join(mtgjson, scryfall)
        mtgjson_row = result[result["uuid"] == "u1"].iloc[0]
        assert pd.isna(mtgjson_row["image_uris"])

    def test_camel_case_scryfallId_key_produces_null_scryfall_id(self, card_join):
        # Pydantic serialises MtgjsonIdentifiers with snake_case field names.
        # Using the raw alias ("scryfallId") must NOT accidentally match.
        mtgjson = pd.DataFrame(
            {
                "uuid": ["u1"],
                "identifiers": [{"scryfallId": "s1"}],
                "name": ["CardA"],
            }
        )
        scryfall = pd.DataFrame({"id": ["s1"], "image_uris": ["img1"]})
        result = card_join.join(mtgjson, scryfall)
        assert pd.isna(result["scryfall_id"].iloc[0])

    def test_canonical_uuid_equals_uuid_for_mtgjson_matched_rows(self, card_join):
        # Use post-transform column names: MTGJson 'number' → 'collector_number',
        # Scryfall 'set' → 'set_code' (transforms run before card_join in production).
        mtgjson = pd.DataFrame(
            {
                "uuid": ["u1"],
                "identifiers": [{"scryfall_id": "s1"}],
                "name": ["CardA"],
                "set_code": ["TST"],
                "collector_number": ["1"],
            }
        )
        scryfall = pd.DataFrame(
            {"id": ["s1"], "set_code": ["TST"], "collector_number": ["1"]}
        )
        result = card_join.join(mtgjson, scryfall)
        assert result.loc[result["uuid"] == "u1", "canonical_uuid"].iloc[0] == "u1"

    def test_canonical_uuid_resolved_for_scryfall_only_language_variant(
        self, card_join
    ):
        # English card matched to MTGJson; Japanese variant is Scryfall-only.
        # Both share set_code=TST, collector_number=1 → canonical_uuid links to u1.
        # Post-transform column names used (set_code, collector_number).
        mtgjson = pd.DataFrame(
            {
                "uuid": ["u1"],
                "identifiers": [{"scryfall_id": "s1-en"}],
                "name": ["CardA"],
                "set_code": ["TST"],
                "collector_number": ["1"],
            }
        )
        scryfall = pd.DataFrame(
            {
                "id": ["s1-en", "s1-ja"],
                "set_code": ["TST", "TST"],
                "collector_number": ["1", "1"],
                "language": ["English", "Japanese"],
            }
        )
        result = card_join.join(mtgjson, scryfall)
        ja_row = result[result["scryfall_id"] == "s1-ja"].iloc[0]
        assert pd.isna(ja_row["uuid"])
        assert ja_row["canonical_uuid"] == "u1"

    def test_canonical_uuid_null_for_scryfall_only_with_no_en_match(self, card_join):
        # Digital-exclusive card — only in Scryfall, no MTGJson UUID anywhere with
        # the same set+collector_number (because set "DIGT" has no paper MTGJson entry).
        mtgjson = pd.DataFrame(
            {
                "uuid": ["u1"],
                "identifiers": [{"scryfall_id": "s1"}],
                "name": ["PaperCard"],
                "set_code": ["TST"],
                "collector_number": ["1"],
            }
        )
        scryfall = pd.DataFrame(
            {
                "id": ["s1", "s-digital"],
                "set_code": ["TST", "DIGT"],
                "collector_number": ["1", "99"],
                "language": ["English", "English"],
            }
        )
        result = card_join.join(mtgjson, scryfall)
        dig_row = result[result["scryfall_id"] == "s-digital"].iloc[0]
        assert pd.isna(dig_row["canonical_uuid"])

    def test_scryfall_only_row_name_filled_from_scryfall_fallback(self, card_join):
        # Scryfall-only rows (no MTGJson match) should have their 'name' backfilled
        # from the Scryfall 'name' column via _SCRYFALL_FALLBACK_MAP.
        mtgjson = pd.DataFrame(
            {
                "uuid": ["u1"],
                "identifiers": [{"scryfall_id": "s1"}],
                "name": ["CardA"],
            }
        )
        scryfall = pd.DataFrame(
            {
                "id": ["s1", "s2"],
                "name": ["CardA", "CardB"],
                "image_uris": ["img1", "img2"],
            }
        )
        result = card_join.join(mtgjson, scryfall)
        s2_row = result[result["scryfall_id"] == "s2"].iloc[0]
        assert s2_row["name"] == "CardB"

    def test_dfc_back_face_deduplicated_keeping_front(self, card_join):
        # MTGJson stores DFC front and back as separate rows sharing the same scryfall_id.
        # After join, only the front face (first row) should remain.
        mtgjson = pd.DataFrame(
            {
                "uuid": ["u1-front", "u1-back"],
                "identifiers": [{"scryfall_id": "s1"}, {"scryfall_id": "s1"}],
                "name": ["CardFront", "CardBack"],
            }
        )
        scryfall = pd.DataFrame({"id": ["s1"], "image_uris": ["img1"]})
        result = card_join.join(mtgjson, scryfall)
        s1_rows = result[result["scryfall_id"] == "s1"]
        assert len(s1_rows) == 1
        assert s1_rows.iloc[0]["uuid"] == "u1-front"


# ---------------------------------------------------------------------------
# populate / update (integration smoke tests)
# ---------------------------------------------------------------------------


_META_HISTORY_SOURCE_CONFIG = {
    "drop_entries": {},
    "drop_columns": [],
    "json_columns": [],
    "string_ops": {"id": ["strip"], "snapshot_date": ["strip"]},
    "numeric_columns": [],
    "list_operations": {},
    "bool_columns": [],
    "rename_columns": {},
}


def _make_storage_with_meta_bronze(
    tmp_path: Path, rows: list[tuple[str, str]]
) -> SilverStorage:
    """SilverStorage whose config includes scryfall_meta_history and bronze has rows."""
    config = {
        **MINIMAL_CONFIG,
        "sources": {"scryfall_meta_history": _META_HISTORY_SOURCE_CONFIG},
    }
    config_path = tmp_path / "silver_config.json"
    config_path.write_text(json.dumps(config))

    bronze_path = str(tmp_path / "bronze.duckdb")
    con = duckdb.connect(bronze_path)
    df = pd.DataFrame(rows, columns=["id", "snapshot_date"])
    con.register("_df", df)
    con.execute("CREATE TABLE bronze_scryfall_meta_history AS SELECT * FROM _df")
    con.unregister("_df")
    con.close()

    return SilverStorage(bronze_path, ":memory:", str(config_path))


class TestPopulateUpdate:
    def test_populate_with_empty_sources_does_not_raise(self, tmp_path):
        with patch.object(SilverStorage, "_write_report", create=True):
            with _make_storage(tmp_path) as s:
                s.populate()

    def test_update_with_empty_sources_does_not_raise(self, tmp_path):
        with patch.object(SilverStorage, "_write_report", create=True):
            with _make_storage(tmp_path) as s:
                s.update()

    def test_pipeline_writes_meta_history_to_silver_meta_history(self, tmp_path):
        with patch("src.data.cards.storage.silver.storage.write_report"):
            with _make_storage_with_meta_bronze(
                tmp_path, [("abc", "2026-05-11"), ("def", "2026-05-11")]
            ) as s:
                s.populate()
                tables = {r[0] for r in s._silver_con.execute("SHOW TABLES").fetchall()}
                assert "silver_meta_history" in tables
                row = s._silver_con.execute(
                    "SELECT count(*) FROM silver_meta_history"
                ).fetchone()
                assert row is not None and row[0] == 2

    def test_pipeline_does_not_create_silver_scryfall_meta_history(self, tmp_path):
        with patch("src.data.cards.storage.silver.storage.write_report"):
            with _make_storage_with_meta_bronze(tmp_path, [("abc", "2026-05-11")]) as s:
                s.populate()
                tables = {r[0] for r in s._silver_con.execute("SHOW TABLES").fetchall()}
                assert "silver_scryfall_meta_history" not in tables

    def test_meta_history_filtered_to_ids_in_silver_cards(self, tmp_path):
        # meta_df has two IDs; _join_cards is patched to return a cards_df that
        # only contains one of them — the other should be excluded from silver_meta_history.
        with patch("src.data.cards.storage.silver.storage.write_report"):
            s = _make_storage_with_meta_bronze(
                tmp_path,
                [("id-keep", "2026-05-11"), ("id-drop", "2026-05-11")],
            )
            cards_mock = pd.DataFrame({"scryfall_id": ["id-keep"]})
            with s, patch.object(s, "_join_cards", return_value=cards_mock):
                s.populate()
                row = s._silver_con.execute(
                    "SELECT count(*) FROM silver_meta_history"
                ).fetchone()
                assert row is not None and row[0] == 1
                kept = s._silver_con.execute(
                    "SELECT id FROM silver_meta_history"
                ).fetchone()
                assert kept is not None and kept[0] == "id-keep"


# ---------------------------------------------------------------------------
# Helpers for SilverPriceBuilder tests
# ---------------------------------------------------------------------------

_SCRYFALL_PRICES_JSON = json.dumps(
    {"eur": 3.20, "eur_foil": 8.50, "usd": 3.50, "usd_foil": 9.00, "tix": 0.05}
)

_MTGJSON_PAPER_JSON = json.dumps(
    {
        "cardmarket": {
            "retail": {"normal": {"2026-05-11": 3.20}, "foil": {"2026-05-11": 8.50}},
            "buylist": {"normal": {"2026-05-11": 1.80}},
        },
        "tcgplayer": {
            "retail": {"normal": {"2026-05-11": 3.50}, "foil": {"2026-05-11": 9.00}},
            "buylist": {"normal": {"2026-05-11": 2.10}},
        },
    }
)


def _make_storage_with_bronze(
    tmp_path: Path, bronze_tables: dict[str, pd.DataFrame]
) -> SilverStorage:
    """Create SilverStorage backed by a pre-populated Bronze DuckDB file."""
    config_path = tmp_path / "silver_config.json"
    config_path.write_text(json.dumps(MINIMAL_CONFIG))
    bronze_path = str(tmp_path / "bronze.duckdb")

    con = duckdb.connect(bronze_path)
    for table_name, df in bronze_tables.items():
        con.register("_df", df)
        con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM _df")
        con.unregister("_df")
    con.close()

    return SilverStorage(bronze_path, ":memory:", str(config_path))


def _seed_silver_cards(storage: SilverStorage, rows: list[tuple]) -> None:
    """Insert rows into the in-memory silver_cards table as (uuid, scryfall_id[, canonical_uuid, language])."""
    storage._silver_con.execute(
        "CREATE TABLE IF NOT EXISTS silver_cards "
        "(uuid VARCHAR, scryfall_id VARCHAR, canonical_uuid VARCHAR, language VARCHAR)"
    )
    for row in rows:
        uuid = row[0]
        scryfall_id = row[1]
        canonical_uuid = row[2] if len(row) > 2 else uuid
        language = row[3] if len(row) > 3 else "English"
        storage._silver_con.execute(
            "INSERT INTO silver_cards VALUES (?, ?, ?, ?)",
            [uuid, scryfall_id, canonical_uuid, language],
        )


# ---------------------------------------------------------------------------
# SilverPriceBuilder.build
# ---------------------------------------------------------------------------


class TestSilverPriceBuilder:
    def test_returns_empty_dataframe_when_silver_cards_missing(self, tmp_path):
        scryfall_hist = pd.DataFrame(
            {
                "id": ["s1"],
                "snapshot_date": ["2026-05-11"],
                "prices": [_SCRYFALL_PRICES_JSON],
            }
        )
        with _make_storage_with_bronze(
            tmp_path, {"bronze_scryfall_prices_history": scryfall_hist}
        ) as s:
            result = s._prices.build("2026-05-11")
            assert result.empty

    def test_returns_empty_dataframe_when_bronze_scryfall_history_missing(
        self, tmp_path
    ):
        with _make_storage_with_bronze(tmp_path, {}) as s:
            _seed_silver_cards(s, [("u1", "s1")])
            result = s._prices.build("2026-05-11")
            assert result.empty

    def test_happy_path_both_sources_present(self, tmp_path):
        scryfall_hist = pd.DataFrame(
            {
                "id": ["s1"],
                "snapshot_date": ["2026-05-11"],
                "prices": [_SCRYFALL_PRICES_JSON],
            }
        )
        mtgjson_hist = pd.DataFrame(
            {
                "uuid": ["u1"],
                "snapshot_date": ["2026-05-11"],
                "paper": [_MTGJSON_PAPER_JSON],
            }
        )
        with _make_storage_with_bronze(
            tmp_path,
            {
                "bronze_scryfall_prices_history": scryfall_hist,
                "bronze_mtgjson_prices_history": mtgjson_hist,
            },
        ) as s:
            _seed_silver_cards(s, [("u1", "s1")])
            result = s._prices.build("2026-05-11")

            assert len(result) == 1
            row = result.iloc[0]
            assert row["uuid"] == "u1"
            assert row["scryfall_id"] == "s1"
            assert row["eur"] == pytest.approx(3.20)
            assert row["cardmarket_eur"] == pytest.approx(3.20)
            assert row["cardmarket_buylist_eur"] == pytest.approx(1.80)
            assert row["tcgplayer_usd"] == pytest.approx(3.50)

    def test_happy_path_has_all_expected_columns(self, tmp_path):
        scryfall_hist = pd.DataFrame(
            {
                "id": ["s1"],
                "snapshot_date": ["2026-05-11"],
                "prices": [_SCRYFALL_PRICES_JSON],
            }
        )
        with _make_storage_with_bronze(
            tmp_path, {"bronze_scryfall_prices_history": scryfall_hist}
        ) as s:
            _seed_silver_cards(s, [("u1", "s1")])
            result = s._prices.build("2026-05-11")

            expected_columns = [
                "uuid",
                "scryfall_id",
                "snapshot_date",
                "eur",
                "eur_foil",
                "usd",
                "usd_foil",
                "cardmarket_eur",
                "cardmarket_eur_foil",
                "cardmarket_buylist_eur",
                "tcgplayer_usd",
                "tcgplayer_usd_foil",
                "tcgplayer_buylist_usd",
            ]
            assert list(result.columns) == expected_columns

    def test_mtgjson_missing_fills_columns_with_none(self, tmp_path):
        scryfall_hist = pd.DataFrame(
            {
                "id": ["s1"],
                "snapshot_date": ["2026-05-11"],
                "prices": [_SCRYFALL_PRICES_JSON],
            }
        )
        with _make_storage_with_bronze(
            tmp_path, {"bronze_scryfall_prices_history": scryfall_hist}
        ) as s:
            _seed_silver_cards(s, [("u1", "s1")])
            result = s._prices.build("2026-05-11")

            assert len(result) == 1
            assert pd.isna(result.iloc[0]["cardmarket_eur"])
            assert pd.isna(result.iloc[0]["tcgplayer_usd"])

    def test_scryfall_card_with_no_silver_match_is_dropped(self, tmp_path):
        scryfall_hist = pd.DataFrame(
            {
                "id": ["s1", "s-no-match"],
                "snapshot_date": ["2026-05-11", "2026-05-11"],
                "prices": [_SCRYFALL_PRICES_JSON, _SCRYFALL_PRICES_JSON],
            }
        )
        with _make_storage_with_bronze(
            tmp_path, {"bronze_scryfall_prices_history": scryfall_hist}
        ) as s:
            _seed_silver_cards(s, [("u1", "s1")])
            result = s._prices.build("2026-05-11")

            assert len(result) == 1
            assert result.iloc[0]["scryfall_id"] == "s1"

    def test_mtgjson_card_with_no_scryfall_history_row_is_excluded(self, tmp_path):
        scryfall_hist = pd.DataFrame(
            {
                "id": ["s1"],
                "snapshot_date": ["2026-05-11"],
                "prices": [_SCRYFALL_PRICES_JSON],
            }
        )
        mtgjson_hist = pd.DataFrame(
            {
                "uuid": ["u1", "u-no-scryfall"],
                "snapshot_date": ["2026-05-11", "2026-05-11"],
                "paper": [_MTGJSON_PAPER_JSON, _MTGJSON_PAPER_JSON],
            }
        )
        with _make_storage_with_bronze(
            tmp_path,
            {
                "bronze_scryfall_prices_history": scryfall_hist,
                "bronze_mtgjson_prices_history": mtgjson_hist,
            },
        ) as s:
            _seed_silver_cards(s, [("u1", "s1"), ("u-no-scryfall", None)])
            result = s._prices.build("2026-05-11")

            assert len(result) == 1
            assert result.iloc[0]["uuid"] == "u1"

    def test_build_ignores_bronze_rows_from_other_dates(self, tmp_path):
        scryfall_hist = pd.DataFrame(
            {
                "id": ["s1", "s1"],
                "snapshot_date": ["2026-05-10", "2026-05-11"],
                "prices": [_SCRYFALL_PRICES_JSON, _SCRYFALL_PRICES_JSON],
            }
        )
        with _make_storage_with_bronze(
            tmp_path, {"bronze_scryfall_prices_history": scryfall_hist}
        ) as s:
            _seed_silver_cards(s, [("u1", "s1")])
            result = s._prices.build("2026-05-11")

            assert len(result) == 1
            assert result.iloc[0]["snapshot_date"] == "2026-05-11"

    def test_english_card_with_stale_scryfall_id_uses_canonical_uuid(self, tmp_path):
        # Simulate an English paper card where MTGJson holds a stale scryfall_id:
        # the direct scryfall_id→uuid join in card_join missed, leaving uuid=NULL,
        # but (set_code, collector_number) resolved canonical_uuid="u1".
        # The card's current Scryfall ID "s-stale" has real prices and must be
        # included in silver_prices_history under canonical_uuid.
        scryfall_hist = pd.DataFrame(
            {
                "id": ["s-stale"],
                "snapshot_date": ["2026-05-11"],
                "prices": [_SCRYFALL_PRICES_JSON],
            }
        )
        with _make_storage_with_bronze(
            tmp_path, {"bronze_scryfall_prices_history": scryfall_hist}
        ) as s:
            # uuid=None forces COALESCE path; canonical_uuid resolves to "u1"
            _seed_silver_cards(s, [(None, "s-stale", "u1", "English")])
            result = s._prices.build("2026-05-11")

            assert len(result) == 1
            assert result.iloc[0]["uuid"] == "u1"
            assert result.iloc[0]["scryfall_id"] == "s-stale"
            assert result.iloc[0]["eur"] == pytest.approx(3.20)

    def test_non_english_canonical_uuid_card_excluded_from_main_prices(self, tmp_path):
        # Non-English language variants (uuid=NULL, canonical_uuid=NOT NULL, language≠English)
        # must NOT appear in the main price history — they are handled by
        # build_language_prices to avoid duplicating the canonical card's prices.
        scryfall_hist = pd.DataFrame(
            {
                "id": ["s1-en", "s1-ja"],
                "snapshot_date": ["2026-05-11", "2026-05-11"],
                "prices": [_SCRYFALL_PRICES_JSON, _SCRYFALL_PRICES_JSON],
            }
        )
        with _make_storage_with_bronze(
            tmp_path, {"bronze_scryfall_prices_history": scryfall_hist}
        ) as s:
            _seed_silver_cards(s, [("u1", "s1-en")])
            _seed_silver_language_variant_cards(s, [("s1-ja", "u1", "Japanese")])
            result = s._prices.build("2026-05-11")

            assert len(result) == 1
            assert result.iloc[0]["scryfall_id"] == "s1-en"


# ---------------------------------------------------------------------------
# SilverPriceBuilder._fill_price_history
# ---------------------------------------------------------------------------


def _make_price_df(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal silver_prices_history DataFrame for fill tests."""
    null_cols = [
        "eur_foil",
        "usd",
        "usd_foil",
        "cardmarket_eur",
        "cardmarket_eur_foil",
        "cardmarket_buylist_eur",
        "tcgplayer_usd",
        "tcgplayer_usd_foil",
        "tcgplayer_buylist_usd",
    ]
    return pd.DataFrame(
        [
            {
                "uuid": r.get("uuid"),
                "scryfall_id": r.get("scryfall_id"),
                "snapshot_date": r["snapshot_date"],
                "eur": r.get("eur"),
                **{c: r.get(c) for c in null_cols},
            }
            for r in rows
        ]
    )


def _seed_silver_prices_history(storage: SilverStorage, rows: list[dict]) -> None:
    """Insert rows into silver_prices_history to simulate prior-day snapshots."""
    df = _make_price_df(rows)
    storage._silver_con.register("_ph", df)
    storage._silver_con.execute(
        "CREATE TABLE IF NOT EXISTS silver_prices_history AS SELECT * FROM _ph"
    )
    storage._silver_con.unregister("_ph")


class TestFillPriceHistory:
    def test_empty_df_returned_unchanged(self, storage):
        df = _make_price_df([])
        result = storage._prices._fill_price_history(df, "2026-05-11")
        assert result.empty

    def test_no_null_prices_returns_df_unchanged(self, storage):
        df = _make_price_df(
            [
                {
                    "uuid": "u1",
                    "scryfall_id": "s1",
                    "snapshot_date": "2026-05-11",
                    "eur": 5.0,
                }
            ]
        )
        result = storage._prices._fill_price_history(df, "2026-05-11")
        assert len(result) == 1
        assert result.iloc[0]["eur"] == pytest.approx(5.0)

    def test_no_prior_silver_table_returns_df_unchanged(self, storage):
        df = _make_price_df(
            [
                {
                    "uuid": "u1",
                    "scryfall_id": "s1",
                    "snapshot_date": "2026-05-11",
                    "eur": None,
                }
            ]
        )
        result = storage._prices._fill_price_history(df, "2026-05-11")
        assert len(result) == 1
        assert pd.isna(result.iloc[0]["eur"])

    def test_null_prices_filled_from_prior_silver_row(self, storage):
        _seed_silver_prices_history(
            storage,
            [
                {
                    "uuid": "u1",
                    "scryfall_id": "s1",
                    "snapshot_date": "2026-05-10",
                    "eur": 5.0,
                }
            ],
        )
        df = _make_price_df(
            [
                {
                    "uuid": "u1",
                    "scryfall_id": "s1",
                    "snapshot_date": "2026-05-11",
                    "eur": None,
                }
            ]
        )
        result = storage._prices._fill_price_history(df, "2026-05-11")
        assert result.iloc[0]["eur"] == pytest.approx(5.0)

    def test_non_null_prices_not_overwritten_by_prior_silver(self, storage):
        _seed_silver_prices_history(
            storage,
            [
                {
                    "uuid": "u1",
                    "scryfall_id": "s1",
                    "snapshot_date": "2026-05-10",
                    "eur": 99.0,
                }
            ],
        )
        df = _make_price_df(
            [
                {
                    "uuid": "u1",
                    "scryfall_id": "s1",
                    "snapshot_date": "2026-05-11",
                    "eur": 5.0,
                }
            ]
        )
        result = storage._prices._fill_price_history(df, "2026-05-11")
        assert result.iloc[0]["eur"] == pytest.approx(5.0)

    def test_no_prior_silver_row_leaves_null(self, storage):
        # silver_prices_history exists but has no row for this card
        _seed_silver_prices_history(
            storage,
            [
                {
                    "uuid": "u2",
                    "scryfall_id": "s2",
                    "snapshot_date": "2026-05-10",
                    "eur": 99.0,
                }
            ],
        )
        df = _make_price_df(
            [
                {
                    "uuid": "u1",
                    "scryfall_id": "s1",
                    "snapshot_date": "2026-05-11",
                    "eur": None,
                }
            ]
        )
        result = storage._prices._fill_price_history(df, "2026-05-11")
        assert pd.isna(result.iloc[0]["eur"])

    def test_two_cards_do_not_mix_fills(self, storage):
        _seed_silver_prices_history(
            storage,
            [
                {
                    "uuid": "u1",
                    "scryfall_id": "s1",
                    "snapshot_date": "2026-05-10",
                    "eur": 5.0,
                },
                {
                    "uuid": "u2",
                    "scryfall_id": "s2",
                    "snapshot_date": "2026-05-10",
                    "eur": 99.0,
                },
            ],
        )
        df = _make_price_df(
            [
                {
                    "uuid": "u1",
                    "scryfall_id": "s1",
                    "snapshot_date": "2026-05-11",
                    "eur": None,
                },
                {
                    "uuid": "u2",
                    "scryfall_id": "s2",
                    "snapshot_date": "2026-05-11",
                    "eur": None,
                },
            ]
        )
        result = storage._prices._fill_price_history(df, "2026-05-11")
        assert result[result["scryfall_id"] == "s1"].iloc[0]["eur"] == pytest.approx(
            5.0
        )
        assert result[result["scryfall_id"] == "s2"].iloc[0]["eur"] == pytest.approx(
            99.0
        )

    def test_most_recent_prior_row_used_when_multiple_exist(self, storage):
        _seed_silver_prices_history(
            storage,
            [
                {
                    "uuid": "u1",
                    "scryfall_id": "s1",
                    "snapshot_date": "2026-05-08",
                    "eur": 1.0,
                },
                {
                    "uuid": "u1",
                    "scryfall_id": "s1",
                    "snapshot_date": "2026-05-10",
                    "eur": 7.0,
                },
            ],
        )
        df = _make_price_df(
            [
                {
                    "uuid": "u1",
                    "scryfall_id": "s1",
                    "snapshot_date": "2026-05-11",
                    "eur": None,
                }
            ]
        )
        result = storage._prices._fill_price_history(df, "2026-05-11")
        assert result.iloc[0]["eur"] == pytest.approx(7.0)

    def test_same_date_row_not_used_as_fill_source(self, storage):
        # snapshot_date = today must not be used as the fill source (WHERE < today)
        _seed_silver_prices_history(
            storage,
            [
                {
                    "uuid": "u1",
                    "scryfall_id": "s1",
                    "snapshot_date": "2026-05-11",
                    "eur": 99.0,
                }
            ],
        )
        df = _make_price_df(
            [
                {
                    "uuid": "u1",
                    "scryfall_id": "s1",
                    "snapshot_date": "2026-05-11",
                    "eur": None,
                }
            ]
        )
        result = storage._prices._fill_price_history(df, "2026-05-11")
        assert pd.isna(result.iloc[0]["eur"])

    def test_column_order_preserved(self, storage):
        df = _make_price_df(
            [
                {
                    "uuid": "u1",
                    "scryfall_id": "s1",
                    "snapshot_date": "2026-05-11",
                    "eur": 5.0,
                }
            ]
        )
        result = storage._prices._fill_price_history(df, "2026-05-11")
        assert list(result.columns) == list(df.columns)


# ---------------------------------------------------------------------------
# SilverPriceBuilder.build_language_prices
# ---------------------------------------------------------------------------


def _seed_silver_language_variant_cards(
    storage: SilverStorage, rows: list[tuple]
) -> None:
    """Insert language-variant rows into silver_cards as (uuid=NULL, scryfall_id, canonical_uuid, language)."""
    storage._silver_con.execute(
        "CREATE TABLE IF NOT EXISTS silver_cards "
        "(uuid VARCHAR, scryfall_id VARCHAR, canonical_uuid VARCHAR, language VARCHAR)"
    )
    for scryfall_id, canonical_uuid, language in rows:
        storage._silver_con.execute(
            "INSERT INTO silver_cards VALUES (NULL, ?, ?, ?)",
            [scryfall_id, canonical_uuid, language],
        )


class TestBuildLanguagePrices:
    def test_returns_empty_when_silver_cards_missing(self, tmp_path):
        scryfall_hist = pd.DataFrame(
            {
                "id": ["s1-ja"],
                "snapshot_date": ["2026-05-11"],
                "prices": [_SCRYFALL_PRICES_JSON],
            }
        )
        with _make_storage_with_bronze(
            tmp_path, {"bronze_scryfall_prices_history": scryfall_hist}
        ) as s:
            result = s._prices.build_language_prices("2026-05-11")
            assert result.empty

    def test_returns_empty_when_no_language_variant_cards(self, tmp_path):
        scryfall_hist = pd.DataFrame(
            {
                "id": ["s1"],
                "snapshot_date": ["2026-05-11"],
                "prices": [_SCRYFALL_PRICES_JSON],
            }
        )
        with _make_storage_with_bronze(
            tmp_path,
            {
                "bronze_scryfall_prices_history": scryfall_hist,
                "bronze_scryfall_cards": pd.DataFrame({"id": ["s1"], "lang": ["en"]}),
            },
        ) as s:
            # Only English card (uuid is not NULL) — no language variants
            _seed_silver_cards(s, [("u1", "s1")])
            result = s._prices.build_language_prices("2026-05-11")
            assert result.empty

    def test_happy_path_language_variant_gets_prices(self, tmp_path):
        scryfall_hist = pd.DataFrame(
            {
                "id": ["s1-ja"],
                "snapshot_date": ["2026-05-11"],
                "prices": [_SCRYFALL_PRICES_JSON],
            }
        )
        with _make_storage_with_bronze(
            tmp_path,
            {
                "bronze_scryfall_prices_history": scryfall_hist,
                "bronze_scryfall_cards": pd.DataFrame(
                    {"id": ["s1-ja"], "lang": ["ja"]}
                ),
            },
        ) as s:
            _seed_silver_language_variant_cards(s, [("s1-ja", "u1", "Japanese")])
            result = s._prices.build_language_prices("2026-05-11")

            assert len(result) == 1
            row = result.iloc[0]
            assert row["scryfall_id"] == "s1-ja"
            assert row["canonical_uuid"] == "u1"
            assert row["lang"] == "ja"
            assert row["eur"] == pytest.approx(3.20)

    def test_has_expected_columns(self, tmp_path):
        scryfall_hist = pd.DataFrame(
            {
                "id": ["s1-ja"],
                "snapshot_date": ["2026-05-11"],
                "prices": [_SCRYFALL_PRICES_JSON],
            }
        )
        with _make_storage_with_bronze(
            tmp_path,
            {
                "bronze_scryfall_prices_history": scryfall_hist,
                "bronze_scryfall_cards": pd.DataFrame(
                    {"id": ["s1-ja"], "lang": ["ja"]}
                ),
            },
        ) as s:
            _seed_silver_language_variant_cards(s, [("s1-ja", "u1", "Japanese")])
            result = s._prices.build_language_prices("2026-05-11")

            expected = [
                "scryfall_id",
                "canonical_uuid",
                "lang",
                "snapshot_date",
                "eur",
                "eur_foil",
                "usd",
                "usd_foil",
            ]
            assert list(result.columns) == expected

    def test_english_card_scryfall_id_not_included(self, tmp_path):
        # English card has uuid NOT NULL — must not appear in language prices
        scryfall_hist = pd.DataFrame(
            {
                "id": ["s1-en", "s1-ja"],
                "snapshot_date": ["2026-05-11", "2026-05-11"],
                "prices": [_SCRYFALL_PRICES_JSON, _SCRYFALL_PRICES_JSON],
            }
        )
        with _make_storage_with_bronze(
            tmp_path,
            {
                "bronze_scryfall_prices_history": scryfall_hist,
                "bronze_scryfall_cards": pd.DataFrame(
                    {"id": ["s1-en", "s1-ja"], "lang": ["en", "ja"]}
                ),
            },
        ) as s:
            _seed_silver_cards(s, [("u1", "s1-en")])
            _seed_silver_language_variant_cards(s, [("s1-ja", "u1", "Japanese")])
            result = s._prices.build_language_prices("2026-05-11")

            assert len(result) == 1
            assert result.iloc[0]["scryfall_id"] == "s1-ja"


# ---------------------------------------------------------------------------
# SilverStorage._pipeline — oracle ID name conflict check (EDA-01 §7)
# ---------------------------------------------------------------------------


class TestOracleIdConflictCheck:
    def test_no_warning_when_all_names_have_unique_oracle_id(self, tmp_path, caplog):
        cards_mock = pd.DataFrame(
            {
                "scryfall_id": ["s1", "s2"],
                "name": ["CardA", "CardB"],
                "oracle_id": ["o1", "o2"],
            }
        )
        with _make_storage(tmp_path) as s:
            with (
                patch.object(s, "_join_cards", return_value=cards_mock),
                patch("src.data.cards.storage.silver.storage.write_report"),
                caplog.at_level(
                    logging.INFO,
                    logger="src.data.cards.storage.silver.storage",
                ),
            ):
                s.populate()

        warning_records = [
            r
            for r in caplog.records
            if r.levelno == logging.WARNING and "Oracle ID conflict" in r.message
        ]
        assert warning_records == []

    def test_warning_logged_when_name_maps_to_multiple_oracle_ids(
        self, tmp_path, caplog
    ):
        # "Fire // Ice" appears with two different oracle_ids — split card regression.
        cards_mock = pd.DataFrame(
            {
                "scryfall_id": ["s1", "s2"],
                "name": ["Fire // Ice", "Fire // Ice"],
                "oracle_id": ["o1", "o2"],
            }
        )
        with _make_storage(tmp_path) as s:
            with (
                patch.object(s, "_join_cards", return_value=cards_mock),
                patch("src.data.cards.storage.silver.storage.write_report"),
                caplog.at_level(
                    logging.WARNING,
                    logger="src.data.cards.storage.silver.storage",
                ),
            ):
                s.populate()

        warning_records = [
            r
            for r in caplog.records
            if r.levelno == logging.WARNING and "Oracle ID conflict" in r.message
        ]
        assert len(warning_records) == 1
        assert "Fire // Ice" in warning_records[0].message


# ---------------------------------------------------------------------------
# SilverWriter (silver/persistence.py)
# ---------------------------------------------------------------------------


@pytest.fixture
def silver_con():
    con = duckdb.connect(":memory:")
    yield con
    con.close()


class TestSilverWriterAppendLoad:
    def test_append_creates_table_on_first_call(self, silver_con):
        writer = SilverWriter(silver_con)
        df = pd.DataFrame(
            {"uuid": ["a"], "snapshot_date": ["2026-01-01"], "val": [1.0]}
        )
        writer.append(df, "hist", "uuid")
        count = silver_con.execute("SELECT count(*) FROM hist").fetchone()[0]
        assert count == 1

    def test_append_inserts_new_rows_into_existing_table(self, silver_con):
        writer = SilverWriter(silver_con)
        df1 = pd.DataFrame(
            {"uuid": ["a"], "snapshot_date": ["2026-01-01"], "val": [1.0]}
        )
        df2 = pd.DataFrame(
            {"uuid": ["b"], "snapshot_date": ["2026-01-02"], "val": [2.0]}
        )
        writer.append(df1, "hist", "uuid")
        writer.append(df2, "hist", "uuid")
        count = silver_con.execute("SELECT count(*) FROM hist").fetchone()[0]
        assert count == 2

    def test_append_skips_duplicate_key_snapshot_pair(self, silver_con):
        writer = SilverWriter(silver_con)
        df = pd.DataFrame(
            {"uuid": ["a"], "snapshot_date": ["2026-01-01"], "val": [1.0]}
        )
        writer.append(df, "hist", "uuid")
        writer.append(df, "hist", "uuid")  # same key + date — must be skipped
        count = silver_con.execute("SELECT count(*) FROM hist").fetchone()[0]
        assert count == 1

    def test_append_skips_empty_dataframe(self, silver_con):
        writer = SilverWriter(silver_con)
        empty = pd.DataFrame(
            {
                "uuid": pd.Series([], dtype=str),
                "snapshot_date": pd.Series([], dtype=str),
            }
        )
        writer.append(empty, "hist", "uuid")
        tables = silver_con.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_name='hist'"
        ).fetchall()
        assert tables == []

    def test_append_raises_storage_write_error_on_duckdb_error(self):
        from unittest.mock import MagicMock

        mock_con = MagicMock()
        # _table_exists → execute().fetchone() → (1,) means table exists
        table_exists_result = MagicMock()
        table_exists_result.fetchone.return_value = (1,)
        # second execute (the INSERT) raises duckdb.Error
        mock_con.execute.side_effect = [table_exists_result, duckdb.Error("fail")]
        writer = SilverWriter(mock_con)
        df = pd.DataFrame({"uuid": ["a"], "snapshot_date": ["2026-01-01"]})
        with pytest.raises(StorageWriteError):
            writer.append(df, "hist", "uuid")


class TestSilverWriterFullLoad:
    def test_full_load_creates_table(self, silver_con):
        writer = SilverWriter(silver_con)
        df = pd.DataFrame({"uuid": ["x"], "val": [5]})
        writer.full_load(df, "cards")
        count = silver_con.execute("SELECT count(*) FROM cards").fetchone()[0]
        assert count == 1

    def test_full_load_replaces_existing_table(self, silver_con):
        writer = SilverWriter(silver_con)
        df1 = pd.DataFrame({"uuid": ["x"], "val": [1]})
        df2 = pd.DataFrame({"uuid": ["y"], "val": [2]})
        writer.full_load(df1, "cards")
        writer.full_load(df2, "cards")
        result = silver_con.execute("SELECT uuid FROM cards").df()
        assert list(result["uuid"]) == ["y"]

    def test_full_load_skips_empty_dataframe(self, silver_con):
        writer = SilverWriter(silver_con)
        empty = pd.DataFrame({"uuid": pd.Series([], dtype=str)})
        writer.full_load(empty, "cards")
        tables = silver_con.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_name='cards'"
        ).fetchall()
        assert tables == []

    def test_full_load_raises_storage_write_error_on_duckdb_error(self):
        from unittest.mock import MagicMock

        mock_con = MagicMock()
        mock_con.execute.side_effect = duckdb.Error("fail")
        writer = SilverWriter(mock_con)
        df = pd.DataFrame({"uuid": ["x"]})
        with pytest.raises(StorageWriteError):
            writer.full_load(df, "cards")


class TestSilverWriterIncremental:
    def test_incremental_creates_table_on_first_call(self, silver_con):
        writer = SilverWriter(silver_con)
        df = pd.DataFrame({"uuid": ["a"], "val": [1]})
        writer.upsert(df, "cards", "uuid")
        count = silver_con.execute("SELECT count(*) FROM cards").fetchone()[0]
        assert count == 1

    def test_incremental_upserts_existing_key(self, silver_con):
        writer = SilverWriter(silver_con)
        df1 = pd.DataFrame({"uuid": ["a"], "val": [1]})
        writer.upsert(df1, "cards", "uuid")
        df2 = pd.DataFrame({"uuid": ["a"], "val": [99]})
        writer.upsert(df2, "cards", "uuid")
        result = silver_con.execute("SELECT val FROM cards WHERE uuid='a'").fetchone()[
            0
        ]
        assert result == 99

    def test_incremental_leaves_untouched_rows(self, silver_con):
        writer = SilverWriter(silver_con)
        df1 = pd.DataFrame({"uuid": ["a", "b"], "val": [1, 2]})
        writer.upsert(df1, "cards", "uuid")
        df2 = pd.DataFrame({"uuid": ["a"], "val": [10]})
        writer.upsert(df2, "cards", "uuid")
        result = silver_con.execute("SELECT val FROM cards WHERE uuid='b'").fetchone()[
            0
        ]
        assert result == 2

    def test_incremental_skips_empty_dataframe(self, silver_con):
        writer = SilverWriter(silver_con)
        empty = pd.DataFrame(
            {"uuid": pd.Series([], dtype=str), "val": pd.Series([], dtype=int)}
        )
        writer.upsert(empty, "cards", "uuid")
        tables = silver_con.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_name='cards'"
        ).fetchall()
        assert tables == []

    def test_incremental_raises_storage_write_error_on_duckdb_error(self):
        from unittest.mock import MagicMock

        mock_con = MagicMock()
        # _table_exists → execute().fetchone() → (1,) means table exists
        table_exists_result = MagicMock()
        table_exists_result.fetchone.return_value = (1,)
        # second execute (the DELETE) raises duckdb.Error
        mock_con.execute.side_effect = [table_exists_result, duckdb.Error("fail")]
        writer = SilverWriter(mock_con)
        df = pd.DataFrame({"uuid": ["a"], "val": [1]})
        with pytest.raises(StorageWriteError):
            writer.upsert(df, "cards", "uuid")


# ---------------------------------------------------------------------------
# SilverTransforms._extract_legality_features
# ---------------------------------------------------------------------------


class TestExtractLegalityFeatures:
    """SilverTransforms._extract_legality_features produces scalar columns."""

    @pytest.fixture
    def t(self) -> SilverTransforms:
        return SilverTransforms(
            language_map={},
            legality_map={},
            supertypes=[],
            card_types=[],
        )

    def test_extracts_is_commander_legal_true(self, t):
        df = pd.DataFrame({"legalities": [{"commander": "legal"}]})
        result = t._extract_legality_features(df, [])
        assert result["is_commander_legal"].iloc[0] is True

    def test_extracts_is_commander_legal_false(self, t):
        df = pd.DataFrame({"legalities": [{"commander": "not_legal"}]})
        result = t._extract_legality_features(df, [])
        assert result["is_commander_legal"].iloc[0] is False

    def test_extracts_all_four_format_columns(self, t):
        df = pd.DataFrame(
            {
                "legalities": [
                    {
                        "commander": "legal",
                        "standard": "legal",
                        "modern": "not_legal",
                        "legacy": "legal",
                    }
                ]
            }
        )
        result = t._extract_legality_features(df, [])
        assert result["is_commander_legal"].iloc[0] is True
        assert result["is_standard_legal"].iloc[0] is True
        assert result["is_modern_legal"].iloc[0] is False
        assert result["is_legacy_legal"].iloc[0] is True

    def test_format_count_sums_all_legal_formats(self, t):
        df = pd.DataFrame(
            {
                "legalities": [
                    {
                        "commander": "legal",
                        "modern": "legal",
                        "standard": "not_legal",
                        "vintage": "legal",
                    }
                ]
            }
        )
        result = t._extract_legality_features(df, [])
        assert result["format_count"].iloc[0] == 3

    def test_drops_legalities_column(self, t):
        df = pd.DataFrame({"legalities": [{"commander": "legal"}]})
        result = t._extract_legality_features(df, [])
        assert "legalities" not in result.columns

    def test_none_legalities_produces_false_and_zero(self, t):
        df = pd.DataFrame({"legalities": [None]})
        result = t._extract_legality_features(df, [])
        assert result["is_commander_legal"].iloc[0] is False
        assert result["format_count"].iloc[0] == 0

    def test_no_legalities_column_is_noop(self, t):
        df = pd.DataFrame({"name": ["Lightning Bolt"]})
        result = t._extract_legality_features(df, [])
        assert "is_commander_legal" not in result.columns
        assert list(result.columns) == ["name"]
