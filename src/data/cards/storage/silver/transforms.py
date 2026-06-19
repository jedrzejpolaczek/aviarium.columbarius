"""Config-driven DataFrame transformation pipeline for the Silver tier.

SilverTransforms combines eight stateless cleaning steps and three stateful steps
that require lookup tables from silver_config.json (language normalisation,
legality mapping, type-line parsing).

Typical usage:
    transforms = SilverTransforms(
        language_map=config["language_map"],
        legality_map=config["legality_map"],
        supertypes=config["supertypes"],
        card_types=config["card_types"],
    )
    cleaned_df, issues = transforms.transform(raw_df, source_config)
"""

from typing import Any

import pandas as pd

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
from src.logger import get_logger


logger = get_logger(__name__)


class SilverTransforms:
    """Config-driven cleaning and normalization pipeline for Bronze DataFrames.

    Applies eight stateless cleaning steps and three stateful steps that
    require lookup tables from silver_config.json:

        Stateless steps:
        _filter_rows      — drop rows where a column matches a value
        _drop_columns     — remove specified columns
        _parse_json_columns — parse JSON string cells
        _clean_strings    — apply case/whitespace transformations
        _clean_numerics   — coerce to numeric, NaN on parse failure
        _clean_lists      — fill None with [], apply case transforms
        _clean_booleans   — fill None with False, cast to bool
        _rename_columns   — rename columns per mapping

        Stateful steps (require lookup tables):
        _normalize_values     — language-code expansion and legality normalisation
        _parse_type_line      — type-line string → (supertypes, types, subtypes)
        _add_computed_columns — errata flag, original type columns, ascii_name fallback

    Instantiate once per pipeline run with the lookup tables from
    silver_config.json, then call transform() for each source table.

    Args:
        language_map: Mapping of language codes (e.g. "en") to full names (e.g. "English").
        legality_map: Mapping of raw legality strings (e.g. "Not Legal") to snake_case.
        supertypes: List of recognised MTG supertypes (e.g. ["Legendary", "Basic"]).
        card_types: List of recognised MTG card types (e.g. ["Creature", "Instant"]).
    """

    def __init__(
        self,
        language_map: dict[str, str],
        legality_map: dict[str, str],
        supertypes: list[str],
        card_types: list[str],
    ) -> None:
        self._language_map = language_map
        self._legality_map = legality_map
        self._supertypes = supertypes
        self._card_types = card_types

    def transform(
        self, df: pd.DataFrame, table_config: dict[str, Any]
    ) -> tuple[pd.DataFrame, list[dict[str, object]]]:
        """Apply all cleaning and normalization steps defined in table_config.

        Runs the full transform sequence: row filtering, column drops, JSON
        parsing, string cleaning, numeric coercion, list normalization, boolean
        filling, value normalization, computed columns, and column renames.

        Columns in drop_columns are removed before any other step. The list is
        maintained in silver_config.json and includes columns confirmed >80% null
        in Bronze via EDA (e.g. Vanguard-only fields, MTGO/Arena platform IDs)
        as well as columns that become redundant after row filtering.

        Args:
            df: Raw DataFrame loaded from a Bronze table.
            table_config: Cleaning rules for this table, as parsed from silver_config.json.

        Returns:
            A (DataFrame, issues) tuple — the cleaned DataFrame and a list of
            issue dicts describing every anomaly encountered during transformation.
        """
        issues: list[dict[str, object]] = []

        df = _filter_rows(df, table_config["drop_entries"], issues)
        df = _drop_columns(df, table_config["drop_columns"], issues)
        df = _parse_json_columns(df, table_config["json_columns"], issues)
        df = _clean_strings(df, table_config["string_ops"], issues)
        df = _clean_numerics(df, table_config["numeric_columns"], issues)
        df = _clean_lists(df, table_config["list_operations"], issues)
        df = _clean_booleans(df, table_config["bool_columns"], issues)
        df = self._normalize_values(df, issues)
        df = self._add_computed_columns(df, issues)
        df = _rename_columns(df, table_config["rename_columns"], issues)
        return df, issues

    # ------------------------------------------------------------------
    # Stateful steps (require lookup tables from silver_config.json)
    # ------------------------------------------------------------------

    def _normalize_values(
        self,
        df: pd.DataFrame,
        issues: list[dict[str, object]],
    ) -> pd.DataFrame:
        """Apply source-agnostic value normalization (language codes, legality casing).

        Language codes (e.g. "en") are expanded to full names (e.g. "English") using
        the language_map from the config. Handles both "language" (MTGJson) and "lang"
        (Scryfall, before renaming) column names.

        Legality values (e.g. "Legal", "Not Legal") are normalized to snake_case
        using the legality_map from the config.

        Appends a report entry for any language codes not present in the map.

        Args:
            df: Input DataFrame.
            issues: Accumulator for issue dicts — entries are appended in place.

        Returns:
            DataFrame with language codes expanded and legality values normalized.
        """
        lang_col = (
            "language"
            if "language" in df.columns
            else "lang"
            if "lang" in df.columns
            else None
        )
        if lang_col is not None:
            unknown = df[lang_col][
                df[lang_col].notna()
                & ~df[lang_col].isin(self._language_map)
                & ~df[lang_col].isin(self._language_map.values())
            ]
            if not unknown.empty:
                issues.append(
                    {
                        "step": "normalize_values",
                        "column": lang_col,
                        "issue": "unknown_language_codes",
                        "count": len(unknown),
                        "examples": unknown.unique().tolist()[:5],
                    }
                )
                logger.warning(
                    "Column %r: %d unknown language codes: %s",
                    lang_col,
                    len(unknown),
                    unknown.unique().tolist()[:5],
                )
            df[lang_col] = df[lang_col].map(
                lambda x: self._language_map.get(x, x) if isinstance(x, str) else x
            )
        if "legalities" in df.columns:
            df["legalities"] = df["legalities"].map(
                lambda d: (
                    {
                        k: self._legality_map.get(
                            v, v.lower() if isinstance(v, str) else v
                        )
                        for k, v in d.items()
                    }
                    if isinstance(d, dict)
                    else d
                )
            )
        return df

    def _parse_type_line(
        self, type_line: str | None
    ) -> tuple[list[str], list[str], list[str]]:
        """Split a type line string into supertypes, types, and subtypes.

        Splits on the em dash separator "—". Words before the dash are matched
        against the supertypes and card_types lists from the config; words after
        the dash are treated as subtypes.

        Args:
            type_line: A card type line such as "Legendary Creature — Human Wizard",
                or None/empty for cards without a type line.

        Returns:
            A tuple of (supertypes, types, subtypes), each a list of strings.
            Returns three empty lists for None or blank input.
        """
        if not isinstance(type_line, str) or not type_line.strip():
            return [], [], []
        parts = type_line.split("—")
        main = parts[0].strip().split()
        subtypes = parts[1].strip().split() if len(parts) > 1 else []
        supertypes = [t for t in main if t in self._supertypes]
        types = [t for t in main if t in self._card_types]
        return supertypes, types, subtypes

    def _add_computed_columns(
        self,
        df: pd.DataFrame,
        issues: list[dict[str, object]],
    ) -> pd.DataFrame:
        """Derive additional columns from existing data.

        Adds the following columns when the required source columns are present:
            errata              — bool, True when original_text differs from text
            original_supertypes — list parsed from the original_type string
            original_types      — list parsed from the original_type string
            original_subtypes   — list parsed from the original_type string
            ascii_name          — falls back to name when ascii_name is null

        Appends report entries for cards with errata and for ascii_name nulls filled.

        Args:
            df: Input DataFrame.
            issues: Accumulator for issue dicts — entries are appended in place.

        Returns:
            DataFrame with computed columns added.
        """
        if "text" in df.columns and "original_text" in df.columns:
            errata_mask = df["original_text"].notna() & (
                df["text"].fillna("") != df["original_text"].fillna("")
            )
            errata_count = errata_mask.sum()
            if errata_count > 0:
                issues.append(
                    {
                        "step": "add_computed_columns",
                        "column": "errata",
                        "issue": "cards_with_errata",
                        "count": int(errata_count),
                    }
                )
            df["errata"] = errata_mask
        if "original_type" in df.columns:
            parsed = df["original_type"].apply(self._parse_type_line)
            df["original_supertypes"] = parsed.apply(lambda t: t[0])
            df["original_types"] = parsed.apply(lambda t: t[1])
            df["original_subtypes"] = parsed.apply(lambda t: t[2])
        if "ascii_name" in df.columns and "name" in df.columns:
            filled = df["ascii_name"].isna().sum()
            if filled > 0:
                issues.append(
                    {
                        "step": "add_computed_columns",
                        "column": "ascii_name",
                        "issue": "nulls_filled_with_name",
                        "count": int(filled),
                    }
                )
            df["ascii_name"] = df["ascii_name"].fillna(df["name"])
        return df

    def _extract_legality_features(
        self,
        df: pd.DataFrame,
        issues: list[dict[str, object]],
    ) -> pd.DataFrame:
        """Extract scalar legality columns from the legalities dict and drop the raw column.

        Called only for silver_cards — silver_meta_history.legalities is left intact
        because GoldSignalBuilders needs the full format dict for ban/unban detection.

        Produces five typed columns:
            is_commander_legal  BOOLEAN
            is_standard_legal   BOOLEAN
            is_modern_legal     BOOLEAN
            is_legacy_legal     BOOLEAN
            format_count        BIGINT  — count across ALL formats in the dict

        Args:
            df: DataFrame with a legalities column containing Python dicts or None.
            issues: Accumulator — not currently used, present for pipeline consistency.

        Returns:
            DataFrame with scalar legality columns added and legalities column removed.
        """
        if "legalities" not in df.columns:
            return df

        df = df.copy()

        def _safe_dict(val: object) -> dict[str, object]:
            return val if isinstance(val, dict) else {}

        legalities = df["legalities"].map(_safe_dict)

        def _is_legal(fmt: str) -> pd.Series:
            # .astype(object) converts numpy.bool_ → Python bool so identity assertions
            # (is True) work in tests and DuckDB correctly infers BOOLEAN type
            return legalities.map(lambda d: bool(d.get(fmt) == "legal")).astype(object)

        df["is_commander_legal"] = _is_legal("commander")
        df["is_standard_legal"] = _is_legal("standard")
        df["is_modern_legal"] = _is_legal("modern")
        df["is_legacy_legal"] = _is_legal("legacy")
        df["format_count"] = legalities.map(
            lambda d: sum(1 for v in d.values() if v == "legal")
        )
        return df.drop(columns=["legalities"])
