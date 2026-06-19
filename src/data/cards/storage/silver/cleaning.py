"""Stateless DataFrame cleaning steps for the Silver tier.

Each function follows the same contract:
    - Accepts a DataFrame and an issues accumulator list.
    - Appends issue dicts in place; never raises on bad input.
    - Returns the (possibly modified) DataFrame.

Typical usage:
    cleaned_df = _filter_rows(df, drop_entries, issues)
    cleaned_df = _drop_columns(cleaned_df, drop_columns, issues)
    cleaned_df = _parse_json_columns(cleaned_df, json_columns, issues)
"""

import json

import numpy as np
import pandas as pd

from src.logger import get_logger


logger = get_logger(__name__)


def _filter_rows(
    df: pd.DataFrame,
    drop_entries: dict[str, object],
    issues: list[dict[str, object]],
) -> pd.DataFrame:
    """Drop rows where a column equals the specified value (or any of several values).

    Appends a report entry for every column that is absent or has rows dropped.

    Args:
        df: Input DataFrame.
        drop_entries: Mapping of column name to the value (or list of values) that
            marks rows for removal.  When the value is a list, rows are dropped if
            the column value is any of the list items.
        issues: Accumulator for issue dicts — entries are appended in place.

    Returns:
        DataFrame with matching rows removed and index reset.
    """
    for column, value in drop_entries.items():
        if column not in df.columns:
            issues.append(
                {
                    "step": "filter_rows",
                    "column": column,
                    "issue": "column_not_found",
                }
            )
            continue
        if isinstance(value, list):
            drop_mask = df[column].isin(value).fillna(False)
        else:
            drop_mask = (df[column] == value).fillna(False)
        dropped = drop_mask.sum()
        if dropped > 0:
            issues.append(
                {
                    "step": "filter_rows",
                    "column": column,
                    "issue": "rows_dropped",
                    "count": int(dropped),
                    "value": value,
                }
            )
            logger.info("Dropped %d rows where %r == %r", int(dropped), column, value)
        df = df[~drop_mask]
    return df.reset_index(drop=True)


def _drop_columns(
    df: pd.DataFrame,
    drop_columns: list[str],
    issues: list[dict[str, object]],
) -> pd.DataFrame:
    """Drop columns by name, ignoring any that are not present.

    Appends a report entry for every requested column that does not exist.

    Args:
        df: Input DataFrame.
        drop_columns: Names of columns to remove.
        issues: Accumulator for issue dicts — entries are appended in place.

    Returns:
        DataFrame without the specified columns.
    """
    for column in drop_columns:
        if column not in df.columns:
            issues.append(
                {
                    "step": "drop_columns",
                    "column": column,
                    "issue": "column_not_found",
                }
            )
    return df.drop(columns=[c for c in drop_columns if c in df.columns])


def _parse_json_columns(
    df: pd.DataFrame,
    json_columns: list[str],
    issues: list[dict[str, object]],
) -> pd.DataFrame:
    """Parse JSON string cells back to Python lists or dicts.

    Cells that are not strings are left unchanged. Cells that fail to parse
    are also left as-is, and a report entry is added with example values.

    Args:
        df: Input DataFrame.
        json_columns: Names of columns whose string cells should be parsed as JSON.
        issues: Accumulator for issue dicts — entries are appended in place.

    Returns:
        DataFrame with JSON string cells replaced by their parsed Python objects.
    """
    for column in json_columns:
        if column not in df.columns:
            continue
        failures: list[str] = []

        def try_parse(x: object, _failures: list[str] = failures) -> object:
            if not isinstance(x, str):
                return x
            try:
                return json.loads(x)
            except json.JSONDecodeError:
                _failures.append(x)
                return x

        df[column] = df[column].map(try_parse)
        if failures:
            issues.append(
                {
                    "step": "parse_json_columns",
                    "column": column,
                    "issue": "json_parse_failed",
                    "count": len(failures),
                    "examples": list(set(failures))[:5],
                }
            )
            logger.warning(
                "Column %r: %d cells failed JSON parsing", column, len(failures)
            )
    return df


def _clean_strings(
    df: pd.DataFrame,
    string_operations: dict[str, list[str]],
    issues: list[dict[str, object]],
) -> pd.DataFrame:
    """Apply strip, upper, lower, title, or replace_sentinel to string columns.

    Supported operations (applied in declaration order per column):
        "strip"            — remove leading/trailing whitespace
        "upper"            — convert to upper case
        "lower"            — convert to lower case
        "title"            — convert to title case
        "replace_sentinel" — replace the MTGJson null sentinel "_" with None

    Appends a report entry for every column that is absent or has sentinel values replaced.

    Args:
        df: Input DataFrame.
        string_operations: Mapping of column name to a list of operation names.
        issues: Accumulator for issue dicts — entries are appended in place.

    Returns:
        DataFrame with string columns cleaned according to the specified operations.
    """
    for column, operations in string_operations.items():
        if column not in df.columns:
            issues.append(
                {
                    "step": "clean_strings",
                    "column": column,
                    "issue": "column_not_found",
                }
            )
            continue
        if "strip" in operations:
            df[column] = df[column].str.strip()
        if "upper" in operations:
            df[column] = df[column].str.upper()
        if "lower" in operations:
            df[column] = df[column].str.lower()
        if "title" in operations:
            df[column] = df[column].str.title()
        if "replace_sentinel" in operations:
            count = (df[column] == "_").sum()
            if count > 0:
                issues.append(
                    {
                        "step": "clean_strings",
                        "column": column,
                        "issue": "sentinel_replaced",
                        "count": int(count),
                    }
                )
            df[column] = df[column].replace("_", None)
    return df


def _clean_numerics(
    df: pd.DataFrame,
    numeric_columns: list[str],
    issues: list[dict[str, object]],
) -> pd.DataFrame:
    """Coerce columns to numeric, converting unparseable values to NaN.

    Appends a report entry for every column that contains values that could
    not be converted, including up to five example values.

    Args:
        df: Input DataFrame.
        numeric_columns: Names of columns that should hold numeric values.
        issues: Accumulator for issue dicts — entries are appended in place.

    Returns:
        DataFrame with the specified columns coerced to a numeric dtype.
    """
    for column in numeric_columns:
        if column not in df.columns:
            continue
        converted = pd.to_numeric(df[column], errors="coerce")
        failed = df[column][converted.isna() & df[column].notna()]
        if not failed.empty:
            issues.append(
                {
                    "step": "clean_numerics",
                    "column": column,
                    "issue": "unparseable_values",
                    "count": len(failed),
                    "examples": failed.unique().tolist()[:5],
                }
            )
            logger.warning(
                "Column %r: %d unparseable values → NaN: %s",
                column,
                len(failed),
                failed.unique().tolist(),
            )
        df[column] = converted
    return df


def _clean_lists(
    df: pd.DataFrame,
    list_operations: dict[str, list[str]],
    issues: list[dict[str, object]],
) -> pd.DataFrame:
    """Fill None list cells with [] and apply per-item case transforms.

    Supported operations:
        "fill_empty" — replace None/non-list values with an empty list
        "upper"      — convert every string item to upper case
        "lower"      — convert every string item to lower case
        "title"      — convert every string item to title case

    Appends a report entry for every column that has null values filled.

    Args:
        df: Input DataFrame.
        list_operations: Mapping of column name to a list of operation names.
        issues: Accumulator for issue dicts — entries are appended in place.

    Returns:
        DataFrame with list columns cleaned according to the specified operations.
    """
    for column, operations in list_operations.items():
        if column not in df.columns:
            continue
        null_count = df[column].isna().sum()
        if null_count > 0:
            issues.append(
                {
                    "step": "clean_lists",
                    "column": column,
                    "issue": "nulls_filled_with_empty_list",
                    "count": int(null_count),
                }
            )
        df[column] = df[column].map(
            lambda x: (
                x.tolist()
                if isinstance(x, np.ndarray)
                else (x if isinstance(x, list) else [])
            )
        )
        if "upper" in operations:
            df[column] = df[column].apply(
                lambda xs: [x.upper() for x in xs if isinstance(x, str)]
            )
        if "lower" in operations:
            df[column] = df[column].apply(
                lambda xs: [x.lower() for x in xs if isinstance(x, str)]
            )
        if "title" in operations:
            df[column] = df[column].apply(
                lambda xs: [x.title() for x in xs if isinstance(x, str)]
            )
    return df


def _clean_booleans(
    df: pd.DataFrame,
    bool_columns: list[str],
    issues: list[dict[str, object]],
) -> pd.DataFrame:
    """Fill None boolean cells with False and cast the column to bool.

    Appends a report entry for every column that has null values filled.

    Args:
        df: Input DataFrame.
        bool_columns: Names of columns that should hold boolean values.
        issues: Accumulator for issue dicts — entries are appended in place.

    Returns:
        DataFrame with the specified columns cast to bool, nulls replaced with False.
    """
    for column in bool_columns:
        if column not in df.columns:
            continue
        null_count = df[column].isna().sum()
        if null_count > 0:
            issues.append(
                {
                    "step": "clean_booleans",
                    "column": column,
                    "issue": "nulls_filled_with_false",
                    "count": int(null_count),
                }
            )
        df[column] = df[column].fillna(False).infer_objects(copy=False).astype(bool)
    return df


def _rename_columns(
    df: pd.DataFrame,
    rename_columns: dict[str, str],
    issues: list[dict[str, object]],
) -> pd.DataFrame:
    """Rename columns according to the mapping, silently skipping absent columns.

    Appends a report entry listing any source column names that were not found.

    Args:
        df: Input DataFrame.
        rename_columns: Mapping of current column name to desired column name.
        issues: Accumulator for issue dicts — entries are appended in place.

    Returns:
        DataFrame with columns renamed according to the mapping.
    """
    missing = [k for k in rename_columns if k not in df.columns]
    if missing:
        issues.append(
            {
                "step": "rename_columns",
                "issue": "columns_not_found",
                "columns": missing,
            }
        )
        logger.warning("Rename skipped for missing columns: %s", missing)
    return df.rename(
        columns={k: v for k, v in rename_columns.items() if k in df.columns}
    )
