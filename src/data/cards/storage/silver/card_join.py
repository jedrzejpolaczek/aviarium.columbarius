"""MTGJson × Scryfall card join logic for the Silver tier."""

import pandas as pd

from src.logger import get_logger

logger = get_logger(__name__)


class SilverCardJoin:
    """Merges transformed MTGJson and Scryfall card DataFrames into silver_cards."""

    # Maps MTGJson column names to the equivalent Scryfall column used as a fallback
    # for Scryfall-only rows (language variants, promos, etc. with no MTGJson match).
    # Applied after the outer join: if the MTGJson column is NaN, the Scryfall value fills it.
    _SCRYFALL_FALLBACK_MAP: dict[str, str] = {
        "name": "name",
        "layout": "layout",
        "mana_cost": "mana_cost",
        "mana_value": "cmc",
        "text": "oracle_text",
        "colors": "colors",
        "color_identity": "color_identity",
        "color_indicator": "color_indicator",
        "keywords": "keywords",
        "produced_mana": "produced_mana",
        "power": "power",
        "toughness": "toughness",
        "loyalty": "loyalty",
        "defense": "defense",
        "legalities": "legalities",
        "rarity": "rarity",
        "border_color": "border_color",
        "frame_version": "frame",
        "frame_effects": "frame_effects",
        "finishes": "finishes",
        "watermark": "watermark",
        "security_stamp": "security_stamp",
        "flavor_text": "flavor_text",
        "flavor_name": "flavor_name",
        "artist": "artist",
        "set_code": "set",
        "number": "collector_number",
        "is_reprint": "reprint",
        "is_reserved": "reserved",
        "is_promo": "promo",
        "is_full_art": "full_art",
        "is_textless": "textless",
        "is_story_spotlight": "story_spotlight",
    }

    def __init__(self, language_map: dict[str, str]) -> None:
        self._language_map = language_map

    def join(self, mtgjson: pd.DataFrame, scryfall: pd.DataFrame) -> pd.DataFrame:
        """Build the unified silver_cards DataFrame.

        Outer-joins the two sources on scryfall_id, applies Scryfall fallback
        values for MTGJson-only columns on Scryfall-only rows, resolves the
        language code using the configured language_map, and deduplicates
        multi-faced card rows (DFC/split/adventure) to one row per scryfall_id.

        Args:
            mtgjson: Transformed mtgjson_cards DataFrame.
            scryfall: Transformed scryfall_cards DataFrame.

        Returns:
            Merged DataFrame ready to be persisted as silver_cards.
        """
        logger.progress(
            "MTGJson rows: %d, Scryfall rows: %d", len(mtgjson), len(scryfall)
        )

        mtgjson = self._extract_scryfall_id(mtgjson)

        scryfall_only_cols = [
            col
            for col in scryfall.columns
            if col not in mtgjson.columns and col != "id"
        ]
        scryfall_subset = scryfall[["id"] + scryfall_only_cols]

        logger.progress(
            "Joining cards: %d MTGJson cols + %d Scryfall-only cols",
            len(mtgjson.columns),
            len(scryfall_only_cols),
        )

        result = self._outer_join(mtgjson, scryfall_subset)

        scryfall_only_mask = ~result["has_mtgjson_data"]
        scryfall_lookup = scryfall.set_index("id")

        result = self._apply_scryfall_fallbacks(
            result, scryfall_only_mask, scryfall_lookup
        )
        result = result.drop(columns=["cmc"], errors="ignore")
        result = self._resolve_canonical_uuids(
            result, scryfall_only_mask, scryfall_lookup
        )
        result = self._dedup_multiface_rows(result)

        logger.info(
            "Merged result rows: %d (%d MTGJson, %d Scryfall-only, %d with canonical_uuid)",
            len(result),
            result["has_mtgjson_data"].sum(),
            (~result["has_mtgjson_data"]).sum(),
            result["canonical_uuid"].notna().sum(),
        )
        return result

    def _extract_scryfall_id(self, mtgjson: pd.DataFrame) -> pd.DataFrame:
        """Extract scryfall_id from the identifiers dict and add it as a column.

        Args:
            mtgjson: MTGJson cards DataFrame with an ``identifiers`` column
                containing dicts (e.g. ``{"scryfall_id": "abc..."}``.

        Returns:
            Copy of *mtgjson* with a new ``scryfall_id`` column (str or NaN).
        """
        mtgjson = mtgjson.copy()
        mtgjson["scryfall_id"] = mtgjson["identifiers"].apply(
            lambda x: x.get("scryfall_id") if isinstance(x, dict) else None
        )
        logger.progress(
            "scryfall_id nulls: %d / %d",
            mtgjson["scryfall_id"].isna().sum(),
            len(mtgjson),
        )
        return mtgjson

    def _outer_join(
        self, mtgjson: pd.DataFrame, scryfall_subset: pd.DataFrame
    ) -> pd.DataFrame:
        """Outer-join MTGJson and Scryfall subset, fix scryfall_id for Scryfall-only rows.

        Uses outer (not inner/left) join to preserve two classes of orphan rows:
        - MTGJson-only: DFC back faces share a scryfall_id with the front face but
          have no Scryfall counterpart; a left join would silently drop ~10% of rows.
        - Scryfall-only: digital exclusives, oversized cards, and promos absent from
          MTGJson; kept so silver_cards is a superset of both sources.

        Args:
            mtgjson: MTGJson cards DataFrame with a ``scryfall_id`` column
                (added by :meth:`_extract_scryfall_id`).
            scryfall_subset: Scryfall DataFrame containing only ``id`` and
                columns not already present in *mtgjson*.

        Returns:
            Merged DataFrame with ``scryfall_id`` fixed for Scryfall-only rows
            and a boolean ``has_mtgjson_data`` column.
        """
        result = pd.merge(
            mtgjson, scryfall_subset, left_on="scryfall_id", right_on="id", how="outer"
        )
        result["scryfall_id"] = result["scryfall_id"].fillna(result["id"])
        result = result.drop(columns=["id"])
        result["has_mtgjson_data"] = result["uuid"].notna()
        return result

    def _apply_scryfall_fallbacks(
        self,
        result: pd.DataFrame,
        scryfall_only_mask: pd.Series,
        scryfall_lookup: pd.DataFrame,
    ) -> pd.DataFrame:
        """Fill MTGJson columns for Scryfall-only rows using equivalent Scryfall values.

        Iterates over _SCRYFALL_FALLBACK_MAP and, for each pair, backfills NaN
        values in the MTGJson column using the mapped Scryfall column.  Also
        resolves the language name for Scryfall-only rows via _language_map.

        Args:
            result: Merged DataFrame produced by :meth:`_outer_join`.
            scryfall_only_mask: Boolean Series (index aligned with *result*)
                that is ``True`` for Scryfall-only rows (``has_mtgjson_data == False``).
            scryfall_lookup: Full Scryfall DataFrame indexed on ``id``, used to
                look up fallback values by scryfall_id.

        Returns:
            *result* with NaN values filled in-place for Scryfall-only rows.
        """
        for mtgjson_col, scryfall_col in self._SCRYFALL_FALLBACK_MAP.items():
            if (
                mtgjson_col not in result.columns
                or scryfall_col not in scryfall_lookup.columns
            ):
                continue
            fallback = result.loc[scryfall_only_mask, "scryfall_id"].map(
                scryfall_lookup[scryfall_col]
            )
            result.loc[scryfall_only_mask, mtgjson_col] = result.loc[
                scryfall_only_mask, mtgjson_col
            ].fillna(fallback)

        if "language" in result.columns and "lang" in result.columns:
            result.loc[scryfall_only_mask, "language"] = result.loc[
                scryfall_only_mask, "lang"
            ].map(self._language_map)

        return result

    def _resolve_canonical_uuids(
        self,
        result: pd.DataFrame,
        scryfall_only_mask: pd.Series,
        scryfall_lookup: pd.DataFrame,
    ) -> pd.DataFrame:
        """Add canonical_uuid: uuid for MTGJson rows, resolved UUID for Scryfall-only rows.

        For MTGJson-matched rows canonical_uuid equals uuid.  For Scryfall-only rows
        (non-English language variants, promos without an MTGJson counterpart) it is
        resolved via (set_code, collector_number) to the UUID of the English canonical
        printing in the same set.  This link is needed to compute language premiums in
        the Gold layer.

        Scryfall-only rows have set_code=NULL in the result because set_code is shared
        between both sources and excluded from scryfall_subset.  set_code and
        collector_number are therefore looked up from scryfall_lookup, which retains
        the full Scryfall DataFrame indexed on id.  NULL canonical_uuid means no
        matching English UUID (e.g. digital-exclusive sets entirely absent from MTGJson).

        Args:
            result: Merged DataFrame after fallback filling (from
                :meth:`_apply_scryfall_fallbacks`).
            scryfall_only_mask: Boolean Series (index aligned with *result*)
                that is ``True`` for Scryfall-only rows.
            scryfall_lookup: Full Scryfall DataFrame indexed on ``id``, used to
                retrieve ``set_code`` and ``collector_number`` for Scryfall-only rows.

        Returns:
            *result* with a new ``canonical_uuid`` column.
        """
        key_cols = {"set_code", "collector_number"}
        if not (
            key_cols.issubset(result.columns)
            and key_cols.issubset(scryfall_lookup.columns)
        ):
            result["canonical_uuid"] = result["uuid"]
            return result

        en_lookup = (
            result[result["uuid"].notna()][["set_code", "collector_number", "uuid"]]
            .dropna(subset=["set_code", "collector_number"])
            .drop_duplicates(subset=["set_code", "collector_number"], keep="first")
            .set_index(["set_code", "collector_number"])["uuid"]
        )

        def _resolve_canonical(scryfall_id: object) -> object:
            if not isinstance(scryfall_id, str):
                return None
            row = (
                scryfall_lookup.loc[scryfall_id]
                if scryfall_id in scryfall_lookup.index
                else None
            )
            if row is None:
                return None
            return en_lookup.get((row["set_code"], row["collector_number"]))

        canonical_for_variants = result.loc[scryfall_only_mask, "scryfall_id"].map(
            _resolve_canonical
        )
        result["canonical_uuid"] = result["uuid"].where(
            result["uuid"].notna(),
            other=canonical_for_variants.reindex(result.index),
        )
        return result

    def _dedup_multiface_rows(self, result: pd.DataFrame) -> pd.DataFrame:
        """Drop duplicate scryfall_id rows from multi-faced cards (DFC/split/adventure).

        MTGJson stores each face as a separate row, all sharing the same
        scryfall_id.  After the outer join those faces produce N rows per
        scryfall_id; only the first (front face / side-a) is kept to match
        Scryfall's one-row-per-card model.  canonical_uuid is resolved from the
        full pre-dedup result, so this drop is safe.

        Args:
            result: Merged DataFrame with ``canonical_uuid`` already set (from
                :meth:`_resolve_canonical_uuids`).

        Returns:
            *result* deduplicated on ``scryfall_id``, keeping the first occurrence.
        """
        pre_dedup = len(result)
        result = result.drop_duplicates(subset=["scryfall_id"], keep="first")
        dropped = pre_dedup - len(result)
        if dropped:
            logger.info(
                "Deduped %d multi-face rows (DFC/split/adventure) on scryfall_id",
                dropped,
            )
        return result
