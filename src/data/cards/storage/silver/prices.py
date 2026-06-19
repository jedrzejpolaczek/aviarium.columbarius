"""Silver-tier price pipeline: extraction, joining, and forward-fill."""

import json

import duckdb
import pandas as pd

from src.data.cards.storage.base import get_tables
from src.logger import get_logger


logger = get_logger(__name__)

_PRICE_COLS: list[str] = [
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

_MTGJSON_PRICE_MAP: dict[str, tuple[str, str, str]] = {
    "cardmarket_eur": ("cardmarket", "retail", "normal"),
    "cardmarket_eur_foil": ("cardmarket", "retail", "foil"),
    "cardmarket_buylist_eur": ("cardmarket", "buylist", "normal"),
    "tcgplayer_usd": ("tcgplayer", "retail", "normal"),
    "tcgplayer_usd_foil": ("tcgplayer", "retail", "foil"),
    "tcgplayer_buylist_usd": ("tcgplayer", "buylist", "normal"),
}


class SilverPriceBuilder:
    """Builds today's silver_prices_history snapshot from Bronze sources."""

    def __init__(
        self,
        bronze_con: duckdb.DuckDBPyConnection,
        silver_con: duckdb.DuckDBPyConnection,
    ) -> None:
        self._bronze_con = bronze_con
        self._silver_con = silver_con

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def build(self, today: str) -> pd.DataFrame:
        """Build today's price snapshot from Bronze Scryfall and MTGJson sources.

        Only today's rows are read from each Bronze history table, avoiding
        full-history scans on every daily run. Cards with all-NULL prices in
        today's snapshot are forward-filled from the most recent prior row in
        silver_prices_history.

        Scryfall side:
            Reads today's rows from bronze_scryfall_prices_history, extracts
            scalar EUR/USD fields from the stored JSON prices column, and joins
            to silver_cards on scryfall_id to resolve the canonical MTGJson uuid.
            UUID resolution uses COALESCE(uuid, canonical_uuid) so that English
            paper cards whose direct scryfall_id→MTGJson join missed (e.g. due to
            a stale identifier in MTGJson) are still included via their
            (set_code, collector_number)-resolved canonical_uuid. Cards with no
            resolvable UUID in silver_cards are dropped (inner join).

        MTGJson side:
            Reads today's rows from bronze_mtgjson_prices_history. Each row's
            JSON blob is parsed once and all six price columns are extracted in
            a single pass. Left-joined to the Scryfall base on (uuid, snapshot_date).

        Returns an empty DataFrame when silver_cards or
        bronze_scryfall_prices_history do not yet exist.

        Args:
            today: ISO date string (YYYY-MM-DD) for the snapshot being built.
        """
        silver_tables = get_tables(self._silver_con)
        if "silver_cards" not in silver_tables:
            logger.warning("silver_cards not yet available — skipping price build")
            return pd.DataFrame()

        bronze_tables = get_tables(self._bronze_con)
        if "bronze_scryfall_prices_history" not in bronze_tables:
            logger.warning(
                "bronze_scryfall_prices_history not found — skipping price build"
            )
            return pd.DataFrame()

        df = self._build_scryfall_base(today)
        df = self._join_mtgjson_prices(df, bronze_tables, today)
        return self._fill_price_history(
            df[
                [
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
            ],
            today,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_scryfall_base(self, today: str) -> pd.DataFrame:
        """Read today's Scryfall price snapshot and join to silver_cards for UUID.

        Filters bronze_scryfall_prices_history to snapshot_date = today so that
        only a single day's rows are processed instead of the full history table.

        UUID resolution: COALESCE(uuid, canonical_uuid) captures English paper
        cards whose scryfall_id→MTGJson direct join missed (uuid=NULL) but whose
        (set_code, collector_number) fallback resolved a canonical_uuid. The most
        common cause is MTGJson holding a stale scryfall_id after Scryfall
        reissues a card identifier. The language='English' guard prevents
        non-English language variants — which share the same canonical_uuid
        pattern — from creating duplicate price rows; their prices are tracked
        separately by build_language_prices.

        Args:
            today: ISO date string; only rows with this snapshot_date are read.
        """
        card_map = self._silver_con.execute(
            "SELECT COALESCE(uuid, canonical_uuid) AS uuid, scryfall_id"
            " FROM silver_cards"
            " WHERE scryfall_id IS NOT NULL"
            "   AND COALESCE(uuid, canonical_uuid) IS NOT NULL"
            "   AND (uuid IS NOT NULL OR language = 'English')"
        ).df()

        scryfall_prices = self._bronze_con.execute(
            """
            SELECT
                id                                                       AS scryfall_id,
                snapshot_date,
                CAST(json_extract_string(prices, '$.eur')      AS FLOAT) AS eur,
                CAST(json_extract_string(prices, '$.eur_foil') AS FLOAT) AS eur_foil,
                CAST(json_extract_string(prices, '$.usd')      AS FLOAT) AS usd,
                CAST(json_extract_string(prices, '$.usd_foil') AS FLOAT) AS usd_foil
            FROM bronze_scryfall_prices_history
            WHERE snapshot_date = ?
            """,
            [today],
        ).df()

        return scryfall_prices.merge(card_map, on="scryfall_id", how="inner")

    def _join_mtgjson_prices(
        self, df: pd.DataFrame, bronze_tables: set[str], today: str
    ) -> pd.DataFrame:
        """Join today's MTGJson paper prices onto the Scryfall base DataFrame.

        Reads only today's rows from bronze_mtgjson_prices_history. Each row's
        JSON blob is parsed once and all six price columns are extracted in a
        single pass, avoiding repeated json.loads calls per column.

        Args:
            df: Scryfall base DataFrame (uuid, scryfall_id, snapshot_date, …).
            bronze_tables: Set of table names present in Bronze DuckDB.
            today: ISO date string used to filter bronze_mtgjson_prices_history.
        """
        if "bronze_mtgjson_prices_history" not in bronze_tables:
            logger.warning(
                "bronze_mtgjson_prices_history not found — MTGJson prices omitted"
            )
            for col in _MTGJSON_PRICE_MAP:
                df[col] = None
            return df

        mtgjson = self._bronze_con.execute(
            "SELECT uuid, snapshot_date, paper "
            "FROM bronze_mtgjson_prices_history "
            "WHERE snapshot_date = ?",
            [today],
        ).df()

        extracted = pd.DataFrame(
            [
                self._extract_all_prices(paper, snap)
                for paper, snap in zip(mtgjson["paper"], mtgjson["snapshot_date"])
            ]
        )
        mtgjson = pd.concat([mtgjson.drop(columns=["paper"]), extracted], axis=1)
        # When bronze has no rows for today (e.g. pipeline runs before the daily
        # snapshot lands), extracted is empty and has no columns. Ensure the price
        # columns are present so the LEFT merge adds them as NULL rather than
        # silently dropping them.
        for col in _MTGJSON_PRICE_MAP:
            if col not in mtgjson.columns:
                mtgjson[col] = None
        return df.merge(mtgjson, on=["uuid", "snapshot_date"], how="left")

    def _fill_from_history(
        self,
        df: pd.DataFrame,
        today: str,
        history_table: str,
        candidate_price_cols: list[str],
    ) -> pd.DataFrame:
        """Forward-fill null price rows from the most recent prior snapshot.

        Args:
            df: Today's price DataFrame (must have a 'scryfall_id' column).
            today: ISO date string; only rows with snapshot_date < today are used.
            history_table: DuckDB table name to query for historical prices.
            candidate_price_cols: Price column names to fill; only those present
                in df are actually used.

        Returns:
            df with null-price rows filled from history, unchanged if the
            history table does not exist or no prior rows are found.
        """
        if df.empty:
            return df

        price_cols = [c for c in candidate_price_cols if c in df.columns]
        null_mask = df[price_cols].isna().all(axis=1)

        if not null_mask.any():
            return df

        silver_tables = get_tables(self._silver_con)
        if history_table not in silver_tables:
            return df

        price_select = ", ".join(price_cols)
        prev = self._silver_con.execute(
            f"SELECT scryfall_id, {price_select} "
            f"FROM {history_table} "
            f"WHERE snapshot_date < ? "
            f"QUALIFY ROW_NUMBER() OVER "
            f"  (PARTITION BY scryfall_id ORDER BY snapshot_date DESC) = 1",
            [today],
        ).df()

        if prev.empty:
            return df

        null_rows = df.loc[null_mask, ["scryfall_id"]].merge(
            prev, on="scryfall_id", how="left"
        )
        df = df.copy()
        df.loc[null_mask, price_cols] = null_rows[price_cols].values
        return df

    def _fill_price_history(self, df: pd.DataFrame, today: str) -> pd.DataFrame:
        """Forward-fill prices from the most recent prior silver snapshot.

        For cards where all price columns are NULL in today's snapshot, copies
        prices from the latest available row in silver_prices_history with
        snapshot_date < today. Cards with at least one non-NULL price today are
        left untouched. Cards with no prior silver record remain NULL.

        Args:
            df: Today's price DataFrame (one row per card).
            today: ISO date string; only rows with snapshot_date < today are
                   considered as fill sources.
        """
        return self._fill_from_history(
            df, today, "silver_prices_history", list(_PRICE_COLS)
        )

    # ------------------------------------------------------------------
    # Language variant prices
    # ------------------------------------------------------------------

    def build_language_prices(self, today: str) -> pd.DataFrame:
        """Build today's price snapshot for non-English language variant cards.

        Reads today's rows from bronze_scryfall_prices_history for Scryfall IDs
        that belong to language variants (uuid IS NULL, canonical_uuid IS NOT NULL
        in silver_cards). Returns one row per (scryfall_id, snapshot_date) with
        EUR/USD prices and the canonical_uuid link to the English printing.

        Forward-fills from the most recent prior silver_language_prices_history
        row for cards where all prices are NULL today.

        Returns an empty DataFrame when silver_cards or
        bronze_scryfall_prices_history are unavailable.

        Args:
            today: ISO date string (YYYY-MM-DD) for the snapshot being built.
        """
        silver_tables = get_tables(self._silver_con)
        if "silver_cards" not in silver_tables:
            logger.warning(
                "silver_cards not yet available — skipping language price build"
            )
            return pd.DataFrame()

        bronze_tables = get_tables(self._bronze_con)
        if "bronze_scryfall_prices_history" not in bronze_tables:
            logger.warning(
                "bronze_scryfall_prices_history not found — skipping language price build"
            )
            return pd.DataFrame()

        lang_map = self._silver_con.execute(
            "SELECT scryfall_id, canonical_uuid "
            "FROM silver_cards "
            "WHERE uuid IS NULL AND canonical_uuid IS NOT NULL AND scryfall_id IS NOT NULL"
        ).df()

        scryfall_langs = self._bronze_con.execute(
            "SELECT id AS scryfall_id, lang FROM bronze_scryfall_cards"
        ).df()

        lang_map = lang_map.merge(scryfall_langs, on="scryfall_id", how="left")

        if lang_map.empty:
            logger.info("No language variant cards in silver_cards — skipping")
            return pd.DataFrame()

        scryfall_prices = self._bronze_con.execute(
            """
            SELECT
                id                                                       AS scryfall_id,
                snapshot_date,
                CAST(json_extract_string(prices, '$.eur')      AS FLOAT) AS eur,
                CAST(json_extract_string(prices, '$.eur_foil') AS FLOAT) AS eur_foil,
                CAST(json_extract_string(prices, '$.usd')      AS FLOAT) AS usd,
                CAST(json_extract_string(prices, '$.usd_foil') AS FLOAT) AS usd_foil
            FROM bronze_scryfall_prices_history
            WHERE snapshot_date = ?
            """,
            [today],
        ).df()

        df = scryfall_prices.merge(lang_map, on="scryfall_id", how="inner")
        if df.empty:
            return pd.DataFrame()

        df = df[
            [
                "scryfall_id",
                "canonical_uuid",
                "lang",
                "snapshot_date",
                "eur",
                "eur_foil",
                "usd",
                "usd_foil",
            ]
        ]
        return self._fill_language_price_history(df, today)

    def _fill_language_price_history(
        self, df: pd.DataFrame, today: str
    ) -> pd.DataFrame:
        """Forward-fill language variant prices from the most recent prior snapshot.

        Same semantics as _fill_price_history but keyed on scryfall_id and
        reading from silver_language_prices_history.

        Args:
            df: Today's language variant price DataFrame.
            today: ISO date string; only rows with snapshot_date < today are
                   considered as fill sources.
        """
        return self._fill_from_history(
            df,
            today,
            "silver_language_prices_history",
            ["eur", "eur_foil", "usd", "usd_foil"],
        )

    @staticmethod
    def _extract_all_prices(
        paper_json: object, snapshot_date: str
    ) -> dict[str, float | None]:
        """Parse a MTGJson paper JSON blob and extract all price columns at once.

        Parses the JSON blob only once and extracts all six configured price
        columns in a single pass, avoiding redundant parsing.

        Args:
            paper_json: Raw JSON string from bronze_mtgjson_prices_history.paper.
            snapshot_date: ISO date string; only prices on or before this date are used.

        Returns:
            Dict mapping each column in _MTGJSON_PRICE_MAP to its value or None.
        """
        result: dict[str, float | None] = {col: None for col in _MTGJSON_PRICE_MAP}
        if not isinstance(paper_json, str):
            return result
        try:
            data = json.loads(paper_json)
            for col, (retailer, tx_type, finish) in _MTGJSON_PRICE_MAP.items():
                prices = ((data.get(retailer) or {}).get(tx_type) or {}).get(
                    finish
                ) or {}
                candidates = {k: v for k, v in prices.items() if k <= snapshot_date}
                result[col] = float(candidates[max(candidates)]) if candidates else None
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            pass
        return result
