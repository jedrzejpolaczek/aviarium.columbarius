"""Silver-tier price pipeline: extraction, joining, and forward-fill."""

from pathlib import Path

import duckdb
import pandas as pd

from src.data.cards.storage.base.storage import get_tables
from src.logger import get_logger


logger = get_logger(__name__)


class SilverPriceBuilder:
    """Builds today's silver_prices_history snapshot from Bronze sources."""

    # All price columns in the silver schema. Scryfall supplies the first four
    # (eur, eur_foil, usd, usd_foil); the rest come from MTGJson.
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
        "cardkingdom_usd",
        "cardkingdom_usd_foil",
        "cardkingdom_buylist_usd",
        "cardkingdom_buylist_usd_foil",
        "manapool_usd",
        "manapool_usd_foil",
    ]

    # Maps each silver column name to its (retailer, tx_type, finish) triple in
    # the Bronze EAV table. Used by the fallback path in _join_mtgjson_prices.
    _MTGJSON_PRICE_MAP: dict[str, tuple[str, str, str]] = {
        "cardmarket_eur": ("cardmarket", "retail", "normal"),
        "cardmarket_eur_foil": ("cardmarket", "retail", "foil"),
        "cardmarket_buylist_eur": ("cardmarket", "buylist", "normal"),
        "tcgplayer_usd": ("tcgplayer", "retail", "normal"),
        "tcgplayer_usd_foil": ("tcgplayer", "retail", "foil"),
        "tcgplayer_buylist_usd": ("tcgplayer", "buylist", "normal"),
        "cardkingdom_usd": ("cardkingdom", "retail", "normal"),
        "cardkingdom_usd_foil": ("cardkingdom", "retail", "foil"),
        "cardkingdom_buylist_usd": ("cardkingdom", "buylist", "normal"),
        "cardkingdom_buylist_usd_foil": ("cardkingdom", "buylist", "foil"),
        "manapool_usd": ("manapool", "retail", "normal"),
        "manapool_usd_foil": ("manapool", "retail", "foil"),
    }

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
            Reads today's rows from bronze_scryfall_prices_history, selecting
            scalar EUR/USD float columns directly (eur, eur_foil, usd, usd_foil),
            and joins to silver_cards on scryfall_id to resolve the canonical MTGJson uuid.
            UUID resolution uses COALESCE(uuid, canonical_uuid) so that English
            paper cards whose direct scryfall_id→MTGJson join missed (e.g. due to
            a stale identifier in MTGJson) are still included via their
            (set_code, collector_number)-resolved canonical_uuid. Cards with no
            resolvable UUID in silver_cards are dropped (inner join).

        MTGJson side:
            Reads today's EAV rows from bronze_mtgjson_prices_history and pivots
            them to wide columns via CASE WHEN aggregation. Left-joined to the
            Scryfall base on (uuid, snapshot_date).

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
            df[["uuid", "scryfall_id", "snapshot_date"] + self._PRICE_COLS],
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

        sql = (Path(__file__).parent / "sql" / "scryfall_prices_daily.sql").read_text()
        scryfall_prices = self._bronze_con.execute(sql, [today]).df()
        return scryfall_prices.merge(card_map, on="scryfall_id", how="inner")

    def _join_mtgjson_prices(
        self, df: pd.DataFrame, bronze_tables: set[str], today: str
    ) -> pd.DataFrame:
        """Join today's MTGJson EAV prices onto the Scryfall base DataFrame.

        Pivots EAV rows to wide columns via CASE WHEN aggregation in SQL,
        then left-joins to the Scryfall base on (uuid, snapshot_date).

        Args:
            df: Scryfall base DataFrame (uuid, scryfall_id, snapshot_date, …).
            bronze_tables: Set of table names present in Bronze DuckDB.
            today: ISO date string used to filter bronze_mtgjson_prices_history.
        """
        if "bronze_mtgjson_prices_history" not in bronze_tables:
            logger.warning(
                "bronze_mtgjson_prices_history not found — MTGJson prices omitted"
            )
            for col in self._MTGJSON_PRICE_MAP:
                df[col] = None
            return df

        sql = (Path(__file__).parent / "sql" / "mtgjson_prices_daily.sql").read_text()
        mtgjson = self._bronze_con.execute(sql, [today]).df()

        for col in self._MTGJSON_PRICE_MAP:
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
            df, today, "silver_prices_history", list(self._PRICE_COLS)
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

        sql = (Path(__file__).parent / "sql" / "scryfall_prices_daily.sql").read_text()
        scryfall_prices = self._bronze_con.execute(sql, [today]).df()

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


MTGJSON_PRICE_COMBOS: frozenset[tuple[str, str, str]] = frozenset(
    SilverPriceBuilder._MTGJSON_PRICE_MAP.values()
)
"""Public export of the (retailer, tx_type, finish) combinations Silver expects.

Consumed by health.py's schema-drift check. Kept as a module-level constant
(not a class attribute reach-through) so callers outside this module don't
depend on SilverPriceBuilder's internal naming. This is the sanctioned access
point for these combinations — new code needing them should import
MTGJSON_PRICE_COMBOS rather than reaching into
SilverPriceBuilder._MTGJSON_PRICE_MAP directly, even though Python won't stop
you from doing so.
"""
