"""DuckDB persistence layer for the Silver (cleaned) tier.

Exposes SilverStorage, a context-manager class that reads raw data from Bronze
DuckDB tables, applies config-driven transformations, and writes clean data to
Silver DuckDB tables.

Transformation rules for each source table are declared in silver_config.json —
adding a new source requires only a new config entry and no code changes.

Typical usage:
    with SilverStorage(
        "data/bronze/cards.duckdb",
        "data/silver/cards.duckdb",
        "configs/silver_config.json",
    ) as storage:
        storage.populate()   # initial load
        # or
        storage.update()     # incremental daily run
"""

import datetime
import json
from pathlib import Path

_SQL_DIR = Path(__file__).parent / "sql"

import duckdb

from src.data.cards.storage.base.storage import get_tables
from src.data.cards.storage.base.transformer import TransformStorage
from src.data.cards.storage.errors import StorageConnectionError, StorageWriteError
from src.data.cards.storage.base.writers import DuckDBWriter as SilverWriter
from src.data.cards.storage.silver.prices import SilverPriceBuilder
from src.logger import get_logger

logger = get_logger(__name__)

class SilverStorage(TransformStorage):
    """Persistence layer for the Silver (cleaned) tier.

    Inherits connection management and the populate()/update() entry points
    from TransformStorage. Reads raw DataFrames from Bronze DuckDB, applies a
    config-driven transformation pipeline (row filtering, type coercion,
    normalization, column renames), and writes the results to Silver DuckDB.

    Transformation rules for each source are declared in silver_config.json.
    Adding a new source requires only a new entry there — no code changes needed.

    Composition:
        _card_join  (SilverCardJoin)    — MTGJson × Scryfall merge logic
        _prices     (SilverPriceBuilder)— price extraction, join, and forward-fill
        _writer     (DuckDBWriter)      — DuckDB append / full-load / upsert helpers

    Usage:
        with SilverStorage(
            "data/bronze/cards.duckdb",
            "data/silver/cards.duckdb",
            "configs/silver_config.json",
        ) as storage:
            storage.populate()     # initial load or full rebuild
            storage.update()       # incremental daily run

    Raises:
        StorageConnectionError: If either DuckDB connection cannot be opened.
    """

    def __init__(
        self, bronze_db_path: str, silver_db_path: str, config_path: str
    ) -> None:
        """Open Bronze (read-only) and Silver (read-write) DuckDB connections.

        Args:
            bronze_db_path: Path to the Bronze DuckDB file.
            silver_db_path: Path to the Silver DuckDB file (created if it does not exist).
            config_path: Path to the silver_config.json file.

        Raises:
            StorageConnectionError: If either connection cannot be established.
        """
        self._bronze_db_path = bronze_db_path
        self._bronze_con = self._open_connection(bronze_db_path, read_only=True)
        self._silver_con = self._open_connection(silver_db_path, read_only=False)

        try:
            self._config = json.loads(Path(config_path).read_text())
        except FileNotFoundError:
            raise StorageConnectionError(
                f"Silver config not found: {config_path}"
            ) from None
        except json.JSONDecodeError as e:
            raise StorageConnectionError(
                f"Invalid JSON in silver config {config_path}: {e}"
            ) from e

        self._writer = SilverWriter(self._silver_con)
        self._prices = SilverPriceBuilder(self._bronze_con, self._silver_con)

    def close(self) -> None:
        """Close both Bronze and Silver DuckDB connections."""
        self._bronze_con.close()
        self._silver_con.close()
        logger.progress("Closed SilverStorage connections")

    # ------------------------------------------------------------------
    # History appenders
    # ------------------------------------------------------------------

    def _append_tournament_results_history(self) -> None:
        """Append tournament results from Bronze to silver_tournament_results_history.

        Reads bronze_tournament_results, normalises card_name, then joins to
        silver_cards on name to resolve oracle_id and one representative
        scryfall_id. Unmatched cards are kept with oracle_id = NULL so no
        Bronze data is silently lost.
        """
        bronze_tables = get_tables(self._bronze_con)
        if "bronze_tournament_results" not in bronze_tables:
            logger.warning(
                "bronze_tournament_results not found — skipping tournament results history"
            )
            return

        df = self._bronze_con.execute("SELECT * FROM bronze_tournament_results").df()
        if df.empty:
            logger.warning("bronze_tournament_results is empty — skipping")
            return

        df["card_name"] = (
            df["card_name"].str.strip().str.replace(" / ", " // ", regex=False)
        )

        silver_tables = get_tables(self._silver_con)
        if "silver_cards" in silver_tables:
            card_map = (
                self._silver_con.execute(
                    "SELECT DISTINCT name, oracle_id, scryfall_id "
                    "FROM silver_cards WHERE name IS NOT NULL"
                )
                .df()
                .drop_duplicates(subset=["name"], keep="first")
            )
            df = df.merge(card_map, left_on="card_name", right_on="name", how="left")
            df = df.drop(columns=["name"])
        else:
            logger.warning(
                "silver_cards not yet available — oracle_id/scryfall_id will be NULL"
            )
            df["oracle_id"] = None
            df["scryfall_id"] = None

        df["snapshot_date"] = df["tournament_date"]
        self._writer.append(df, "silver_tournament_results_history", key_column="id")

    def _append_format_staples_history(self) -> None:
        """Append today's format-staples snapshot from Bronze to silver_format_staples_history.

        Safe to call multiple times per day — duplicate (id, snapshot_date) pairs
        are skipped.
        """
        bronze_tables = get_tables(self._bronze_con)
        if "bronze_format_staples_history" not in bronze_tables:
            logger.warning(
                "bronze_format_staples_history not found — skipping format staples history"
            )
            return
        df = self._bronze_con.execute(
            "SELECT * FROM bronze_format_staples_history"
        ).df()
        self._writer.append(df, "silver_format_staples_history", key_column="id")

    def _build_silver_cards_sql(self) -> None:
        """Build silver_cards entirely in DuckDB SQL via ATTACH of the Bronze file.

        Executes _SILVER_CARDS_SQL: a multi-CTE CREATE OR REPLACE TABLE that filters,
        cleans, joins MTGJson × Scryfall, resolves canonical_uuid, deduplicates
        multi-face DFC rows, and extracts scalar legality columns — all in one
        DuckDB query. No pandas DataFrame is allocated.

        silver_cards is always fully rebuilt (CREATE OR REPLACE) because Scryfall
        delivers a complete daily snapshot and orphan rows from previous runs must
        not persist (identical semantics to the previous full_load call).
        """
        bronze_tables = get_tables(self._bronze_con)
        missing = [
            t
            for t in ("bronze_mtgjson_cards", "bronze_scryfall_cards")
            if t not in bronze_tables
        ]
        if missing:
            logger.warning(
                "Missing Bronze tables %s — skipping silver_cards build", missing
            )
            return
        try:
            self._silver_con.execute(
                f"ATTACH '{self._bronze_db_path}' AS _bronze (READ_ONLY)"
            )
            logger.progress("Building silver_cards (MTGJson × Scryfall SQL join — long step)...")
            self._silver_con.execute((_SQL_DIR / "silver_cards.sql").read_text(encoding="utf-8"))
            count = self._silver_con.execute(
                "SELECT count(*) FROM silver_cards"
            ).fetchone()
            logger.info(
                "Built silver_cards via SQL path: %d rows",
                count[0] if count else 0,
            )
        except duckdb.Error as e:
            logger.error("Failed to build silver_cards via SQL: %s", e)
            raise StorageWriteError(f"Failed to build silver_cards: {e}") from e
        finally:
            try:
                self._silver_con.execute("DETACH _bronze")
            except duckdb.Error:
                pass

    def _append_meta_history_sql(self) -> None:
        """Append Bronze scryfall_meta_history to Silver via DuckDB SQL.

        ATTACHes the Bronze file read-only, transforms with TRIM/TRY_CAST/COALESCE/
        lower, filters via INNER JOIN silver_cards (when available), and INSERTs via
        anti-join dedup — same contract as DuckDBWriter.append().

        legalities is passed through unchanged: Scryfall already stores values
        lowercase and Gold reads them via json_extract_string.
        """
        bronze_tables = get_tables(self._bronze_con)
        if "bronze_scryfall_meta_history" not in bronze_tables:
            logger.warning(
                "bronze_scryfall_meta_history not found — skipping silver_meta_history"
            )
            return

        silver_tables = get_tables(self._silver_con)
        has_silver_cards = "silver_cards" in silver_tables
        join_clause = (
            "INNER JOIN silver_cards sc ON sc.scryfall_id = b.id"
            if has_silver_cards
            else ""
        )
        if not has_silver_cards:
            logger.warning(
                "silver_cards not available — writing all meta_history rows unfiltered"
            )

        transform_sql = (
            (_SQL_DIR / "silver_meta_history_transform.sql")
            .read_text(encoding="utf-8")
            .format(join_clause=join_clause)
        )
        try:
            self._silver_con.execute(
                f"ATTACH '{self._bronze_db_path}' AS _bronze (READ_ONLY)"
            )
            if "silver_meta_history" not in silver_tables:
                self._silver_con.execute(
                    f"CREATE TABLE silver_meta_history AS {transform_sql}"
                )
                logger.info("Created silver_meta_history via SQL path")
            else:
                self._silver_con.execute(f"""
                    INSERT INTO silver_meta_history
                    SELECT src.*
                    FROM ({transform_sql}) src
                    LEFT JOIN silver_meta_history t
                        ON  t.id            = src.id
                        AND t.snapshot_date = src.snapshot_date
                    WHERE t.id IS NULL
                """)
                logger.info("Appended to silver_meta_history via SQL path")
        except duckdb.Error as e:
            logger.error("Failed to append silver_meta_history via SQL: %s", e)
            raise StorageWriteError(f"Failed to append silver_meta_history: {e}") from e
        finally:
            try:
                self._silver_con.execute("DETACH _bronze")
            except duckdb.Error:
                pass

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    def _check_oracle_id_conflicts(self) -> None:
        """Log a warning if any card name maps to more than one oracle_id.

        Signals a split-card handling regression — DFC faces should share one
        oracle_id, not create two. Runs as a pure SQL query on silver_cards.
        """
        silver_tables = get_tables(self._silver_con)
        if "silver_cards" not in silver_tables:
            return
        conflicts = self._silver_con.execute("""
            SELECT name, COUNT(DISTINCT oracle_id) AS n
            FROM silver_cards
            WHERE oracle_id IS NOT NULL AND name IS NOT NULL
            GROUP BY name
            HAVING COUNT(DISTINCT oracle_id) > 1
            LIMIT 5
        """).fetchall()
        if conflicts:
            logger.warning(
                "Oracle ID conflict check: %d name(s) map to multiple oracle_ids"
                " — split card handling may have regressed. Examples: %s",
                len(conflicts),
                [r[0] for r in conflicts],
            )
        else:
            logger.info("Oracle ID conflict check: 0 conflicts")

    def _pipeline(self, update: bool) -> None:
        """Run the full Bronze → Silver transformation pipeline via DuckDB SQL.

        All source transformations happen in DuckDB — no pandas DataFrames are
        allocated for Bronze data. SilverPriceBuilder is kept as-is (it already
        reads Silver tables via SQL; its MTGJson price parsing operates on a single
        day's rows and is too complex to express in SQL without UDFs).

        Args:
            update: Unused — silver_cards always does a full rebuild (Scryfall is a
                complete daily snapshot, so orphan rows from previous runs must not
                persist). History tables use append-style anti-join dedup regardless.
        """
        logger.progress("Step 1/6 — silver_cards SQL build")
        self._build_silver_cards_sql()
        self._check_oracle_id_conflicts()

        logger.progress("Step 2/6 — meta_history append")
        self._append_meta_history_sql()

        logger.progress("Step 3/6 — checkpoint")
        self._silver_con.execute("CHECKPOINT")

        today = datetime.date.today().isoformat()
        logger.progress("Step 4/6 — prices build")
        prices_df = self._prices.build(today)
        logger.debug(
            "silver_prices_history: %d price records for %s", len(prices_df), today
        )
        self._writer.append(
            prices_df, "silver_prices_history", key_column="scryfall_id"
        )

        logger.progress("Step 5/6 — language prices build")
        lang_prices_df = self._prices.build_language_prices(today)
        logger.debug(
            "silver_language_prices_history: %d language price records for %s",
            len(lang_prices_df),
            today,
        )
        self._writer.append(
            lang_prices_df, "silver_language_prices_history", key_column="scryfall_id"
        )

        logger.progress("Step 6/6 — format staples + tournament results")
        self._append_format_staples_history()
        self._append_tournament_results_history()
