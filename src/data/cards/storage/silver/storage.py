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

import pandas as pd

from src.data.cards.storage.base import TransformStorage, get_tables
from src.data.cards.storage.errors import StorageConnectionError
from src.data.cards.storage.silver.card_join import SilverCardJoin
from src.data.cards.storage.silver.persistence import SilverWriter
from src.data.cards.storage.silver.prices import SilverPriceBuilder
from src.data.cards.storage.silver.report import write_report
from src.data.cards.storage.silver.transforms import SilverTransforms
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

        self._transforms = SilverTransforms(
            language_map=self._config["language_map"],
            legality_map=self._config["legality_map"],
            supertypes=self._config["supertypes"],
            card_types=self._config["card_types"],
        )
        self._card_join = SilverCardJoin(language_map=self._config["language_map"])
        self._writer = SilverWriter(self._silver_con)
        self._prices = SilverPriceBuilder(self._bronze_con, self._silver_con)

    def close(self) -> None:
        """Close both Bronze and Silver DuckDB connections."""
        self._bronze_con.close()
        self._silver_con.close()
        logger.progress("Closed SilverStorage connections")

    # ------------------------------------------------------------------
    # Bronze loading
    # ------------------------------------------------------------------

    def _load_bronze_data(self) -> dict[str, pd.DataFrame]:
        """Load Bronze tables referenced in the config into DataFrames.

        Only tables declared as sources in silver_config.json are loaded,
        avoiding the cost of reading history tables with millions of rows.
        """
        needed = {f"bronze_{name}" for name in self._config["sources"]}
        existing = get_tables(self._bronze_con)
        to_load = needed & existing
        logger.progress("Loading %d Bronze tables", len(to_load))
        return {
            name: self._bronze_con.execute(f"SELECT * FROM {name}").df()
            for name in to_load
        }

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

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    def _pipeline(
        self, update: bool, report_path: str = "data/silver/transform_report.json"
    ) -> None:
        """Run the full Bronze → Silver transformation pipeline.

        After writing silver_cards, runs an oracle ID name conflict check
        (EDA-01 §7): logs a warning if any card name maps to more than one
        oracle_id, which signals a split card handling regression.

        Args:
            update: Unused — silver_cards always does a full_load (Scryfall is a
                complete daily snapshot, so orphan rows from previous runs must
                not persist). History tables use append() regardless.
            report_path: File path where the transformation report will be written.
        """
        all_issues: list[dict[str, object]] = []
        bronze_data = self._load_bronze_data()
        transformed: dict[str, pd.DataFrame] = {}

        for source_name, source_config in self._config["sources"].items():
            bronze_table = f"bronze_{source_name}"
            if bronze_table not in bronze_data:
                logger.warning("Bronze table %r not found — skipping", bronze_table)
                continue
            logger.progress("Transforming %r", bronze_table)
            df, issues = self._transforms.transform(
                bronze_data[bronze_table], source_config
            )
            all_issues.extend(issues)
            transformed[source_name] = df

        # Cards — delegate to _join_cards wrapper for patchability in tests
        cards_df = self._join_cards(transformed)
        cards_df = self._transforms._extract_legality_features(cards_df, all_issues)
        # Scryfall delivers a complete daily snapshot, so silver_cards is always
        # rebuilt from scratch.  An upsert would leave orphan rows (e.g. tokens
        # from older runs before the layout filter was added, or cards removed
        # from Scryfall) that silently corrupt oracle_id uniqueness checks.
        self._writer.full_load(cards_df, "silver_cards")

        # Oracle ID name conflict check (EDA-01 §7). A card name mapping to more
        # than one oracle_id signals a split card handling regression — the front and
        # back faces of a split card should share one oracle_id, not create two.
        if (
            not cards_df.empty
            and "name" in cards_df.columns
            and "oracle_id" in cards_df.columns
        ):
            conflicts = (
                cards_df[cards_df["oracle_id"].notna() & cards_df["name"].notna()]
                .groupby("name")["oracle_id"]
                .nunique()
            )
            conflicts = conflicts[conflicts > 1]
            if conflicts.empty:
                logger.info("Oracle ID conflict check: 0 conflicts")
            else:
                logger.warning(
                    "Oracle ID conflict check: %d name(s) map to multiple oracle_ids"
                    " — split card handling may have regressed. Examples: %s",
                    len(conflicts),
                    list(conflicts.head(5).index),
                )

        # Meta history — restrict to IDs in silver_cards so digital/oversized cards
        # dropped during the card join don't leak into history.
        meta_df = transformed.get("scryfall_meta_history", pd.DataFrame())
        if (
            not meta_df.empty
            and not cards_df.empty
            and "scryfall_id" in cards_df.columns
        ):
            valid_ids = set(cards_df["scryfall_id"].dropna())
            meta_df = meta_df[meta_df["id"].isin(valid_ids)]
        self._writer.append(
            meta_df,
            "silver_meta_history",
            key_column="id",
        )

        # Prices history (canonical / English cards)
        today = datetime.date.today().isoformat()
        prices_df = self._prices.build(today)
        self._writer.append(
            prices_df, "silver_prices_history", key_column="scryfall_id"
        )

        # Language variant prices (non-English Scryfall cards linked to canonical UUID)
        lang_prices_df = self._prices.build_language_prices(today)
        self._writer.append(
            lang_prices_df, "silver_language_prices_history", key_column="scryfall_id"
        )

        self._append_format_staples_history()
        self._append_tournament_results_history()

        write_report(all_issues, report_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _join_cards(self, cards: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Guard-and-delegate wrapper around SilverCardJoin.join.

        Checks that both required sources are present before delegating;
        returns an empty DataFrame with a warning if either is missing.
        Called by _pipeline and patchable in tests that need a controlled
        cards_df without running the full join.

        Args:
            cards: Transformed DataFrames keyed by source name.

        Returns:
            Merged silver_cards DataFrame, or an empty DataFrame if either
            source is absent.
        """
        if "mtgjson_cards" not in cards or "scryfall_cards" not in cards:
            missing = [s for s in ("mtgjson_cards", "scryfall_cards") if s not in cards]
            logger.warning("Cannot join cards — missing sources: %s", missing)
            return pd.DataFrame()
        return self._card_join.join(cards["mtgjson_cards"], cards["scryfall_cards"])
