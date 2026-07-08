"""DuckDB persistence layer for the Gold (aggregated) tier.

Exposes GoldStorage, a context-manager class that reads clean Silver DuckDB
tables, delegates feature and signal computation to specialised builder classes,
and writes results to Gold DuckDB tables.

Both populate() and update() perform a full rebuild of all Gold tables because
window-function features (moving averages, lag-based deltas) span the entire
price history and cannot be incrementally patched.

Typical usage:
    with GoldStorage(
        "data/silver/cards.duckdb",
        "data/gold/cards.duckdb",
    ) as storage:
        storage.populate()   # initial load
        # or
        storage.update()     # incremental daily run (same as populate for Gold)

Composition:
    _features  (GoldFeatureBuilders) — gold_card_features, gold_price_features
    _signals   (GoldSignalBuilders)  — gold_demand_signals, gold_events,
                                       gold_format_staples, gold_ban_price_impact,
                                       gold_tournament_signals
    _writer    (DuckDBWriter)         — DROP-AND-REPLACE write helper
"""

from collections.abc import Callable

import duckdb
import pandas as pd

from src.data.cards.storage.base.storage import get_tables, warn_if_missing
from src.data.cards.storage.base.transformer import TransformStorage
from src.data.cards.storage.gold.features import GoldFeatureBuilders
from src.data.cards.storage.gold.signals import GoldSignalBuilders
from src.data.cards.storage.base.writers import DuckDBWriter as GoldWriter
from src.data.cards.storage.gold.ml_dataset import GoldMLDatasetBuilder
from src.logger import get_logger

logger = get_logger(__name__)

# Minimum date span (inclusive) between the earliest and latest snapshot in
# gold_price_features required before building gold_ml_dataset. With fewer
# than 7 days all target_price_7d values are NULL (no t+7 pair exists) and
# the training frame has zero usable rows. Matches validation_config.json
# min_train_days intent; the hard floor is the prediction horizon itself.
_MIN_ML_HORIZON_DAYS = 7


def get_latest_gold_snapshot_date(con: duckdb.DuckDBPyConnection) -> str | None:
    """Return the latest snapshot_date in gold_price_features, or None if empty/absent.

    Shared by app/main.py, scripts/train_model.py, and scripts/check_and_retrain.py,
    each of which need the latest available price snapshot from a raw DuckDB
    connection (not a GoldStorage instance) and previously ran this query inline.
    """
    if "gold_price_features" not in get_tables(con):
        return None
    row = con.execute("SELECT MAX(snapshot_date) FROM gold_price_features").fetchone()
    if row is None or row[0] is None:
        return None
    return str(row[0])


class GoldStorage(TransformStorage):
    """Persistence layer for the Gold (aggregated) tier.

    Reads clean Silver data, builds aggregated views and derived feature tables
    via GoldFeatureBuilders and GoldSignalBuilders, and writes results to Gold
    DuckDB using DuckDBWriter.

    Because all Gold features rely on window functions over the full history,
    populate() and update() are equivalent — both trigger a complete rebuild.

    Usage:
        with GoldStorage(
            "data/silver/cards.duckdb",
            "data/gold/cards.duckdb",
        ) as storage:
            storage.populate()

    Raises:
        StorageConnectionError: If either DuckDB connection cannot be opened.
    """

    # Every Gold table name any builder in this class can currently produce.
    # _prune_orphaned_tables uses this to distinguish "no builder makes this
    # anymore" (safe to drop) from "this run's Silver tier is partial" (must
    # stay untouched — see _pipeline's docstring on partial-tier tolerance).
    _KNOWN_GOLD_TABLES = frozenset(
        {
            "gold_card_features",
            "gold_price_features",
            "gold_language_premiums",
            "gold_demand_signals",
            "gold_events",
            "gold_format_staples",
            "gold_ban_price_impact",
            "gold_tournament_signals",
            "gold_ml_dataset",
        }
    )

    def __init__(self, silver_db_path: str, gold_db_path: str) -> None:
        """Open Silver (read-only) and Gold (read-write) DuckDB connections.

        Args:
            silver_db_path: Path to the Silver DuckDB file.
            gold_db_path: Path to the Gold DuckDB file (created if it does not exist).

        Raises:
            StorageConnectionError: If either connection cannot be established.
        """
        self._silver_con = self._open_connection(silver_db_path, read_only=True)
        self._gold_con = self._open_connection(gold_db_path, read_only=False)

        self._features = GoldFeatureBuilders(self._silver_con)
        self._signals = GoldSignalBuilders(self._silver_con)
        self._writer = GoldWriter(self._gold_con)
        self._ml = GoldMLDatasetBuilder(self._gold_con)

    def close(self) -> None:
        """Close both Silver and Gold DuckDB connections."""
        self._silver_con.close()
        self._gold_con.close()
        logger.progress("Closed GoldStorage connections")

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    def _build_if_present(
        self,
        required: tuple[str, ...],
        silver_tables: set[str],
        build_fn: Callable[[], pd.DataFrame],
        gold_table: str,
    ) -> None:
        """Build and write `gold_table` iff every table in `required` exists.

        Otherwise logs which of `required` is missing and skips the build —
        the shared shape behind the guard clauses in _pipeline.
        """
        if warn_if_missing(
            logger, required, silver_tables, gold_table, tier_label="Silver"
        ):
            return
        logger.progress("Building %s", gold_table)
        self._writer.full_load(build_fn(), gold_table)

    def _pipeline(self, update: bool) -> None:
        """Run the full Silver → Gold transformation pipeline.

        Checks which Silver tables are available and builds only the
        corresponding Gold tables. Missing Silver tables produce a warning
        and are silently skipped so a partial Silver tier does not abort
        the whole pipeline.

        Both populate() and update() call this with different `update` flags,
        but the flag is unused here because all Gold tables are always fully
        rebuilt — window-function features cannot be incrementally patched.

        gold_ml_dataset is only built when gold_price_features spans at least
        _MIN_ML_HORIZON_DAYS days, ensuring at least one t+7 target pair exists.
        Shorter histories produce a NULL-target frame with no usable training rows.

        Args:
            update: Unused; kept for interface compatibility with TransformStorage.
        """
        silver_tables = get_tables(self._silver_con)

        self._build_if_present(
            ("silver_cards",),
            silver_tables,
            self._features.build_card_features,
            "gold_card_features",
        )
        self._build_if_present(
            ("silver_prices_history",),
            silver_tables,
            self._features.build_price_features,
            "gold_price_features",
        )
        self._build_if_present(
            ("silver_language_prices_history", "silver_prices_history"),
            silver_tables,
            self._features.build_language_premiums,
            "gold_language_premiums",
        )

        # gold_demand_signals and gold_events share a single Silver dependency
        # (silver_meta_history) — kept as a pair with one combined warning so
        # a missing-table log line doesn't repeat itself twice for one cause.
        if not warn_if_missing(
            logger,
            ("silver_meta_history",),
            silver_tables,
            "gold_demand_signals, gold_events",
            tier_label="Silver",
        ):
            logger.progress("Building gold_demand_signals")
            self._writer.full_load(
                self._signals.build_demand_signals(), "gold_demand_signals"
            )
            logger.progress("Building gold_events")
            self._writer.full_load(self._signals.build_events(), "gold_events")

        self._build_if_present(
            ("silver_format_staples_history",),
            silver_tables,
            self._signals.build_format_staples,
            "gold_format_staples",
        )
        self._build_if_present(
            ("silver_meta_history", "silver_prices_history"),
            silver_tables,
            self._signals.build_ban_price_impact,
            "gold_ban_price_impact",
        )
        self._build_if_present(
            ("silver_tournament_results_history",),
            silver_tables,
            self._signals.build_tournament_signals,
            "gold_tournament_signals",
        )

        gold_tables = get_tables(self._gold_con)
        if "gold_price_features" in gold_tables:
            _row = self._gold_con.execute(
                "SELECT DATEDIFF('day',"
                "   MIN(CAST(snapshot_date AS DATE)),"
                "   MAX(CAST(snapshot_date AS DATE)))"
                " FROM gold_price_features"
            ).fetchone()
            horizon = (_row[0] if _row is not None else None) or 0
            if horizon < _MIN_ML_HORIZON_DAYS:
                logger.warning(
                    "gold_price_features spans %d day(s) — need ≥%d for t+7 targets"
                    " to exist; skipping gold_ml_dataset",
                    horizon,
                    _MIN_ML_HORIZON_DAYS,
                )
            else:
                logger.progress("Building gold_ml_dataset")
                self._writer.full_load(self._ml.build_ml_dataset(), "gold_ml_dataset")
        else:
            logger.warning("gold_price_features not found — skipping gold_ml_dataset")

        self._prune_orphaned_tables()

    def _prune_orphaned_tables(self) -> None:
        """Drop any gold_* table that no builder in this class can produce.

        A table only reaches here if the code that used to build it was
        removed. A table whose Silver source is merely absent on this run
        stays in _KNOWN_GOLD_TABLES and is left untouched.
        """
        gold_tables = get_tables(self._gold_con)
        orphaned = {
            t
            for t in gold_tables
            if t.startswith("gold_") and t not in self._KNOWN_GOLD_TABLES
        }
        for table_name in sorted(orphaned):
            logger.warning(
                "Dropping orphaned Gold table %r — no builder produces it anymore",
                table_name,
            )
            self._gold_con.execute(f"DROP TABLE {table_name}")
