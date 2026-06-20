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
        "configs/gold_config.json",
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

from src.data.cards.storage.base import TransformStorage, get_tables
from src.data.cards.storage.gold.features import GoldFeatureBuilders
from src.data.cards.storage.gold.signals import GoldSignalBuilders
from src.data.cards.storage.base import DuckDBWriter as GoldWriter
from src.data.cards.storage.gold.ml_dataset import GoldMLDatasetBuilder
from src.logger import get_logger

logger = get_logger(__name__)

# Minimum date span (inclusive) between the earliest and latest snapshot in
# gold_price_features required before building gold_ml_dataset. With fewer
# than 7 days all target_price_7d values are NULL (no t+7 pair exists) and
# the training frame has zero usable rows. Matches validation_config.json
# min_train_days intent; the hard floor is the prediction horizon itself.
_MIN_ML_HORIZON_DAYS = 7


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
            "configs/gold_config.json",
        ) as storage:
            storage.populate()

    Raises:
        StorageConnectionError: If either DuckDB connection cannot be opened.
    """

    def __init__(
        self, silver_db_path: str, gold_db_path: str, config_path: str
    ) -> None:
        """Open Silver (read-only) and Gold (read-write) DuckDB connections.

        Args:
            silver_db_path: Path to the Silver DuckDB file.
            gold_db_path: Path to the Gold DuckDB file (created if it does not exist).
            config_path: Path to the gold_config.json file (reserved for future use).

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

        if "silver_cards" in silver_tables:
            logger.progress("Building gold_card_features")
            self._writer.full_load(
                self._features.build_card_features(), "gold_card_features"
            )
        else:
            logger.warning("silver_cards not found — skipping gold_card_features")

        if "silver_prices_history" in silver_tables:
            logger.progress("Building gold_price_features")
            self._writer.full_load(
                self._features.build_price_features(), "gold_price_features"
            )
        else:
            logger.warning(
                "silver_prices_history not found — skipping gold_price_features"
            )

        if (
            "silver_language_prices_history" in silver_tables
            and "silver_prices_history" in silver_tables
        ):
            logger.progress("Building gold_language_premiums")
            self._writer.full_load(
                self._features.build_language_premiums(), "gold_language_premiums"
            )
        else:
            missing = [
                t
                for t in ("silver_language_prices_history", "silver_prices_history")
                if t not in silver_tables
            ]
            logger.warning(
                "Missing Silver tables %s — skipping gold_language_premiums", missing
            )

        if "silver_meta_history" in silver_tables:
            logger.progress("Building gold_demand_signals")
            self._writer.full_load(
                self._signals.build_demand_signals(), "gold_demand_signals"
            )
            logger.progress("Building gold_events")
            self._writer.full_load(self._signals.build_events(), "gold_events")
        else:
            logger.warning(
                "silver_meta_history not found — skipping gold_demand_signals, gold_events"
            )

        if "silver_format_staples_history" in silver_tables:
            logger.progress("Building gold_format_staples")
            self._writer.full_load(
                self._signals.build_format_staples(), "gold_format_staples"
            )
        else:
            logger.warning(
                "silver_format_staples_history not found — skipping gold_format_staples"
            )

        if (
            "silver_meta_history" in silver_tables
            and "silver_prices_history" in silver_tables
        ):
            logger.progress("Building gold_ban_price_impact")
            self._writer.full_load(
                self._signals.build_ban_price_impact(), "gold_ban_price_impact"
            )
        else:
            missing = [
                t
                for t in ("silver_meta_history", "silver_prices_history")
                if t not in silver_tables
            ]
            logger.warning(
                "Missing Silver tables %s — skipping gold_ban_price_impact", missing
            )

        if "silver_tournament_results_history" in silver_tables:
            logger.progress("Building gold_tournament_signals")
            self._writer.full_load(
                self._signals.build_tournament_signals(), "gold_tournament_signals"
            )
        else:
            logger.warning(
                "silver_tournament_results_history not found — skipping gold_tournament_signals"
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
