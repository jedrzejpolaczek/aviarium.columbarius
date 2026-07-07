"""Demand, ban-event, tournament and format-staple signal builders for the Gold tier.

GoldSignalBuilders reads clean Silver tables and produces five Gold tables:

    gold_demand_signals     — per-(scryfall_id, date) ban/unban event flags and
                              EDHREC rank deltas as a proxy for demand momentum.
    gold_events             — format-level ban/unban event calendar; one row per
                              (event_date, format, event_type) with card_count.
                              Anchor table for Chow structural break tests and
                              days_since_last_ban/days_since_last_unban features.
    gold_format_staples     — per-(card, format, date) rolling deck-inclusion
                              averages and momentum signals.
    gold_ban_price_impact   — per-(card, format, event) EUR price windows before
                              and after each ban or unban transition.
    gold_tournament_signals — per-(oracle_id, format) top-8 appearance counts
                              and copy averages over 30-day and 90-day windows.
"""

from pathlib import Path

import duckdb
import pandas as pd

from src.data.cards.storage.base.storage import get_tables
from src.logger import get_logger


logger = get_logger(__name__)

_SQL_DIR = Path(__file__).parent / "sql"


class GoldSignalBuilders:
    """Builds event-driven and time-series signal tables from Silver history data."""

    _EVENTS_COLS = ["event_date", "format", "event_type", "card_count"]

    _BAN_IMPACT_COLS = [
        "scryfall_id",
        "format",
        "event_type",
        "event_date",
        "price_30d_before",
        "price_7d_before",
        "price_at_event",
        "price_7d_after",
        "price_30d_after",
        "price_change_7d_pct",
        "price_change_30d_pct",
    ]

    # Shared by build_demand_signals(), build_events() and
    # build_ban_price_impact(): computes each format's current legality plus
    # its previous-snapshot value via LAG() over (id, snapshot_date), along
    # with edhrec_rank/prev_rank. Canonical columns: id, snapshot_date,
    # edhrec_rank, prev_rank, curr_<format>, prev_<format> for each of
    # commander/standard/modern/legacy/vintage.
    _LEGALITY_LAG_CTE = (_SQL_DIR / "_legality_lag_cte.sql").read_text()

    # Shared by build_events() and build_ban_price_impact(): detects every
    # ban/unban legality transition per (card id, snapshot_date, format) by
    # comparing each snapshot's legality to the previous one via LAG().
    # Canonical columns: id, snapshot_date, format, event_type.
    _TRANSITIONS_CTE = (
        (_SQL_DIR / "transitions_cte.sql")
        .read_text()
        .format(legality_lag=_LEGALITY_LAG_CTE)
    )

    def __init__(self, silver_con: duckdb.DuckDBPyConnection) -> None:
        self._silver_con = silver_con

    def _has_legality_transitions(self) -> bool:
        """Return True if any card has more than one distinct legality value across snapshots.

        Fast SQL pre-check (O(N) GROUP BY) used by build_events() and
        build_ban_price_impact() to skip the full window-function query when
        the silver_meta_history table has no transitions at all.
        """
        sql = (_SQL_DIR / "legality_transitions_check.sql").read_text()
        row = self._silver_con.execute(sql).fetchone()
        return row is not None

    def build_demand_signals(self) -> pd.DataFrame:
        """Build gold_demand_signals from silver_meta_history.

        Detects ban/unban events and EDHREC rank deltas entirely in DuckDB using
        LAG() window functions and json_extract_string(). No data leaves DuckDB.
        """
        silver_tables = get_tables(self._silver_con)
        if "silver_meta_history" not in silver_tables:
            return pd.DataFrame()
        sql = (
            (_SQL_DIR / "demand_signals.sql")
            .read_text()
            .format(legality_lag=self._LEGALITY_LAG_CTE)
        )
        return self._silver_con.execute(sql).df()

    def build_events(self) -> pd.DataFrame:
        """Build gold_events — format-level ban/unban event calendar.

        Detects legality transitions entirely in DuckDB using LAG() window
        functions and json_extract_string(), then aggregates to one row per
        (event_date, format, event_type) with card_count. No data leaves DuckDB.

        Serves as the anchor table for Chow structural break tests (NB06) and
        days_since_last_ban/days_since_last_unban features.

        event_type values: 'ban' (legal → banned), 'unban' (banned → legal).

        Returns:
            One row per (event_date, format, event_type) with card_count.
            Empty DataFrame with correct schema if no transitions detected.
        """
        empty = pd.DataFrame(columns=self._EVENTS_COLS)

        if not self._has_legality_transitions():
            logger.info(
                "No legality transitions in silver_meta_history — skipping build_events"
            )
            return empty

        sql = (
            (_SQL_DIR / "events.sql")
            .read_text()
            .format(transitions_cte=self._TRANSITIONS_CTE)
        )
        result = self._silver_con.execute(sql).df()

        if result.empty:
            return empty
        return result[self._EVENTS_COLS]

    def build_format_staples(self) -> pd.DataFrame:
        """Build gold_format_staples from silver_format_staples_history.

        Computes rolling deck-inclusion averages and momentum signals using
        DuckDB window functions over the full history. Each row represents one
        daily snapshot for a (card, format) pair.

        Returns:
            One row per (id, snapshot_date) with format staple feature columns.
        """
        sql = (_SQL_DIR / "format_staples.sql").read_text()
        return self._silver_con.execute(sql).df()

    def build_ban_price_impact(self) -> pd.DataFrame:
        """Build gold_ban_price_impact from silver_meta_history and silver_prices_history.

        Event detection (legality transitions) runs in DuckDB SQL. Price window
        averaging (merge + groupby) runs in pandas on the small events DataFrame.
        """
        empty = pd.DataFrame(columns=self._BAN_IMPACT_COLS)

        if not self._has_legality_transitions():
            logger.info(
                "No legality transitions in silver_meta_history — skipping build_ban_price_impact"
            )
            return empty

        sql = (
            (_SQL_DIR / "ban_price_impact_events.sql")
            .read_text()
            .format(transitions_cte=self._TRANSITIONS_CTE)
        )
        events_df = self._silver_con.execute(sql).df()

        if events_df.empty:
            _row = self._silver_con.execute(
                "SELECT COUNT(DISTINCT snapshot_date) FROM silver_meta_history"
            ).fetchone()
            snapshot_dates = _row[0] if _row is not None else 0
            logger.info(
                "No ban/unban transitions detected across %d snapshot date(s) "
                "in silver_meta_history — gold_ban_price_impact will be empty",
                snapshot_dates,
            )
            return empty

        silver_tables = get_tables(self._silver_con)
        if "silver_prices_history" not in silver_tables:
            for c in self._BAN_IMPACT_COLS[4:]:
                events_df[c] = None
            return events_df[self._BAN_IMPACT_COLS]

        prices_df = self._silver_con.execute(
            "SELECT scryfall_id, snapshot_date, eur FROM silver_prices_history"
        ).df()

        if prices_df.empty:
            for c in self._BAN_IMPACT_COLS[4:]:
                events_df[c] = None
            return events_df[self._BAN_IMPACT_COLS]

        merged = events_df.merge(prices_df, on="scryfall_id", how="left")
        merged["days_diff"] = (
            pd.to_datetime(merged["snapshot_date"])
            - pd.to_datetime(merged["event_date"])
        ).dt.days

        key_cols = ["scryfall_id", "format", "event_type", "event_date"]

        def _window_avg(lo: int, hi: int, col_name: str) -> pd.DataFrame:
            mask = (merged["days_diff"] >= lo) & (merged["days_diff"] <= hi)
            return (
                merged[mask]
                .groupby(key_cols)["eur"]
                .mean()
                .rename(col_name)
                .reset_index()
            )

        result = events_df.copy()
        for agg in [
            _window_avg(-30, -1, "price_30d_before"),
            _window_avg(-7, -1, "price_7d_before"),
            _window_avg(0, 0, "price_at_event"),
            _window_avg(1, 7, "price_7d_after"),
            _window_avg(1, 30, "price_30d_after"),
        ]:
            result = result.merge(agg, on=key_cols, how="left")

        result["price_change_7d_pct"] = (
            result["price_7d_after"] - result["price_7d_before"]
        ) / result["price_7d_before"].replace(0, float("nan"))
        result["price_change_30d_pct"] = (
            result["price_30d_after"] - result["price_30d_before"]
        ) / result["price_30d_before"].replace(0, float("nan"))

        return result[self._BAN_IMPACT_COLS]

    def build_tournament_signals(self) -> pd.DataFrame:
        """Build gold_tournament_signals from silver_tournament_results_history.

        Groups by (oracle_id, format) and counts distinct top-8 tournament
        appearances over the last 30 and 90 days. Uses date arithmetic rather
        than row-based windowing because tournament dates are sparse and
        irregular — a card may appear in one tournament per month.

        Returns:
            One row per (oracle_id, format) with appearance counts, copy
            averages, sideboard ratio, and last appearance date.
        """
        sql = (_SQL_DIR / "tournament_signals.sql").read_text()
        return self._silver_con.execute(sql).df()
