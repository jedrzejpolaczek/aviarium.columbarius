"""Demand, ban-event, tournament and format-staple signal builders for the Gold tier.

GoldSignalBuilders reads clean Silver tables and produces five Gold tables:

    gold_demand_signals     — per-(scryfall_id, date) ban/unban event flags and
                              EDHREC rank deltas as a proxy for demand momentum.
    gold_events             — format-level ban/unban event calendar; one row per
                              (event_date, format, event_type) with card_count.
                              Anchor table for Chow structural break tests and
                              days_since_last_ban/days_since_last_unban features.
                              A SQL pre-check short-circuits before _load_and_parse_meta
                              when no legality transitions exist.
    gold_format_staples     — per-(card, format, date) rolling deck-inclusion
                              averages and momentum signals.
    gold_ban_price_impact   — per-(card, format, event) EUR price windows before
                              and after each ban or unban transition.
    gold_tournament_signals — per-(oracle_id, format) top-8 appearance counts
                              and copy averages over 30-day and 90-day windows.
"""

import json

import duckdb
import pandas as pd

from src.data.cards.storage.base import get_tables
from src.logger import get_logger


logger = get_logger(__name__)


class GoldSignalBuilders:
    """Builds event-driven and time-series signal tables from Silver history data."""
    _FORMATS = ["commander", "standard", "modern", "legacy", "vintage"]

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

    def __init__(self, silver_con: duckdb.DuckDBPyConnection) -> None:
        self._silver_con = silver_con

    def _load_and_parse_meta(self, extra_cols: tuple[str, ...] = ()) -> pd.DataFrame:
        """Query silver_meta_history and expand to per-format columns.

        Returns a DataFrame with columns: id, snapshot_date, legalities (dict),
        one {fmt}_legality column per format in _FORMATS, plus any extra_cols requested.
        Callers are responsible for dropping the legalities column if not needed.
        """
        # extra_cols must be hard-coded column names — not user input
        select_cols = ", ".join(("id", "snapshot_date", "legalities") + extra_cols)
        df = self._silver_con.execute(
            f"SELECT {select_cols} FROM silver_meta_history"
        ).df()
        if df.empty:
            return df
        df = df.sort_values(["id", "snapshot_date"]).reset_index(drop=True)
        df["legalities"] = df["legalities"].apply(
            lambda x: (
                json.loads(x)
                if isinstance(x, str)
                else (x if isinstance(x, dict) else {})
            )
        )
        for fmt in self._FORMATS:
            df[f"{fmt}_legality"] = df["legalities"].apply(lambda x, f=fmt: x.get(f))
        return df

    def _has_legality_transitions(self) -> bool:
        """Return True if any card has more than one distinct legality value across snapshots.

        Runs entirely in SQL (O(N) table scan with GROUP BY) so it avoids loading
        silver_meta_history into Python when no transitions exist.
        Called as a fast pre-check in build_events() and build_ban_price_impact()
        before the expensive _load_and_parse_meta() + groupby-shift pipeline.
        """
        row = self._silver_con.execute("""
            SELECT 1 FROM (
                SELECT id
                FROM silver_meta_history
                WHERE legalities IS NOT NULL
                GROUP BY id
                HAVING COUNT(DISTINCT legalities) > 1
            ) t
            LIMIT 1
        """).fetchone()
        return row is not None

    def build_demand_signals(self) -> pd.DataFrame:
        """Build gold_demand_signals from silver_meta_history.

        Detects ban/unban events by comparing consecutive daily legality
        snapshots per card, and computes EDHREC rank deltas as a proxy for
        demand momentum.

        Returns:
            One row per (id, snapshot_date) with event flags and rank deltas.
        """
        df = self._load_and_parse_meta(extra_cols=("edhrec_rank",))
        if df.empty:
            return df

        for fmt in self._FORMATS:
            prev = df.groupby("id")[f"{fmt}_legality"].shift(1)
            df[f"{fmt}_banned"] = (prev == "legal") & (
                df[f"{fmt}_legality"] == "banned"
            )
            df[f"{fmt}_unbanned"] = (prev == "banned") & (
                df[f"{fmt}_legality"] == "legal"
            )

        prev_rank = df.groupby("id")["edhrec_rank"].shift(1)
        df["edhrec_rank_change"] = df["edhrec_rank"] - prev_rank

        return df.drop(columns=["legalities"])

    def build_events(self) -> pd.DataFrame:
        """Build gold_events — format-level ban/unban event calendar.

        Aggregates per-card legality transitions from silver_meta_history into
        one row per (event_date, format, event_type), with a card_count of how
        many cards were affected. Serves as the anchor table for Chow structural
        break tests (NB06) and days_since_last_ban/days_since_last_unban features.

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

        meta_df = self._load_and_parse_meta()
        if meta_df.empty:
            return empty

        chunks: list[pd.DataFrame] = []
        for fmt in self._FORMATS:
            col = f"{fmt}_legality"
            prev = meta_df.groupby("id")[col].shift(1)
            for mask, event_type in [
                ((prev == "legal") & (meta_df[col] == "banned"), "ban"),
                ((prev == "banned") & (meta_df[col] == "legal"), "unban"),
            ]:
                agg = (
                    meta_df.loc[mask, ["snapshot_date"]]
                    .assign(format=fmt, event_type=event_type)
                    .groupby(["snapshot_date", "format", "event_type"])
                    .size()
                    .reset_index(name="card_count")
                    .rename(columns={"snapshot_date": "event_date"})
                )
                chunks.append(agg)

        non_empty = [c for c in chunks if not c.empty]
        if not non_empty:
            return empty

        return (
            pd.concat(non_empty, ignore_index=True)[self._EVENTS_COLS]
            .sort_values(["event_date", "format", "event_type"])
            .reset_index(drop=True)
        )

    def build_format_staples(self) -> pd.DataFrame:
        """Build gold_format_staples from silver_format_staples_history.

        Computes rolling deck-inclusion averages and momentum signals using
        DuckDB window functions over the full history. Each row represents one
        daily snapshot for a (card, format) pair.

        Returns:
            One row per (id, snapshot_date) with format staple feature columns.
        """
        return self._silver_con.execute("""
            SELECT
                id,
                card_name,
                format,
                snapshot_date,
                deck_pct,
                played,
                top,
                AVG(deck_pct) OVER w7  AS deck_pct_7d_avg,
                AVG(deck_pct) OVER w30 AS deck_pct_30d_avg,
                deck_pct - LAG(deck_pct, 7)  OVER (PARTITION BY id ORDER BY snapshot_date)
                    AS deck_pct_change_7d,
                deck_pct - LAG(deck_pct, 30) OVER (PARTITION BY id ORDER BY snapshot_date)
                    AS deck_pct_change_30d
            FROM silver_format_staples_history
            WINDOW
                w7  AS (PARTITION BY id ORDER BY snapshot_date
                         ROWS BETWEEN 6 PRECEDING AND CURRENT ROW),
                w30 AS (PARTITION BY id ORDER BY snapshot_date
                         ROWS BETWEEN 29 PRECEDING AND CURRENT ROW)
            ORDER BY id, snapshot_date
        """).df()

    def build_ban_price_impact(self) -> pd.DataFrame:
        """Build gold_ban_price_impact from silver_meta_history and silver_prices_history.

        For each ban/unban event detected in daily legality snapshots, computes EUR
        price windows (7d and 30d) before and after the event date. Useful for
        correlating legality changes with price movements.

        Event detection: a "banned" event fires when a format legality transitions
        from "legal" → "banned" between two consecutive daily snapshots; "unbanned"
        fires on the reverse transition.

        Price columns are NULL when fewer than 7/30 days of price history exist
        around the event — they fill in over time as the pipeline accumulates data.

        Returns:
            One row per (scryfall_id, format, event_type, event_date).
        """
        empty = pd.DataFrame(columns=self._BAN_IMPACT_COLS)

        if not self._has_legality_transitions():
            logger.info(
                "No legality transitions in silver_meta_history — skipping build_ban_price_impact"
            )
            return empty

        meta_df = self._load_and_parse_meta()
        if meta_df.empty:
            return empty

        event_chunks: list[pd.DataFrame] = []
        for fmt in self._FORMATS:
            col = f"{fmt}_legality"
            prev = meta_df.groupby("id")[col].shift(1)
            for mask, event_type in [
                ((prev == "legal") & (meta_df[col] == "banned"), "ban"),
                ((prev == "banned") & (meta_df[col] == "legal"), "unban"),
            ]:
                rows = meta_df.loc[mask, ["id", "snapshot_date"]].copy()
                rows["format"] = fmt
                rows["event_type"] = event_type
                rows = rows.rename(
                    columns={"id": "scryfall_id", "snapshot_date": "event_date"}
                )
                event_chunks.append(rows)

        events_df = pd.concat(event_chunks, ignore_index=True)
        if events_df.empty:
            snapshot_dates = meta_df["snapshot_date"].nunique()
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
        return self._silver_con.execute("""
            WITH base AS (
                SELECT *,
                    CAST(tournament_date AS DATE) AS tournament_dt,
                    (CURRENT_DATE - CAST(tournament_date AS DATE)) AS days_ago
                FROM silver_tournament_results_history
                WHERE oracle_id IS NOT NULL
            )
            SELECT
                oracle_id,
                MIN(scryfall_id) AS scryfall_id,
                format,
                COUNT(DISTINCT CASE
                    WHEN days_ago <= 30 AND NOT is_sideboard THEN tournament_id
                END) AS top8_appearances_30d,
                COUNT(DISTINCT CASE
                    WHEN days_ago <= 90 AND NOT is_sideboard THEN tournament_id
                END) AS top8_appearances_90d,
                AVG(CASE
                    WHEN NOT is_sideboard THEN CAST(copies AS FLOAT)
                END) AS top8_copies_avg,
                COUNT(DISTINCT CASE
                    WHEN days_ago <= 30 AND is_sideboard THEN tournament_id
                END) AS sideboard_appearances_30d,
                COUNT(DISTINCT CASE
                    WHEN days_ago <= 90 AND NOT is_sideboard THEN tournament_id
                END) * 100.0
                    / NULLIF(COUNT(DISTINCT CASE
                        WHEN days_ago <= 90 THEN tournament_id
                    END), 0) AS main_deck_pct,
                MAX(CASE
                    WHEN NOT is_sideboard THEN tournament_date
                END) AS last_top8_date
            FROM base
            GROUP BY oracle_id, format
            ORDER BY oracle_id, format
        """).df()
