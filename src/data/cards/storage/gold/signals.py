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

    def _has_legality_transitions(self) -> bool:
        """Return True if any card has more than one distinct legality value across snapshots.

        Fast SQL pre-check (O(N) GROUP BY) used by build_events() and
        build_ban_price_impact() to skip the full window-function query when
        the silver_meta_history table has no transitions at all.
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

        Detects ban/unban events and EDHREC rank deltas entirely in DuckDB using
        LAG() window functions and json_extract_string(). No data leaves DuckDB.
        """
        silver_tables = get_tables(self._silver_con)
        if "silver_meta_history" not in silver_tables:
            return pd.DataFrame()
        return self._silver_con.execute("""
            WITH lagged AS (
                SELECT
                    id,
                    snapshot_date,
                    edhrec_rank,
                    LAG(edhrec_rank) OVER w                                          AS prev_rank,
                    json_extract_string(legalities, '$.commander')                   AS commander_legality,
                    LAG(json_extract_string(legalities, '$.commander')) OVER w       AS prev_commander,
                    json_extract_string(legalities, '$.standard')                    AS standard_legality,
                    LAG(json_extract_string(legalities, '$.standard'))  OVER w       AS prev_standard,
                    json_extract_string(legalities, '$.modern')                      AS modern_legality,
                    LAG(json_extract_string(legalities, '$.modern'))    OVER w       AS prev_modern,
                    json_extract_string(legalities, '$.legacy')                      AS legacy_legality,
                    LAG(json_extract_string(legalities, '$.legacy'))    OVER w       AS prev_legacy,
                    json_extract_string(legalities, '$.vintage')                     AS vintage_legality,
                    LAG(json_extract_string(legalities, '$.vintage'))   OVER w       AS prev_vintage
                FROM silver_meta_history
                WINDOW w AS (PARTITION BY id ORDER BY snapshot_date)
            )
            SELECT
                id,
                snapshot_date,
                edhrec_rank,
                edhrec_rank - prev_rank                                              AS edhrec_rank_change,
                commander_legality,
                standard_legality,
                modern_legality,
                legacy_legality,
                vintage_legality,
                COALESCE(prev_commander = 'legal'  AND commander_legality = 'banned', FALSE) AS commander_banned,
                COALESCE(prev_commander = 'banned' AND commander_legality = 'legal',  FALSE) AS commander_unbanned,
                COALESCE(prev_standard  = 'legal'  AND standard_legality  = 'banned', FALSE) AS standard_banned,
                COALESCE(prev_standard  = 'banned' AND standard_legality  = 'legal',  FALSE) AS standard_unbanned,
                COALESCE(prev_modern    = 'legal'  AND modern_legality    = 'banned', FALSE) AS modern_banned,
                COALESCE(prev_modern    = 'banned' AND modern_legality    = 'legal',  FALSE) AS modern_unbanned,
                COALESCE(prev_legacy    = 'legal'  AND legacy_legality    = 'banned', FALSE) AS legacy_banned,
                COALESCE(prev_legacy    = 'banned' AND legacy_legality    = 'legal',  FALSE) AS legacy_unbanned,
                COALESCE(prev_vintage   = 'legal'  AND vintage_legality   = 'banned', FALSE) AS vintage_banned,
                COALESCE(prev_vintage   = 'banned' AND vintage_legality   = 'legal',  FALSE) AS vintage_unbanned
            FROM lagged
            ORDER BY id, snapshot_date
        """).df()

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

        result = self._silver_con.execute("""
            WITH lagged AS (
                SELECT
                    id,
                    snapshot_date,
                    json_extract_string(legalities, '$.commander') AS curr_commander,
                    LAG(json_extract_string(legalities, '$.commander')) OVER w AS prev_commander,
                    json_extract_string(legalities, '$.standard')  AS curr_standard,
                    LAG(json_extract_string(legalities, '$.standard'))  OVER w AS prev_standard,
                    json_extract_string(legalities, '$.modern')    AS curr_modern,
                    LAG(json_extract_string(legalities, '$.modern'))    OVER w AS prev_modern,
                    json_extract_string(legalities, '$.legacy')    AS curr_legacy,
                    LAG(json_extract_string(legalities, '$.legacy'))    OVER w AS prev_legacy,
                    json_extract_string(legalities, '$.vintage')   AS curr_vintage,
                    LAG(json_extract_string(legalities, '$.vintage'))   OVER w AS prev_vintage
                FROM silver_meta_history
                WINDOW w AS (PARTITION BY id ORDER BY snapshot_date)
            ),
            transitions AS (
                SELECT snapshot_date AS event_date, 'commander' AS format,
                       CASE WHEN prev_commander = 'legal'  AND curr_commander = 'banned' THEN 'ban'
                            WHEN prev_commander = 'banned' AND curr_commander = 'legal'  THEN 'unban'
                       END AS event_type
                FROM lagged
                WHERE prev_commander IS NOT NULL
                  AND (   (prev_commander = 'legal'  AND curr_commander = 'banned')
                       OR (prev_commander = 'banned' AND curr_commander = 'legal'))
                UNION ALL
                SELECT snapshot_date, 'standard',
                       CASE WHEN prev_standard = 'legal'  AND curr_standard = 'banned' THEN 'ban'
                            WHEN prev_standard = 'banned' AND curr_standard = 'legal'  THEN 'unban'
                       END
                FROM lagged
                WHERE prev_standard IS NOT NULL
                  AND (   (prev_standard = 'legal'  AND curr_standard = 'banned')
                       OR (prev_standard = 'banned' AND curr_standard = 'legal'))
                UNION ALL
                SELECT snapshot_date, 'modern',
                       CASE WHEN prev_modern = 'legal'  AND curr_modern = 'banned' THEN 'ban'
                            WHEN prev_modern = 'banned' AND curr_modern = 'legal'  THEN 'unban'
                       END
                FROM lagged
                WHERE prev_modern IS NOT NULL
                  AND (   (prev_modern = 'legal'  AND curr_modern = 'banned')
                       OR (prev_modern = 'banned' AND curr_modern = 'legal'))
                UNION ALL
                SELECT snapshot_date, 'legacy',
                       CASE WHEN prev_legacy = 'legal'  AND curr_legacy = 'banned' THEN 'ban'
                            WHEN prev_legacy = 'banned' AND curr_legacy = 'legal'  THEN 'unban'
                       END
                FROM lagged
                WHERE prev_legacy IS NOT NULL
                  AND (   (prev_legacy = 'legal'  AND curr_legacy = 'banned')
                       OR (prev_legacy = 'banned' AND curr_legacy = 'legal'))
                UNION ALL
                SELECT snapshot_date, 'vintage',
                       CASE WHEN prev_vintage = 'legal'  AND curr_vintage = 'banned' THEN 'ban'
                            WHEN prev_vintage = 'banned' AND curr_vintage = 'legal'  THEN 'unban'
                       END
                FROM lagged
                WHERE prev_vintage IS NOT NULL
                  AND (   (prev_vintage = 'legal'  AND curr_vintage = 'banned')
                       OR (prev_vintage = 'banned' AND curr_vintage = 'legal'))
            )
            SELECT event_date, format, event_type, COUNT(*) AS card_count
            FROM transitions
            GROUP BY event_date, format, event_type
            ORDER BY event_date, format, event_type
        """).df()

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

        Event detection (legality transitions) runs in DuckDB SQL. Price window
        averaging (merge + groupby) runs in pandas on the small events DataFrame.
        """
        empty = pd.DataFrame(columns=self._BAN_IMPACT_COLS)

        if not self._has_legality_transitions():
            logger.info(
                "No legality transitions in silver_meta_history — skipping build_ban_price_impact"
            )
            return empty

        events_df = self._silver_con.execute("""
            WITH lagged AS (
                SELECT
                    id,
                    snapshot_date,
                    json_extract_string(legalities, '$.commander') AS curr_commander,
                    LAG(json_extract_string(legalities, '$.commander')) OVER w AS prev_commander,
                    json_extract_string(legalities, '$.standard')  AS curr_standard,
                    LAG(json_extract_string(legalities, '$.standard'))  OVER w AS prev_standard,
                    json_extract_string(legalities, '$.modern')    AS curr_modern,
                    LAG(json_extract_string(legalities, '$.modern'))    OVER w AS prev_modern,
                    json_extract_string(legalities, '$.legacy')    AS curr_legacy,
                    LAG(json_extract_string(legalities, '$.legacy'))    OVER w AS prev_legacy,
                    json_extract_string(legalities, '$.vintage')   AS curr_vintage,
                    LAG(json_extract_string(legalities, '$.vintage'))   OVER w AS prev_vintage
                FROM silver_meta_history
                WINDOW w AS (PARTITION BY id ORDER BY snapshot_date)
            )
            SELECT id AS scryfall_id, snapshot_date AS event_date, format, event_type
            FROM (
                SELECT id, snapshot_date, 'commander' AS format,
                       CASE WHEN prev_commander = 'legal'  AND curr_commander = 'banned' THEN 'ban'
                            WHEN prev_commander = 'banned' AND curr_commander = 'legal'  THEN 'unban'
                       END AS event_type
                FROM lagged
                WHERE prev_commander IS NOT NULL
                  AND (   (prev_commander = 'legal'  AND curr_commander = 'banned')
                       OR (prev_commander = 'banned' AND curr_commander = 'legal'))
                UNION ALL
                SELECT id, snapshot_date, 'standard',
                       CASE WHEN prev_standard = 'legal'  AND curr_standard = 'banned' THEN 'ban'
                            WHEN prev_standard = 'banned' AND curr_standard = 'legal'  THEN 'unban'
                       END
                FROM lagged
                WHERE prev_standard IS NOT NULL
                  AND (   (prev_standard = 'legal'  AND curr_standard = 'banned')
                       OR (prev_standard = 'banned' AND curr_standard = 'legal'))
                UNION ALL
                SELECT id, snapshot_date, 'modern',
                       CASE WHEN prev_modern = 'legal'  AND curr_modern = 'banned' THEN 'ban'
                            WHEN prev_modern = 'banned' AND curr_modern = 'legal'  THEN 'unban'
                       END
                FROM lagged
                WHERE prev_modern IS NOT NULL
                  AND (   (prev_modern = 'legal'  AND curr_modern = 'banned')
                       OR (prev_modern = 'banned' AND curr_modern = 'legal'))
                UNION ALL
                SELECT id, snapshot_date, 'legacy',
                       CASE WHEN prev_legacy = 'legal'  AND curr_legacy = 'banned' THEN 'ban'
                            WHEN prev_legacy = 'banned' AND curr_legacy = 'legal'  THEN 'unban'
                       END
                FROM lagged
                WHERE prev_legacy IS NOT NULL
                  AND (   (prev_legacy = 'legal'  AND curr_legacy = 'banned')
                       OR (prev_legacy = 'banned' AND curr_legacy = 'legal'))
                UNION ALL
                SELECT id, snapshot_date, 'vintage',
                       CASE WHEN prev_vintage = 'legal'  AND curr_vintage = 'banned' THEN 'ban'
                            WHEN prev_vintage = 'banned' AND curr_vintage = 'legal'  THEN 'unban'
                       END
                FROM lagged
                WHERE prev_vintage IS NOT NULL
                  AND (   (prev_vintage = 'legal'  AND curr_vintage = 'banned')
                       OR (prev_vintage = 'banned' AND curr_vintage = 'legal'))
            ) events
        """).df()

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
