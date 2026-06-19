"""Gold ML dataset builder.

Assembles the analysis-ready training frame by joining all Gold feature tables
into a single wide table keyed on (uuid, snapshot_date).
"""

import duckdb
import pandas as pd

from src.data.cards.storage.base import get_tables


class GoldMLDatasetBuilder:
    def __init__(self, gold_connection: duckdb.DuckDBPyConnection) -> None:
        self._gold_connection = gold_connection

    def build_ml_dataset(self) -> pd.DataFrame:
        """Build gold_ml_dataset — the analysis-ready training frame.

        Combines gold_price_features (spine) with static card attributes,
        demand signals, tournament signals, and format-staple deck-inclusion
        rates into one wide table. Each row is one (uuid, snapshot_date)
        observation with forward price targets attached.

        Returns empty DataFrame if gold_price_features is absent.

        Join strategy
        -------------
        gold_card_features       LEFT JOIN on uuid — static card attributes.
        gold_demand_signals      LEFT JOIN on (scryfall_id, snapshot_date) — time-aligned.
        gold_tournament_signals  LEFT JOIN on oracle_id — cross-sectional aggregate;
                                 no snapshot_date because tournament data is historical
                                 and gold_tournament_signals has no date dimension.
                                 All snapshot rows for a card receive the same signal.
        gold_format_staples      LEFT JOIN on (card_name, snapshot_date, format) —
                                 time-aligned. Pivoted to four format columns:
                                 commander, modern, legacy, vintage.

        Format-staple join note
        -----------------------
        gold_format_staples.card_name is the MTGGoldfish display name.
        gold_card_features.name is the Scryfall canonical name.
        Matching on name is correct at oracle level — all printings of a card
        share the same Scryfall canonical name and the same MTGGoldfish staple row.

        Missing Gold tables degrade gracefully (NULL columns rather than raising).
        The caller (GoldStorage._pipeline) gates on gold_price_features existing;
        all other joins are optional.
        """
        gold_tables = get_tables(self._gold_connection)
        if "gold_price_features" not in gold_tables:
            return pd.DataFrame()

        has_card = "gold_card_features" in gold_tables
        has_signals = "gold_demand_signals" in gold_tables
        has_tournament = "gold_tournament_signals" in gold_tables and has_card
        has_staples = "gold_format_staples" in gold_tables and has_card

        card_cols = (
            """cf.rarity, cf.mana_value, cf.is_reserved, cf.is_reprint,
               cf.color_count, cf.color_identity_count,
               cf.is_commander_legal, cf.is_modern_legal,
               cf.is_legacy_legal, cf.is_standard_legal,
               cf.format_count, cf.print_count, cf.finish_count,
               cf.has_etched_finish, cf.edhrec_saltiness, cf.set_type,"""
            if has_card
            else """NULL AS rarity, NULL AS mana_value,
               NULL AS is_reserved, NULL AS is_reprint,
               NULL AS color_count, NULL AS color_identity_count,
               NULL AS is_commander_legal, NULL AS is_modern_legal,
               NULL AS is_legacy_legal, NULL AS is_standard_legal,
               NULL AS format_count, NULL AS print_count, NULL AS finish_count,
               NULL AS has_etched_finish, NULL AS edhrec_saltiness, NULL AS set_type,"""
        )
        card_join = (
            "LEFT JOIN gold_card_features cf ON wc.uuid = cf.uuid" if has_card else ""
        )

        demand_cols = (
            """ds.commander_banned, ds.modern_banned,
               ds.legacy_banned, ds.standard_banned,
               ds.commander_unbanned, ds.modern_unbanned,
               ds.edhrec_rank_change,"""
            if has_signals
            else """NULL AS commander_banned, NULL AS modern_banned,
               NULL AS legacy_banned, NULL AS standard_banned,
               NULL AS commander_unbanned, NULL AS modern_unbanned,
               NULL AS edhrec_rank_change,"""
        )
        demand_join = (
            """LEFT JOIN gold_demand_signals ds
                   ON wc.scryfall_id = ds.id
                   AND wc.snapshot_date = ds.snapshot_date"""
            if has_signals
            else ""
        )

        tournament_cols = (
            "ts_agg.top8_30d_total, ts_agg.top8_90d_total, ts_agg.top8_copies_avg,"
            if has_tournament
            else "NULL AS top8_30d_total, NULL AS top8_90d_total, NULL AS top8_copies_avg,"
        )
        tournament_join = (
            """LEFT JOIN (
                   SELECT oracle_id,
                       SUM(top8_appearances_30d) AS top8_30d_total,
                       SUM(top8_appearances_90d) AS top8_90d_total,
                       AVG(top8_copies_avg)      AS top8_copies_avg
                   FROM gold_tournament_signals
                   GROUP BY oracle_id
               ) ts_agg ON cf.oracle_id = ts_agg.oracle_id"""
            if has_tournament
            else ""
        )

        staples_cols = (
            """fs_cmd.deck_pct         AS staple_pct_commander,
               fs_cmd.deck_pct_7d_avg  AS staple_7d_commander,
               fs_mod.deck_pct         AS staple_pct_modern,
               fs_leg.deck_pct         AS staple_pct_legacy,
               fs_vint.deck_pct        AS staple_pct_vintage,
               fs_vint.deck_pct_7d_avg AS staple_7d_vintage"""
            if has_staples
            else """NULL AS staple_pct_commander,
               NULL AS staple_7d_commander,
               NULL AS staple_pct_modern,
               NULL AS staple_pct_legacy,
               NULL AS staple_pct_vintage,
               NULL AS staple_7d_vintage"""
        )
        staples_joins = (
            """LEFT JOIN gold_format_staples fs_cmd
                   ON cf.name = fs_cmd.card_name
                   AND wc.snapshot_date = fs_cmd.snapshot_date
                   AND fs_cmd.format = 'commander'
               LEFT JOIN gold_format_staples fs_mod
                   ON cf.name = fs_mod.card_name
                   AND wc.snapshot_date = fs_mod.snapshot_date
                   AND fs_mod.format = 'modern'
               LEFT JOIN gold_format_staples fs_leg
                   ON cf.name = fs_leg.card_name
                   AND wc.snapshot_date = fs_leg.snapshot_date
                   AND fs_leg.format = 'legacy'
               LEFT JOIN gold_format_staples fs_vint
                   ON cf.name = fs_vint.card_name
                   AND wc.snapshot_date = fs_vint.snapshot_date
                   AND fs_vint.format = 'vintage'"""
            if has_staples
            else ""
        )

        return self._gold_connection.execute(f"""
            WITH spine AS (
                SELECT * FROM gold_price_features
            ),
            with_targets AS (
                SELECT
                    s.*,
                    t7.eur  AS target_price_7d,
                    t30.eur AS target_price_30d
                FROM spine s
                LEFT JOIN gold_price_features t7
                    ON s.uuid = t7.uuid
                    AND CAST(t7.snapshot_date AS DATE)
                        = CAST(s.snapshot_date AS DATE) + INTERVAL '7 days'
                LEFT JOIN gold_price_features t30
                    ON s.uuid = t30.uuid
                    AND CAST(t30.snapshot_date AS DATE)
                        = CAST(s.snapshot_date AS DATE) + INTERVAL '30 days'
            ),
            with_change_label AS (
                SELECT *,
                    CASE
                        WHEN target_price_30d IS NULL        THEN NULL
                        WHEN target_price_30d > eur * 1.2   THEN 'up'
                        WHEN target_price_30d < eur * 0.8   THEN 'down'
                        ELSE 'flat'
                    END AS target_change_30d
                FROM with_targets
                WHERE eur IS NOT NULL
            )
            SELECT
                wc.*,
                {card_cols}
                {demand_cols}
                {tournament_cols}
                {staples_cols}
            FROM with_change_label wc
            {card_join}
            {demand_join}
            {tournament_join}
            {staples_joins}
            ORDER BY wc.uuid, wc.snapshot_date
        """).df()
