"""Gold ML dataset builder.

Assembles the analysis-ready training frame by joining all Gold feature tables
into a single wide table keyed on (uuid, snapshot_date).
"""

from pathlib import Path

import duckdb
import pandas as pd

from src.data.cards.storage.base.storage import get_tables

_SQL_DIR = Path(__file__).parent / "sql"


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

        # Each optional Gold table contributes a (flag, real_cols, null_cols, join)
        # tuple. When the flag is False the real columns are replaced by NULL
        # placeholders of the same names and the join is omitted entirely, so
        # missing Gold tables degrade gracefully rather than raising.
        optional_joins: list[tuple[bool, str, str, str]] = [
            (
                has_card,
                """cf.rarity, cf.mana_value, cf.is_reserved, cf.is_reprint,
                   cf.color_count, cf.color_identity_count,
                   cf.is_commander_legal, cf.is_modern_legal,
                   cf.is_legacy_legal, cf.is_standard_legal,
                   cf.format_count, cf.print_count, cf.finish_count,
                   cf.has_etched_finish, cf.edhrec_saltiness, cf.set_type,""",
                """NULL AS rarity, NULL AS mana_value,
                   NULL AS is_reserved, NULL AS is_reprint,
                   NULL AS color_count, NULL AS color_identity_count,
                   NULL AS is_commander_legal, NULL AS is_modern_legal,
                   NULL AS is_legacy_legal, NULL AS is_standard_legal,
                   NULL AS format_count, NULL AS print_count, NULL AS finish_count,
                   NULL AS has_etched_finish, NULL AS edhrec_saltiness, NULL AS set_type,""",
                "LEFT JOIN gold_card_features cf ON wc.uuid = cf.uuid",
            ),
            (
                has_signals,
                """ds.commander_banned, ds.modern_banned,
                   ds.legacy_banned, ds.standard_banned,
                   ds.commander_unbanned, ds.modern_unbanned,
                   ds.edhrec_rank_change,""",
                """NULL AS commander_banned, NULL AS modern_banned,
                   NULL AS legacy_banned, NULL AS standard_banned,
                   NULL AS commander_unbanned, NULL AS modern_unbanned,
                   NULL AS edhrec_rank_change,""",
                """LEFT JOIN gold_demand_signals ds
                       ON wc.scryfall_id = ds.id
                       AND wc.snapshot_date = ds.snapshot_date""",
            ),
            (
                has_tournament,
                "ts_agg.top8_30d_total, ts_agg.top8_90d_total, ts_agg.top8_copies_avg,",
                "NULL AS top8_30d_total, NULL AS top8_90d_total, NULL AS top8_copies_avg,",
                """LEFT JOIN (
                       SELECT oracle_id,
                           SUM(top8_appearances_30d) AS top8_30d_total,
                           SUM(top8_appearances_90d) AS top8_90d_total,
                           AVG(top8_copies_avg)      AS top8_copies_avg
                       FROM gold_tournament_signals
                       GROUP BY oracle_id
                   ) ts_agg ON cf.oracle_id = ts_agg.oracle_id""",
            ),
            (
                has_staples,
                """fs_cmd.deck_pct         AS staple_pct_commander,
                   fs_cmd.deck_pct_7d_avg  AS staple_7d_commander,
                   fs_mod.deck_pct         AS staple_pct_modern,
                   fs_leg.deck_pct         AS staple_pct_legacy,
                   fs_vint.deck_pct        AS staple_pct_vintage,
                   fs_vint.deck_pct_7d_avg AS staple_7d_vintage""",
                """NULL AS staple_pct_commander,
                   NULL AS staple_7d_commander,
                   NULL AS staple_pct_modern,
                   NULL AS staple_pct_legacy,
                   NULL AS staple_pct_vintage,
                   NULL AS staple_7d_vintage""",
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
                       AND fs_vint.format = 'vintage'""",
            ),
        ]

        cols_sql = "\n    ".join(
            real_cols if flag else null_cols
            for flag, real_cols, null_cols, _ in optional_joins
        )
        joins_sql = "\n".join(join for flag, _, _, join in optional_joins if flag)

        sql = (
            (_SQL_DIR / "ml_dataset.sql")
            .read_text()
            .format(cols=cols_sql, joins=joins_sql)
        )
        return self._gold_connection.execute(sql).df()
