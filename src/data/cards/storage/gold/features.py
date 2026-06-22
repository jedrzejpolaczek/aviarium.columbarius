"""Card and price feature builders for the Gold tier.

GoldFeatureBuilders reads clean Silver tables and produces two Gold tables:

    gold_card_features  — static per-printing features (rarity, legality,
                          finish, colour, print-run scarcity).
    gold_price_features — per-(uuid, date) price trend features computed with
                          DuckDB window functions over the full price history.
"""

import duckdb
import pandas as pd

from src.data.cards.storage.base.storage import get_tables
from src.logger import get_logger


logger = get_logger(__name__)

# MTG's legitimate mana-value ceiling; values above this indicate corrupted
# Bronze entries (e.g. mana_value = 1_000_000 observed in raw data).
_MANA_VALUE_MAX = 20


class GoldFeatureBuilders:
    """Builds static and time-series feature tables from Silver card data."""

    def __init__(self, silver_con: duckdb.DuckDBPyConnection) -> None:
        self._silver_con = silver_con

    def build_card_features(self) -> pd.DataFrame:
        """Build gold_card_features from silver_cards.

        Silver provides list columns as native VARCHAR[] and legality as typed scalar
        columns — no JSON parsing is needed here. Gold reads values directly.

        Scryfall-only rows (digital exclusives, oversized cards, promos with no
        MTGJson counterpart) have uuid = NULL and no paper price context.
        Excluding them here prevents NULL-key fan-outs in downstream joins.

        is_legendary is derived from original_supertypes: True when the list
        produced by SilverTransforms._parse_type_line() contains "Legendary".
        When original_supertypes is absent from the Silver table (legacy data),
        the column is emitted as False.

        Returns:
            One row per card printing (uuid) with derived feature columns.
        """
        df = self._silver_con.execute(
            "SELECT * FROM silver_cards WHERE uuid IS NOT NULL"
        ).df()

        if "is_commander_legal" not in df.columns:
            logger.warning(
                "silver_cards missing scalar legality columns — "
                "is_commander_legal/format_count will be NULL. Re-run Silver to fix."
            )

        # Silver guarantees VARCHAR[] for list columns — DuckDB returns Python lists.
        finishes = df["finishes"]
        colors = df["colors"]
        color_identity = df["color_identity"]
        variations = df["variations"]
        supertypes = (
            df["original_supertypes"]
            if "original_supertypes" in df.columns
            else pd.Series([[]] * len(df))
        )

        def col(name: str) -> pd.Series:
            return df[name] if name in df.columns else pd.Series([None] * len(df))

        result = pd.DataFrame(
            {
                "uuid": df["uuid"],
                "scryfall_id": df["scryfall_id"],
                "oracle_id": col("oracle_id"),
                "name": df["name"],
                "set_code": df["set_code"],
                "rarity": df["rarity"],
                "mana_value": df["mana_value"],
                "is_reserved": df["is_reserved"],
                "is_reprint": df["is_reprint"],
                "is_promo": df["is_promo"],
                "is_full_art": df["is_full_art"],
                "is_textless": df["is_textless"],
                # edhrec_rank is intentionally absent here: it is snapshotted
                # daily in silver_meta_history and is added to gold_price_features
                # with proper time-alignment (see build_price_features). Storing
                # it here would freeze the rank at bronze-ingest time, which is
                # stale for ~82 % of cards within a week of pipeline start.
                "edhrec_saltiness": col("edhrec_saltiness"),
                "set_type": col("set_type"),
                # scarcity
                # in_collector_booster / in_play_booster omitted: both were 100 % False
                # (constant) across all current cards — zero information for a model.
                "finish_count": finishes.apply(len),
                "has_etched_finish": finishes.apply(lambda x: "etched" in x),
                # card characteristics
                "color_count": colors.apply(len),
                "color_identity_count": color_identity.apply(len),
                "variation_count": variations.apply(len),
                "is_legendary": supertypes.apply(lambda x: "Legendary" in x),
                # format demand — read directly from Silver scalar columns
                "is_commander_legal": col("is_commander_legal"),
                "is_standard_legal": col("is_standard_legal"),
                "is_modern_legal": col("is_modern_legal"),
                "is_legacy_legal": col("is_legacy_legal"),
                # is_vintage_legal omitted: r = 0.945 with is_legacy_legal (near-redundant).
                # format_count already captures the legacy/vintage overlap numerically.
                "format_count": col("format_count"),
            }
        )

        # Cap mana_value at the legitimate MTG maximum (corrupted entries capped here).
        result["mana_value"] = result["mana_value"].where(
            result["mana_value"] <= _MANA_VALUE_MAX
        )

        # How many unique printings share the same oracle card
        if "oracle_id" in df.columns:
            print_counts = (
                df[df["oracle_id"].notna()]
                .groupby("oracle_id", sort=False)
                .size()
                .rename("print_count")
                .reset_index()
            )
            result = result.merge(print_counts, on="oracle_id", how="left")
            result["print_count"] = result["print_count"].fillna(1).astype(int)
        else:
            result["print_count"] = 1

        return result

    def build_language_premiums(self) -> pd.DataFrame:
        """Build gold_language_premiums from silver_language_prices_history.

        Joins non-English language variant prices (from silver_language_prices_history)
        to the corresponding English canonical prices (from silver_prices_history) on
        (canonical_uuid, snapshot_date), then computes the price premium ratio for
        both standard and foil EUR prices.

        eur_lang_premium > 1.0 means the language variant trades at a premium over
        the English version on that date. NULL when either the variant or the English
        canonical price is absent for that snapshot.

        Returns:
            One row per (scryfall_id, snapshot_date) for non-English language variant
            cards that have a canonical_uuid match in silver_prices_history.
        """
        silver_tables = get_tables(self._silver_con)
        if "silver_language_prices_history" not in silver_tables:
            return pd.DataFrame()
        if "silver_prices_history" not in silver_tables:
            return pd.DataFrame()

        return self._silver_con.execute("""
            SELECT
                lp.scryfall_id,
                lp.canonical_uuid,
                lp.lang,
                lp.snapshot_date,
                lp.eur                                  AS lang_eur,
                lp.eur_foil                             AS lang_eur_foil,
                lp.usd                                  AS lang_usd,
                lp.usd_foil                             AS lang_usd_foil,
                ep.eur                                  AS canonical_eur,
                ep.eur_foil                             AS canonical_eur_foil,
                lp.eur      / NULLIF(ep.eur,      0)   AS eur_lang_premium,
                lp.eur_foil / NULLIF(ep.eur_foil, 0)   AS eur_foil_lang_premium
            FROM silver_language_prices_history lp
            JOIN silver_prices_history ep
                ON  lp.canonical_uuid = ep.uuid
                AND lp.snapshot_date  = ep.snapshot_date
            ORDER BY lp.canonical_uuid, lp.lang, lp.snapshot_date
        """).df()

    def build_price_features(self) -> pd.DataFrame:
        """Build gold_price_features from silver_prices_history.

        Computes rolling averages, lag-based price changes, volatility, spread,
        and relative ranking features using DuckDB window functions over the
        full history. All features are row-based (not date-range-based) and
        assume ~daily snapshot cadence; gaps in snapshots shift the effective
        window.

        The three lag values (lag_1d, lag_7d, lag_30d) are pre-computed once in
        a ``price_lags`` CTE. The outer SELECT references them by alias to avoid
        repeating LAG() expressions — each repeated call in a SELECT list is
        evaluated independently by some engines, so the CTE approach is both
        cleaner and unambiguously correct.

        price_rank_global ranks each card by EUR price among all cards with a
        price on that snapshot date — rank 1 is the most expensive. NULL prices
        are ranked last (NULLS LAST). Ties receive the same rank.

        is_price_spike flags day-over-day EUR changes exceeding 300% (ratio > 3.0).
        Genuine ban/unban events can cross this threshold, so the flag is a hint
        for downstream inspection rather than automatic removal. NULL on the first
        row per uuid (no prior price to compare).

        Returns:
            One row per (uuid, snapshot_date) with price trend and ranking feature columns.
        """
        # edhrec_rank is time-aligned from silver_meta_history so each row reflects the
        # rank current on that exact snapshot date.  If silver_meta_history is absent
        # (e.g. partial Silver tier) the column is emitted as NULL rather than raising a
        # CatalogException — downstream consumers must already handle NULL edhrec_rank.
        silver_tables = get_tables(self._silver_con)
        has_meta = "silver_meta_history" in silver_tables
        edhrec_col = "m.edhrec_rank," if has_meta else "NULL AS edhrec_rank,"
        meta_join = (
            """LEFT JOIN silver_meta_history m
                ON  p.scryfall_id  = m.id
                AND p.snapshot_date = m.snapshot_date"""
            if has_meta
            else ""
        )

        return self._silver_con.execute(f"""
            WITH price_lags AS (
                SELECT
                    *,
                    LAG(eur,  1) OVER (PARTITION BY uuid ORDER BY snapshot_date) AS lag_1d,
                    LAG(eur,  7) OVER (PARTITION BY uuid ORDER BY snapshot_date) AS lag_7d,
                    LAG(eur, 30) OVER (PARTITION BY uuid ORDER BY snapshot_date) AS lag_30d
                FROM silver_prices_history
                WHERE uuid IS NOT NULL
            )
            SELECT
                p.uuid,
                p.scryfall_id,
                p.snapshot_date,
                p.eur,
                p.eur_foil,
                p.usd,
                p.usd_foil,
                p.cardmarket_eur,
                p.cardmarket_eur_foil,
                -- cardmarket_buylist_eur and tcgplayer_buylist_usd omitted:
                -- 100 % NULL in current data (buylist source not yet ingested).
                p.tcgplayer_usd,
                p.tcgplayer_usd_foil,

                {edhrec_col}

                AVG(p.eur) OVER w7  AS price_7d_avg,
                AVG(p.eur) OVER w30 AS price_30d_avg,

                p.eur - p.lag_1d  AS price_change_1d_abs,
                p.eur - p.lag_7d  AS price_change_7d_abs,
                p.eur - p.lag_30d AS price_change_30d_abs,

                (p.eur - p.lag_1d)  / NULLIF(p.lag_1d,  0) AS price_change_1d_pct,
                (p.eur - p.lag_7d)  / NULLIF(p.lag_7d,  0) AS price_change_7d_pct,
                (p.eur - p.lag_30d) / NULLIF(p.lag_30d, 0) AS price_change_30d_pct,

                STDDEV(p.eur) OVER w30 AS price_volatility_30d,

                p.eur_foil / NULLIF(p.eur, 0) AS foil_premium,

                -- Bounded historical windows: ROWS BETWEEN UNBOUNDED PRECEDING AND
                -- CURRENT ROW ensures only past + current data is used, eliminating
                -- the future-leakage that an unordered PARTITION BY uuid produced.
                MAX(p.eur) OVER w_hist   AS price_ath,
                MIN(p.eur) OVER w_hist   AS price_atl,
                COUNT(p.eur) OVER w_hist AS days_with_price,
                DATEDIFF('day',
                    MAX(CASE WHEN p.eur IS NOT NULL THEN p.snapshot_date::DATE END)
                        OVER w_hist,
                    p.snapshot_date::DATE
                ) AS days_since_last_real_price,

                RANK() OVER (
                    PARTITION BY p.snapshot_date ORDER BY p.eur DESC NULLS LAST
                ) AS price_rank_global,

                ABS((p.eur - p.lag_1d) / NULLIF(p.lag_1d, 0)) > 3.0 AS is_price_spike

            FROM price_lags p
            {meta_join}
            WINDOW
                w7     AS (PARTITION BY p.uuid ORDER BY p.snapshot_date
                            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW),
                w30    AS (PARTITION BY p.uuid ORDER BY p.snapshot_date
                            ROWS BETWEEN 29 PRECEDING AND CURRENT ROW),
                w_hist AS (PARTITION BY p.uuid ORDER BY p.snapshot_date
                            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
            ORDER BY p.uuid, p.snapshot_date
        """).df()
