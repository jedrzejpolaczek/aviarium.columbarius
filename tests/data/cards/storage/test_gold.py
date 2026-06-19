import datetime
import json
import logging
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from src.data.cards.storage.gold import GoldStorage
from src.data.cards.storage.gold.ml_dataset import GoldMLDatasetBuilder
from src.data.cards.storage.gold.writers import GoldWriter


def _make_gold_storage(
    tmp_path: Path, silver_tables: dict[str, pd.DataFrame]
) -> GoldStorage:
    """GoldStorage backed by a pre-populated Silver file and an in-memory Gold DB."""
    silver_path = str(tmp_path / "silver.duckdb")
    con = duckdb.connect(silver_path)
    for table_name, df in silver_tables.items():
        con.register("_df", df)
        con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM _df")
        con.unregister("_df")
    con.close()

    config_path = tmp_path / "gold_config.json"
    config_path.write_text("{}")
    return GoldStorage(silver_path, ":memory:", str(config_path))


@pytest.fixture
def staples_df() -> pd.DataFrame:
    """Three daily snapshots for one card in one format."""
    return pd.DataFrame(
        [
            {
                "id": "CardA__commander",
                "card_name": "CardA",
                "format": "commander",
                "snapshot_date": "2026-05-01",
                "deck_pct": 10.0,
                "played": 100,
                "top": 5,
            },
            {
                "id": "CardA__commander",
                "card_name": "CardA",
                "format": "commander",
                "snapshot_date": "2026-05-02",
                "deck_pct": 12.0,
                "played": 120,
                "top": 4,
            },
            {
                "id": "CardA__commander",
                "card_name": "CardA",
                "format": "commander",
                "snapshot_date": "2026-05-03",
                "deck_pct": 14.0,
                "played": 140,
                "top": 3,
            },
        ]
    )


# ---------------------------------------------------------------------------
# GoldSignalBuilders.build_format_staples
# ---------------------------------------------------------------------------


class TestBuildFormatStaples:
    def test_returns_expected_columns(self, tmp_path, staples_df):
        with _make_gold_storage(
            tmp_path, {"silver_format_staples_history": staples_df}
        ) as g:
            result = g._signals.build_format_staples()

        expected = {
            "id",
            "card_name",
            "format",
            "snapshot_date",
            "deck_pct",
            "played",
            "top",
            "deck_pct_7d_avg",
            "deck_pct_30d_avg",
            "deck_pct_change_7d",
            "deck_pct_change_30d",
        }
        assert set(result.columns) == expected

    def test_row_count_equals_input(self, tmp_path, staples_df):
        with _make_gold_storage(
            tmp_path, {"silver_format_staples_history": staples_df}
        ) as g:
            result = g._signals.build_format_staples()

        assert len(result) == len(staples_df)

    def test_7d_avg_equals_deck_pct_for_single_row_window(self, tmp_path, staples_df):
        with _make_gold_storage(
            tmp_path, {"silver_format_staples_history": staples_df}
        ) as g:
            result = (
                g._signals.build_format_staples()
                .sort_values("snapshot_date")
                .reset_index(drop=True)
            )

        # Day 1 window has only one row → avg == deck_pct
        assert result.loc[0, "deck_pct_7d_avg"] == pytest.approx(10.0)

    def test_7d_avg_accumulates_over_window(self, tmp_path, staples_df):
        with _make_gold_storage(
            tmp_path, {"silver_format_staples_history": staples_df}
        ) as g:
            result = (
                g._signals.build_format_staples()
                .sort_values("snapshot_date")
                .reset_index(drop=True)
            )

        # Day 2: avg(10, 12) = 11.0; Day 3: avg(10, 12, 14) = 12.0
        assert result.loc[1, "deck_pct_7d_avg"] == pytest.approx(11.0)
        assert result.loc[2, "deck_pct_7d_avg"] == pytest.approx(12.0)

    def test_30d_avg_same_as_7d_when_fewer_than_30_rows(self, tmp_path, staples_df):
        with _make_gold_storage(
            tmp_path, {"silver_format_staples_history": staples_df}
        ) as g:
            result = (
                g._signals.build_format_staples()
                .sort_values("snapshot_date")
                .reset_index(drop=True)
            )

        # With only 3 rows all rows fall inside both windows
        for i in range(len(result)):
            assert result.loc[i, "deck_pct_7d_avg"] == pytest.approx(
                result.loc[i, "deck_pct_30d_avg"]
            )

    def test_lag_7_change_is_null_when_fewer_than_8_rows(self, tmp_path, staples_df):
        with _make_gold_storage(
            tmp_path, {"silver_format_staples_history": staples_df}
        ) as g:
            result = g._signals.build_format_staples()

        # Only 3 rows — no row has 7 predecessors, so all changes are NULL
        assert result["deck_pct_change_7d"].isna().all()
        assert result["deck_pct_change_30d"].isna().all()

    def test_lag_7_computed_correctly_with_enough_history(self, tmp_path):
        rows = [
            {
                "id": "X__modern",
                "card_name": "X",
                "format": "modern",
                "snapshot_date": f"2026-05-{i:02d}",
                "deck_pct": float(i),
                "played": i * 10,
                "top": i,
            }
            for i in range(
                1, 10
            )  # 9 rows → row 8 (day 08) can lag back to row 1 (day 01)
        ]
        df = pd.DataFrame(rows)
        with _make_gold_storage(tmp_path, {"silver_format_staples_history": df}) as g:
            result = (
                g._signals.build_format_staples()
                .sort_values("snapshot_date")
                .reset_index(drop=True)
            )

        # Row index 7 = day 08; deck_pct=8.0, lag(7) → day 01 deck_pct=1.0 → change=7.0
        assert result.loc[7, "deck_pct_change_7d"] == pytest.approx(7.0)
        # Rows 0-6 (days 01-07) must still be NULL
        assert result.loc[:6, "deck_pct_change_7d"].isna().all()

    def test_partitioned_by_id_so_two_cards_do_not_mix_windows(self, tmp_path):
        rows = [
            {
                "id": "A__commander",
                "card_name": "A",
                "format": "commander",
                "snapshot_date": "2026-05-01",
                "deck_pct": 50.0,
                "played": 500,
                "top": 1,
            },
            {
                "id": "B__commander",
                "card_name": "B",
                "format": "commander",
                "snapshot_date": "2026-05-01",
                "deck_pct": 10.0,
                "played": 100,
                "top": 10,
            },
        ]
        df = pd.DataFrame(rows)
        with _make_gold_storage(tmp_path, {"silver_format_staples_history": df}) as g:
            result = g._signals.build_format_staples()

        a_row = result[result["id"] == "A__commander"].iloc[0]
        b_row = result[result["id"] == "B__commander"].iloc[0]
        # Each card's single-row window must equal its own deck_pct, not a mix
        assert a_row["deck_pct_7d_avg"] == pytest.approx(50.0)
        assert b_row["deck_pct_7d_avg"] == pytest.approx(10.0)

    def test_original_columns_pass_through_unchanged(self, tmp_path, staples_df):
        with _make_gold_storage(
            tmp_path, {"silver_format_staples_history": staples_df}
        ) as g:
            result = (
                g._signals.build_format_staples()
                .sort_values("snapshot_date")
                .reset_index(drop=True)
            )

        assert list(result["card_name"]) == ["CardA", "CardA", "CardA"]
        assert list(result["format"]) == ["commander", "commander", "commander"]
        assert list(result["deck_pct"]) == [10.0, 12.0, 14.0]


# ---------------------------------------------------------------------------
# GoldFeatureBuilders.build_price_features
# ---------------------------------------------------------------------------


def _make_full_prices(rows: list[dict]) -> pd.DataFrame:
    """Build a silver_prices_history DataFrame with all columns build_price_features needs."""
    null_cols = [
        "eur_foil",
        "usd",
        "usd_foil",
        "cardmarket_eur",
        "cardmarket_eur_foil",
        "cardmarket_buylist_eur",
        "tcgplayer_usd",
        "tcgplayer_usd_foil",
        "tcgplayer_buylist_usd",
    ]
    return pd.DataFrame(
        [
            {
                "uuid": r["uuid"],
                "scryfall_id": r["scryfall_id"],
                "snapshot_date": r["snapshot_date"],
                "eur": r.get("eur", None),
                **{col: r.get(col, None) for col in null_cols},
            }
            for r in rows
        ]
    )


class TestBuildPriceFeatures:
    def test_price_rank_global_column_present(self, tmp_path):
        df = _make_full_prices(
            [
                {
                    "uuid": "u1",
                    "scryfall_id": "s1",
                    "snapshot_date": "2026-05-01",
                    "eur": 5.0,
                }
            ]
        )
        with _make_gold_storage(tmp_path, {"silver_prices_history": df}) as g:
            result = g._features.build_price_features()

        assert "price_rank_global" in result.columns

    def test_rank_1_for_highest_eur_on_date(self, tmp_path):
        df = _make_full_prices(
            [
                {
                    "uuid": "u1",
                    "scryfall_id": "s1",
                    "snapshot_date": "2026-05-01",
                    "eur": 10.0,
                },
                {
                    "uuid": "u2",
                    "scryfall_id": "s2",
                    "snapshot_date": "2026-05-01",
                    "eur": 5.0,
                },
                {
                    "uuid": "u3",
                    "scryfall_id": "s3",
                    "snapshot_date": "2026-05-01",
                    "eur": 1.0,
                },
            ]
        )
        with _make_gold_storage(tmp_path, {"silver_prices_history": df}) as g:
            result = (
                g._features.build_price_features()
                .sort_values("eur", ascending=False)
                .reset_index(drop=True)
            )

        assert result.loc[0, "price_rank_global"] == 1
        assert result.loc[1, "price_rank_global"] == 2
        assert result.loc[2, "price_rank_global"] == 3

    def test_rank_resets_per_snapshot_date(self, tmp_path):
        df = _make_full_prices(
            [
                {
                    "uuid": "u1",
                    "scryfall_id": "s1",
                    "snapshot_date": "2026-05-01",
                    "eur": 10.0,
                },
                {
                    "uuid": "u2",
                    "scryfall_id": "s2",
                    "snapshot_date": "2026-05-01",
                    "eur": 5.0,
                },
                # day 2: prices flip — u2 is now more expensive
                {
                    "uuid": "u1",
                    "scryfall_id": "s1",
                    "snapshot_date": "2026-05-02",
                    "eur": 3.0,
                },
                {
                    "uuid": "u2",
                    "scryfall_id": "s2",
                    "snapshot_date": "2026-05-02",
                    "eur": 8.0,
                },
            ]
        )
        with _make_gold_storage(tmp_path, {"silver_prices_history": df}) as g:
            result = g._features.build_price_features()

        def rank(uuid: str, date: str) -> int:
            return int(
                result[
                    (result["uuid"] == uuid) & (result["snapshot_date"] == date)
                ].iloc[0]["price_rank_global"]
            )

        assert rank("u1", "2026-05-01") == 1
        assert rank("u2", "2026-05-01") == 2
        assert rank("u2", "2026-05-02") == 1
        assert rank("u1", "2026-05-02") == 2

    def test_ties_receive_same_rank(self, tmp_path):
        df = _make_full_prices(
            [
                {
                    "uuid": "u1",
                    "scryfall_id": "s1",
                    "snapshot_date": "2026-05-01",
                    "eur": 5.0,
                },
                {
                    "uuid": "u2",
                    "scryfall_id": "s2",
                    "snapshot_date": "2026-05-01",
                    "eur": 5.0,
                },
            ]
        )
        with _make_gold_storage(tmp_path, {"silver_prices_history": df}) as g:
            result = g._features.build_price_features()

        assert set(result["price_rank_global"].tolist()) == {1}

    def test_null_eur_ranked_after_priced_cards(self, tmp_path):
        df = _make_full_prices(
            [
                {
                    "uuid": "u1",
                    "scryfall_id": "s1",
                    "snapshot_date": "2026-05-01",
                    "eur": 10.0,
                },
                {
                    "uuid": "u2",
                    "scryfall_id": "s2",
                    "snapshot_date": "2026-05-01",
                    "eur": None,
                },
            ]
        )
        with _make_gold_storage(tmp_path, {"silver_prices_history": df}) as g:
            result = g._features.build_price_features()

        u1_rank = int(result[result["uuid"] == "u1"].iloc[0]["price_rank_global"])
        u2_rank = int(result[result["uuid"] == "u2"].iloc[0]["price_rank_global"])
        assert u1_rank < u2_rank

    def test_is_price_spike_true_for_over_300pct_jump(self, tmp_path):
        df = _make_full_prices(
            [
                {
                    "uuid": "u1",
                    "scryfall_id": "s1",
                    "snapshot_date": "2026-05-01",
                    "eur": 1.0,
                },
                {
                    "uuid": "u1",
                    "scryfall_id": "s1",
                    "snapshot_date": "2026-05-02",
                    "eur": 5.0,
                },
            ]
        )
        with _make_gold_storage(tmp_path, {"silver_prices_history": df}) as g:
            result = (
                g._features.build_price_features()
                .sort_values("snapshot_date")
                .reset_index(drop=True)
            )

        # +400% jump (1.0 → 5.0) exceeds 300% threshold
        assert (
            result.loc[1, "is_price_spike"] is True
            or result.loc[1, "is_price_spike"] == 1
        )

    def test_is_price_spike_false_for_normal_movement(self, tmp_path):
        df = _make_full_prices(
            [
                {
                    "uuid": "u1",
                    "scryfall_id": "s1",
                    "snapshot_date": "2026-05-01",
                    "eur": 5.0,
                },
                {
                    "uuid": "u1",
                    "scryfall_id": "s1",
                    "snapshot_date": "2026-05-02",
                    "eur": 6.0,
                },
            ]
        )
        with _make_gold_storage(tmp_path, {"silver_prices_history": df}) as g:
            result = (
                g._features.build_price_features()
                .sort_values("snapshot_date")
                .reset_index(drop=True)
            )

        # +20% is not a spike
        assert not result.loc[1, "is_price_spike"]

    def test_is_price_spike_null_on_first_row(self, tmp_path):
        df = _make_full_prices(
            [
                {
                    "uuid": "u1",
                    "scryfall_id": "s1",
                    "snapshot_date": "2026-05-01",
                    "eur": 5.0,
                },
                {
                    "uuid": "u1",
                    "scryfall_id": "s1",
                    "snapshot_date": "2026-05-02",
                    "eur": 6.0,
                },
            ]
        )
        with _make_gold_storage(tmp_path, {"silver_prices_history": df}) as g:
            result = (
                g._features.build_price_features()
                .sort_values("snapshot_date")
                .reset_index(drop=True)
            )

        # First row has no prior price — spike flag is NULL
        assert pd.isna(result.loc[0, "is_price_spike"])

    def test_price_change_1d_pct_computed_correctly(self, tmp_path):
        df = _make_full_prices(
            [
                {
                    "uuid": "u1",
                    "scryfall_id": "s1",
                    "snapshot_date": "2026-05-01",
                    "eur": 4.0,
                },
                {
                    "uuid": "u1",
                    "scryfall_id": "s1",
                    "snapshot_date": "2026-05-02",
                    "eur": 5.0,
                },
            ]
        )
        with _make_gold_storage(tmp_path, {"silver_prices_history": df}) as g:
            result = (
                g._features.build_price_features()
                .sort_values("snapshot_date")
                .reset_index(drop=True)
            )

        # (5 - 4) / 4 = 0.25
        assert result.loc[1, "price_change_1d_pct"] == pytest.approx(0.25)

    def test_spike_exactly_at_threshold_is_not_spike(self, tmp_path):
        # ABS(pct) > 3.0, so exactly 3.0 (= 300%) is NOT a spike
        df = _make_full_prices(
            [
                {
                    "uuid": "u1",
                    "scryfall_id": "s1",
                    "snapshot_date": "2026-05-01",
                    "eur": 1.0,
                },
                {
                    "uuid": "u1",
                    "scryfall_id": "s1",
                    "snapshot_date": "2026-05-02",
                    "eur": 4.0,
                },
            ]
        )
        with _make_gold_storage(tmp_path, {"silver_prices_history": df}) as g:
            result = (
                g._features.build_price_features()
                .sort_values("snapshot_date")
                .reset_index(drop=True)
            )

        # (4 - 1) / 1 = 3.0 exactly — not > 3.0
        assert not result.loc[1, "is_price_spike"]


# ---------------------------------------------------------------------------
# GoldStorage._pipeline — format staples branch
# ---------------------------------------------------------------------------


class TestPipelineFormatStaples:
    def test_creates_gold_format_staples_when_silver_history_present(
        self, tmp_path, staples_df
    ):
        with _make_gold_storage(
            tmp_path, {"silver_format_staples_history": staples_df}
        ) as g:
            g.populate()
            tables = {r[0] for r in g._gold_con.execute("SHOW TABLES").fetchall()}

        assert "gold_format_staples" in tables

    def test_gold_format_staples_row_count_matches_input(self, tmp_path, staples_df):
        with _make_gold_storage(
            tmp_path, {"silver_format_staples_history": staples_df}
        ) as g:
            g.populate()
            row = g._gold_con.execute(
                "SELECT count(*) FROM gold_format_staples"
            ).fetchone()

        assert row is not None and row[0] == len(staples_df)

    def test_skips_gold_format_staples_when_silver_history_absent(self, tmp_path):
        with _make_gold_storage(tmp_path, {}) as g:
            g.populate()
            tables = {r[0] for r in g._gold_con.execute("SHOW TABLES").fetchall()}

        assert "gold_format_staples" not in tables

    def test_pipeline_does_not_raise_when_all_silver_tables_absent(self, tmp_path):
        with _make_gold_storage(tmp_path, {}) as g:
            g.populate()  # should complete without error


# ---------------------------------------------------------------------------
# Helpers for GoldSignalBuilders.build_ban_price_impact tests
# ---------------------------------------------------------------------------


def _make_meta_history(rows: list[dict]) -> pd.DataFrame:
    """Build a silver_meta_history DataFrame from minimal row dicts.

    legalities is serialised as a JSON string to match how Silver stores it
    (DuckDB serialises Python dicts back to VARCHAR when writing to the table).
    """
    return pd.DataFrame(
        [
            {
                "id": r["id"],
                "snapshot_date": r["snapshot_date"],
                "legalities": json.dumps(r.get("legalities", {})),
                "edhrec_rank": r.get("edhrec_rank", None),
            }
            for r in rows
        ]
    )


def _make_prices_history(rows: list[dict]) -> pd.DataFrame:
    """Build a silver_prices_history DataFrame with all columns build_price_features expects."""
    null_cols = [
        "uuid",
        "eur_foil",
        "usd",
        "usd_foil",
        "cardmarket_eur",
        "cardmarket_eur_foil",
        "cardmarket_buylist_eur",
        "tcgplayer_usd",
        "tcgplayer_usd_foil",
        "tcgplayer_buylist_usd",
    ]
    return pd.DataFrame(
        [
            {
                "scryfall_id": r["scryfall_id"],
                "snapshot_date": r["snapshot_date"],
                "eur": r.get("eur", None),
                **{col: r.get(col, None) for col in null_cols},
            }
            for r in rows
        ]
    )


# ---------------------------------------------------------------------------
# GoldSignalBuilders.build_events
# ---------------------------------------------------------------------------


class TestBuildEvents:
    def test_returns_expected_columns(self, tmp_path):
        empty_meta = pd.DataFrame(columns=["id", "snapshot_date", "legalities"])
        with _make_gold_storage(tmp_path, {"silver_meta_history": empty_meta}) as g:
            result = g._signals.build_events()
        assert set(result.columns) == {
            "event_date",
            "format",
            "event_type",
            "card_count",
        }

    def test_returns_empty_when_no_transitions(self, tmp_path):
        meta = _make_meta_history(
            [
                {
                    "id": "s1",
                    "snapshot_date": "2026-05-01",
                    "legalities": {"commander": "legal"},
                },
                {
                    "id": "s1",
                    "snapshot_date": "2026-05-02",
                    "legalities": {"commander": "legal"},
                },
            ]
        )
        with _make_gold_storage(tmp_path, {"silver_meta_history": meta}) as g:
            result = g._signals.build_events()
        assert result.empty

    def test_detects_ban_transition(self, tmp_path):
        meta = _make_meta_history(
            [
                {
                    "id": "s1",
                    "snapshot_date": "2026-05-01",
                    "legalities": {"modern": "legal"},
                },
                {
                    "id": "s1",
                    "snapshot_date": "2026-05-02",
                    "legalities": {"modern": "banned"},
                },
            ]
        )
        with _make_gold_storage(tmp_path, {"silver_meta_history": meta}) as g:
            result = g._signals.build_events()
        assert len(result) == 1
        row = result.iloc[0]
        assert row["event_date"] == "2026-05-02"
        assert row["format"] == "modern"
        assert row["event_type"] == "ban"
        assert row["card_count"] == 1

    def test_detects_unban_transition(self, tmp_path):
        meta = _make_meta_history(
            [
                {
                    "id": "s1",
                    "snapshot_date": "2026-05-01",
                    "legalities": {"legacy": "banned"},
                },
                {
                    "id": "s1",
                    "snapshot_date": "2026-05-02",
                    "legalities": {"legacy": "legal"},
                },
            ]
        )
        with _make_gold_storage(tmp_path, {"silver_meta_history": meta}) as g:
            result = g._signals.build_events()
        assert len(result) == 1
        assert result.iloc[0]["event_type"] == "unban"

    def test_aggregates_card_count_for_same_format_and_date(self, tmp_path):
        meta = _make_meta_history(
            [
                {
                    "id": "s1",
                    "snapshot_date": "2026-05-01",
                    "legalities": {"commander": "legal"},
                },
                {
                    "id": "s1",
                    "snapshot_date": "2026-05-02",
                    "legalities": {"commander": "banned"},
                },
                {
                    "id": "s2",
                    "snapshot_date": "2026-05-01",
                    "legalities": {"commander": "legal"},
                },
                {
                    "id": "s2",
                    "snapshot_date": "2026-05-02",
                    "legalities": {"commander": "banned"},
                },
            ]
        )
        with _make_gold_storage(tmp_path, {"silver_meta_history": meta}) as g:
            result = g._signals.build_events()
        assert len(result) == 1
        assert result.iloc[0]["card_count"] == 2

    def test_different_formats_produce_separate_rows(self, tmp_path):
        meta = _make_meta_history(
            [
                {
                    "id": "s1",
                    "snapshot_date": "2026-05-01",
                    "legalities": {"modern": "legal", "legacy": "legal"},
                },
                {
                    "id": "s1",
                    "snapshot_date": "2026-05-02",
                    "legalities": {"modern": "banned", "legacy": "banned"},
                },
            ]
        )
        with _make_gold_storage(tmp_path, {"silver_meta_history": meta}) as g:
            result = g._signals.build_events()
        assert len(result) == 2
        assert set(result["format"]) == {"modern", "legacy"}

    def test_pipeline_writes_gold_events_table(self, tmp_path):
        meta = _make_meta_history(
            [
                {
                    "id": "s1",
                    "snapshot_date": "2026-05-01",
                    "legalities": {"modern": "legal"},
                },
                {
                    "id": "s1",
                    "snapshot_date": "2026-05-02",
                    "legalities": {"modern": "banned"},
                },
            ]
        )
        with _make_gold_storage(tmp_path, {"silver_meta_history": meta}) as g:
            g.populate()
            tables = {r[0] for r in g._gold_con.execute("SHOW TABLES").fetchall()}
        assert "gold_events" in tables

    def test_returns_empty_when_all_cards_have_single_legality_value(self, tmp_path):
        # All cards have the same legality every day — no transitions possible.
        # _has_legality_transitions returns False, so _load_and_parse_meta is never called.
        meta = _make_meta_history(
            [
                {
                    "id": "s1",
                    "snapshot_date": "2026-05-01",
                    "legalities": {"commander": "legal"},
                },
                {
                    "id": "s1",
                    "snapshot_date": "2026-05-02",
                    "legalities": {"commander": "legal"},
                },
                {
                    "id": "s2",
                    "snapshot_date": "2026-05-01",
                    "legalities": {"modern": "legal"},
                },
                {
                    "id": "s2",
                    "snapshot_date": "2026-05-02",
                    "legalities": {"modern": "legal"},
                },
            ]
        )
        with _make_gold_storage(tmp_path, {"silver_meta_history": meta}) as g:
            result = g._signals.build_events()
        assert result.empty
        assert list(result.columns) == [
            "event_date",
            "format",
            "event_type",
            "card_count",
        ]


# ---------------------------------------------------------------------------
# GoldSignalBuilders.build_ban_price_impact
# ---------------------------------------------------------------------------


class TestBuildBanPriceImpact:
    def test_returns_empty_when_meta_history_empty(self, tmp_path):
        empty_meta = pd.DataFrame(
            columns=["id", "snapshot_date", "legalities", "edhrec_rank"]
        )
        with _make_gold_storage(tmp_path, {"silver_meta_history": empty_meta}) as g:
            result = g._signals.build_ban_price_impact()
        assert result.empty

    def test_returns_empty_when_no_legality_changes(self, tmp_path):
        meta = _make_meta_history(
            [
                {
                    "id": "s1",
                    "snapshot_date": "2026-05-01",
                    "legalities": {"commander": "legal"},
                },
                {
                    "id": "s1",
                    "snapshot_date": "2026-05-02",
                    "legalities": {"commander": "legal"},
                },
            ]
        )
        with _make_gold_storage(tmp_path, {"silver_meta_history": meta}) as g:
            result = g._signals.build_ban_price_impact()
        assert result.empty

    def test_detects_ban_event(self, tmp_path):
        meta = _make_meta_history(
            [
                {
                    "id": "s1",
                    "snapshot_date": "2026-05-01",
                    "legalities": {"commander": "legal"},
                },
                {
                    "id": "s1",
                    "snapshot_date": "2026-05-02",
                    "legalities": {"commander": "banned"},
                },
            ]
        )
        with _make_gold_storage(tmp_path, {"silver_meta_history": meta}) as g:
            result = g._signals.build_ban_price_impact()
        assert len(result) == 1
        row = result.iloc[0]
        assert row["scryfall_id"] == "s1"
        assert row["format"] == "commander"
        assert row["event_type"] == "ban"
        assert row["event_date"] == "2026-05-02"

    def test_detects_unban_event(self, tmp_path):
        meta = _make_meta_history(
            [
                {
                    "id": "s1",
                    "snapshot_date": "2026-05-01",
                    "legalities": {"modern": "banned"},
                },
                {
                    "id": "s1",
                    "snapshot_date": "2026-05-02",
                    "legalities": {"modern": "legal"},
                },
            ]
        )
        with _make_gold_storage(tmp_path, {"silver_meta_history": meta}) as g:
            result = g._signals.build_ban_price_impact()
        assert len(result) == 1
        assert result.iloc[0]["event_type"] == "unban"

    def test_detects_events_across_multiple_formats(self, tmp_path):
        meta = _make_meta_history(
            [
                {
                    "id": "s1",
                    "snapshot_date": "2026-05-01",
                    "legalities": {"commander": "legal", "modern": "legal"},
                },
                {
                    "id": "s1",
                    "snapshot_date": "2026-05-02",
                    "legalities": {"commander": "banned", "modern": "banned"},
                },
            ]
        )
        with _make_gold_storage(tmp_path, {"silver_meta_history": meta}) as g:
            result = g._signals.build_ban_price_impact()
        assert len(result) == 2
        assert set(result["format"]) == {"commander", "modern"}

    def test_two_cards_events_are_independent(self, tmp_path):
        meta = _make_meta_history(
            [
                {
                    "id": "s1",
                    "snapshot_date": "2026-05-01",
                    "legalities": {"legacy": "legal"},
                },
                {
                    "id": "s1",
                    "snapshot_date": "2026-05-02",
                    "legalities": {"legacy": "banned"},
                },
                {
                    "id": "s2",
                    "snapshot_date": "2026-05-01",
                    "legalities": {"legacy": "legal"},
                },
                {
                    "id": "s2",
                    "snapshot_date": "2026-05-02",
                    "legalities": {"legacy": "legal"},
                },
            ]
        )
        with _make_gold_storage(tmp_path, {"silver_meta_history": meta}) as g:
            result = g._signals.build_ban_price_impact()
        assert len(result) == 1
        assert result.iloc[0]["scryfall_id"] == "s1"

    def test_price_at_event_filled_from_prices_history(self, tmp_path):
        meta = _make_meta_history(
            [
                {
                    "id": "s1",
                    "snapshot_date": "2026-05-01",
                    "legalities": {"standard": "legal"},
                },
                {
                    "id": "s1",
                    "snapshot_date": "2026-05-02",
                    "legalities": {"standard": "banned"},
                },
            ]
        )
        prices = _make_prices_history(
            [{"scryfall_id": "s1", "snapshot_date": "2026-05-02", "eur": 5.0}]
        )
        with _make_gold_storage(
            tmp_path,
            {"silver_meta_history": meta, "silver_prices_history": prices},
        ) as g:
            result = g._signals.build_ban_price_impact()
        assert result.iloc[0]["price_at_event"] == pytest.approx(5.0)

    def test_price_7d_before_averages_preceding_window(self, tmp_path):
        meta = _make_meta_history(
            [
                {
                    "id": "s1",
                    "snapshot_date": "2026-05-01",
                    "legalities": {"vintage": "legal"},
                },
                {
                    "id": "s1",
                    "snapshot_date": "2026-05-10",
                    "legalities": {"vintage": "banned"},
                },
            ]
        )
        prices = _make_prices_history(
            [
                {
                    "scryfall_id": "s1",
                    "snapshot_date": "2026-05-03",
                    "eur": 4.0,
                },  # 7 days before
                {
                    "scryfall_id": "s1",
                    "snapshot_date": "2026-05-09",
                    "eur": 6.0,
                },  # 1 day before
                {
                    "scryfall_id": "s1",
                    "snapshot_date": "2026-05-10",
                    "eur": 8.0,
                },  # event day
            ]
        )
        with _make_gold_storage(
            tmp_path,
            {"silver_meta_history": meta, "silver_prices_history": prices},
        ) as g:
            result = g._signals.build_ban_price_impact()
        # price_7d_before covers [-7, -1] → days 03 and 09 → avg(4.0, 6.0) = 5.0
        assert result.iloc[0]["price_7d_before"] == pytest.approx(5.0)

    def test_price_7d_after_averages_following_window(self, tmp_path):
        meta = _make_meta_history(
            [
                {
                    "id": "s1",
                    "snapshot_date": "2026-05-01",
                    "legalities": {"legacy": "legal"},
                },
                {
                    "id": "s1",
                    "snapshot_date": "2026-05-10",
                    "legalities": {"legacy": "banned"},
                },
            ]
        )
        prices = _make_prices_history(
            [
                {"scryfall_id": "s1", "snapshot_date": "2026-05-11", "eur": 2.0},
                {"scryfall_id": "s1", "snapshot_date": "2026-05-17", "eur": 6.0},
                {
                    "scryfall_id": "s1",
                    "snapshot_date": "2026-05-18",
                    "eur": 99.0,
                },  # outside +7
            ]
        )
        with _make_gold_storage(
            tmp_path,
            {"silver_meta_history": meta, "silver_prices_history": prices},
        ) as g:
            result = g._signals.build_ban_price_impact()
        # price_7d_after covers [+1, +7] → days 11 and 17 → avg(2.0, 6.0) = 4.0
        assert result.iloc[0]["price_7d_after"] == pytest.approx(4.0)

    def test_price_change_pct_computed_correctly(self, tmp_path):
        meta = _make_meta_history(
            [
                {
                    "id": "s1",
                    "snapshot_date": "2026-05-01",
                    "legalities": {"modern": "banned"},
                },
                {
                    "id": "s1",
                    "snapshot_date": "2026-05-10",
                    "legalities": {"modern": "legal"},
                },
            ]
        )
        prices = _make_prices_history(
            [
                {
                    "scryfall_id": "s1",
                    "snapshot_date": "2026-05-09",
                    "eur": 10.0,
                },  # 1d before
                {
                    "scryfall_id": "s1",
                    "snapshot_date": "2026-05-11",
                    "eur": 15.0,
                },  # 1d after
            ]
        )
        with _make_gold_storage(
            tmp_path,
            {"silver_meta_history": meta, "silver_prices_history": prices},
        ) as g:
            result = g._signals.build_ban_price_impact()
        # before=10.0, after=15.0 → change = (15-10)/10 = 0.5
        assert result.iloc[0]["price_change_7d_pct"] == pytest.approx(0.5)

    def test_price_columns_null_when_no_price_history(self, tmp_path):
        meta = _make_meta_history(
            [
                {
                    "id": "s1",
                    "snapshot_date": "2026-05-01",
                    "legalities": {"commander": "legal"},
                },
                {
                    "id": "s1",
                    "snapshot_date": "2026-05-02",
                    "legalities": {"commander": "banned"},
                },
            ]
        )
        with _make_gold_storage(tmp_path, {"silver_meta_history": meta}) as g:
            result = g._signals.build_ban_price_impact()
        price_cols = [
            "price_30d_before",
            "price_7d_before",
            "price_at_event",
            "price_7d_after",
            "price_30d_after",
            "price_change_7d_pct",
            "price_change_30d_pct",
        ]
        for col in price_cols:
            assert pd.isna(result.iloc[0][col]), f"{col} should be NULL"

    def test_output_has_expected_columns(self, tmp_path):
        meta = _make_meta_history(
            [
                {
                    "id": "s1",
                    "snapshot_date": "2026-05-01",
                    "legalities": {"commander": "legal"},
                },
                {
                    "id": "s1",
                    "snapshot_date": "2026-05-02",
                    "legalities": {"commander": "banned"},
                },
            ]
        )
        with _make_gold_storage(tmp_path, {"silver_meta_history": meta}) as g:
            result = g._signals.build_ban_price_impact()
        expected = [
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
        assert list(result.columns) == expected

    def test_returns_empty_when_no_legality_transitions(self, tmp_path):
        # No transitions in meta — pre-check must short-circuit before loading prices.
        meta = _make_meta_history(
            [
                {
                    "id": "s1",
                    "snapshot_date": "2026-05-01",
                    "legalities": {"modern": "legal"},
                },
                {
                    "id": "s1",
                    "snapshot_date": "2026-05-02",
                    "legalities": {"modern": "legal"},
                },
            ]
        )
        prices = pd.DataFrame(columns=["scryfall_id", "snapshot_date", "eur"])
        with _make_gold_storage(
            tmp_path,
            {"silver_meta_history": meta, "silver_prices_history": prices},
        ) as g:
            result = g._signals.build_ban_price_impact()
        assert result.empty
        assert set(result.columns) == set(
            [
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
        )


# ---------------------------------------------------------------------------
# GoldStorage._pipeline — ban price impact branch
# ---------------------------------------------------------------------------


class TestPipelineBanPriceImpact:
    def test_creates_gold_ban_price_impact_when_both_silver_tables_present(
        self, tmp_path
    ):
        meta = _make_meta_history(
            [
                {
                    "id": "s1",
                    "snapshot_date": "2026-05-01",
                    "legalities": {"commander": "legal"},
                },
                {
                    "id": "s1",
                    "snapshot_date": "2026-05-02",
                    "legalities": {"commander": "banned"},
                },
            ]
        )
        prices = _make_prices_history(
            [{"scryfall_id": "s1", "snapshot_date": "2026-05-02", "eur": 5.0}]
        )
        with _make_gold_storage(
            tmp_path,
            {"silver_meta_history": meta, "silver_prices_history": prices},
        ) as g:
            g.populate()
            tables = {r[0] for r in g._gold_con.execute("SHOW TABLES").fetchall()}
        assert "gold_ban_price_impact" in tables

    def test_skips_when_silver_prices_history_absent(self, tmp_path):
        meta = _make_meta_history(
            [
                {
                    "id": "s1",
                    "snapshot_date": "2026-05-01",
                    "legalities": {"commander": "legal"},
                },
                {
                    "id": "s1",
                    "snapshot_date": "2026-05-02",
                    "legalities": {"commander": "banned"},
                },
            ]
        )
        with _make_gold_storage(tmp_path, {"silver_meta_history": meta}) as g:
            g.populate()
            tables = {r[0] for r in g._gold_con.execute("SHOW TABLES").fetchall()}
        assert "gold_ban_price_impact" not in tables


# ---------------------------------------------------------------------------
# GoldWriter.write_table
# ---------------------------------------------------------------------------


class TestGoldWriter:
    def test_creates_table_with_correct_row_count(self):
        con = duckdb.connect(":memory:")
        writer = GoldWriter(con)
        writer.full_load(pd.DataFrame({"a": [1, 2, 3]}), "t")
        row = con.execute("SELECT count(*) FROM t").fetchone()
        con.close()
        assert row is not None
        assert row[0] == 3

    def test_skips_empty_dataframe_and_leaves_no_table(self):
        con = duckdb.connect(":memory:")
        writer = GoldWriter(con)
        writer.full_load(pd.DataFrame(), "t")
        tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
        con.close()
        assert "t" not in tables

    def test_replaces_existing_table_fully(self):
        """A second write to the same table replaces all rows (CREATE OR REPLACE)."""
        con = duckdb.connect(":memory:")
        writer = GoldWriter(con)
        writer.full_load(pd.DataFrame({"x": [1, 2, 3]}), "t")
        writer.full_load(pd.DataFrame({"x": [99]}), "t")
        count_row = con.execute("SELECT count(*) FROM t").fetchone()
        val_row = con.execute("SELECT x FROM t").fetchone()
        con.close()
        assert count_row is not None and val_row is not None
        assert count_row[0] == 1
        assert val_row[0] == 99


# ---------------------------------------------------------------------------
# Helpers for GoldFeatureBuilders.build_card_features tests
# ---------------------------------------------------------------------------


def _make_silver_cards(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal silver_cards DataFrame for build_card_features tests.

    Reflects the post-refactor Silver schema:
    - List columns are native Python lists (VARCHAR[] in DuckDB), not JSON strings.
    - legalities dict is gone; replaced by scalar boolean/integer columns.
    """
    return pd.DataFrame(
        [
            {
                "uuid": r.get("uuid"),
                "scryfall_id": r.get("scryfall_id", "s1"),
                "oracle_id": r.get("oracle_id", "o1"),
                "name": r.get("name", "Test Card"),
                "set_code": r.get("set_code", "TST"),
                "rarity": r.get("rarity", "common"),
                "mana_value": r.get("mana_value", 2),
                "is_reserved": r.get("is_reserved", False),
                "is_reprint": r.get("is_reprint", False),
                "is_promo": r.get("is_promo", False),
                "is_full_art": r.get("is_full_art", False),
                "is_textless": r.get("is_textless", False),
                # Native list columns — no json.dumps
                "finishes": r.get("finishes", ["nonfoil"]),
                "colors": r.get("colors", ["W"]),
                "color_identity": r.get("color_identity", ["W"]),
                "variations": r.get("variations", []),
                "original_supertypes": r.get("original_supertypes", []),
                # Scalar legality columns (replace legalities dict)
                "is_commander_legal": r.get("is_commander_legal", True),
                "is_standard_legal": r.get("is_standard_legal", False),
                "is_modern_legal": r.get("is_modern_legal", False),
                "is_legacy_legal": r.get("is_legacy_legal", False),
                "format_count": r.get("format_count", 1),
            }
            for r in rows
        ]
    )


# ---------------------------------------------------------------------------
# GoldFeatureBuilders.build_card_features
# ---------------------------------------------------------------------------


class TestBuildCardFeatures:
    def test_excludes_rows_with_null_uuid(self, tmp_path):
        df = _make_silver_cards(
            [{"uuid": "u1"}, {"uuid": None}]  # second row must be excluded
        )
        with _make_gold_storage(tmp_path, {"silver_cards": df}) as g:
            result = g._features.build_card_features()
        assert len(result) == 1
        assert result.iloc[0]["uuid"] == "u1"

    def test_returns_expected_columns(self, tmp_path):
        df = _make_silver_cards([{"uuid": "u1"}])
        with _make_gold_storage(tmp_path, {"silver_cards": df}) as g:
            result = g._features.build_card_features()
        expected = {
            "uuid",
            "scryfall_id",
            "oracle_id",
            "name",
            "set_code",
            "rarity",
            "mana_value",
            "is_reserved",
            "is_reprint",
            "is_promo",
            "is_full_art",
            "is_textless",
            "finish_count",
            "has_etched_finish",
            "color_count",
            "color_identity_count",
            "variation_count",
            "is_legendary",
            "format_count",
            "is_commander_legal",
            "is_standard_legal",
            "is_modern_legal",
            "is_legacy_legal",
            "print_count",
        }
        assert expected.issubset(set(result.columns))

    def test_is_legendary_true_when_legendary_in_supertypes(self, tmp_path):
        df = _make_silver_cards(
            [{"uuid": "u1", "original_supertypes": ["Legendary", "Snow"]}]
        )
        with _make_gold_storage(tmp_path, {"silver_cards": df}) as g:
            result = g._features.build_card_features()
        assert result.iloc[0]["is_legendary"] == True  # noqa: E712

    def test_is_legendary_false_when_legendary_not_in_supertypes(self, tmp_path):
        df = _make_silver_cards([{"uuid": "u1", "original_supertypes": ["Basic"]}])
        with _make_gold_storage(tmp_path, {"silver_cards": df}) as g:
            result = g._features.build_card_features()
        assert result.iloc[0]["is_legendary"] == False  # noqa: E712

    def test_is_legendary_false_when_supertypes_empty(self, tmp_path):
        df = _make_silver_cards([{"uuid": "u1", "original_supertypes": []}])
        with _make_gold_storage(tmp_path, {"silver_cards": df}) as g:
            result = g._features.build_card_features()
        assert result.iloc[0]["is_legendary"] == False  # noqa: E712

    def test_mana_value_over_20_becomes_null(self, tmp_path):
        df = _make_silver_cards([{"uuid": "u1", "mana_value": 1_000_000}])
        with _make_gold_storage(tmp_path, {"silver_cards": df}) as g:
            result = g._features.build_card_features()
        assert pd.isna(result.iloc[0]["mana_value"])

    def test_mana_value_at_20_is_kept(self, tmp_path):
        df = _make_silver_cards([{"uuid": "u1", "mana_value": 20}])
        with _make_gold_storage(tmp_path, {"silver_cards": df}) as g:
            result = g._features.build_card_features()
        assert result.iloc[0]["mana_value"] == 20

    def test_is_commander_legal_true_when_legality_is_legal(self, tmp_path):
        df = _make_silver_cards([{"uuid": "u1", "is_commander_legal": True}])
        with _make_gold_storage(tmp_path, {"silver_cards": df}) as g:
            result = g._features.build_card_features()
        assert result.iloc[0]["is_commander_legal"] == True  # noqa: E712

    def test_is_commander_legal_false_when_legality_is_banned(self, tmp_path):
        df = _make_silver_cards([{"uuid": "u1", "is_commander_legal": False}])
        with _make_gold_storage(tmp_path, {"silver_cards": df}) as g:
            result = g._features.build_card_features()
        assert result.iloc[0]["is_commander_legal"] == False  # noqa: E712

    def test_format_count_reads_silver_scalar(self, tmp_path):
        df = _make_silver_cards([{"uuid": "u1", "format_count": 3}])
        with _make_gold_storage(tmp_path, {"silver_cards": df}) as g:
            result = g._features.build_card_features()
        assert result.iloc[0]["format_count"] == 3

    def test_print_count_reflects_shared_oracle_id(self, tmp_path):
        df = _make_silver_cards(
            [
                {"uuid": "u1", "oracle_id": "o1"},
                {"uuid": "u2", "oracle_id": "o1"},
                {"uuid": "u3", "oracle_id": "o2"},
            ]
        )
        with _make_gold_storage(tmp_path, {"silver_cards": df}) as g:
            result = g._features.build_card_features().set_index("uuid")
        assert result.loc["u1", "print_count"] == 2
        assert result.loc["u2", "print_count"] == 2
        assert result.loc["u3", "print_count"] == 1

    def test_finish_count_and_has_etched_finish(self, tmp_path):
        df = _make_silver_cards(
            [
                {"uuid": "u1", "finishes": ["nonfoil", "etched"]},
                {"uuid": "u2", "finishes": ["nonfoil"]},
            ]
        )
        with _make_gold_storage(tmp_path, {"silver_cards": df}) as g:
            result = g._features.build_card_features().set_index("uuid")
        assert result.loc["u1", "finish_count"] == 2
        assert result.loc["u1", "has_etched_finish"] == True  # noqa: E712
        assert result.loc["u2", "finish_count"] == 1
        assert result.loc["u2", "has_etched_finish"] == False  # noqa: E712

    def test_warns_when_scalar_legality_columns_missing(self, tmp_path, caplog):
        df = _make_silver_cards([{"uuid": "u1"}])
        df = df.drop(columns=["is_commander_legal"])
        with _make_gold_storage(tmp_path, {"silver_cards": df}) as g:
            with caplog.at_level(logging.WARNING):
                result = g._features.build_card_features()
        assert "missing scalar legality columns" in caplog.text
        assert pd.isna(result.iloc[0]["is_commander_legal"])


# ---------------------------------------------------------------------------
# GoldSignalBuilders.build_demand_signals
# ---------------------------------------------------------------------------


class TestBuildDemandSignals:
    def test_returns_empty_when_meta_history_empty(self, tmp_path):
        empty = pd.DataFrame(
            columns=["id", "snapshot_date", "legalities", "edhrec_rank"]
        )
        with _make_gold_storage(tmp_path, {"silver_meta_history": empty}) as g:
            result = g._signals.build_demand_signals()
        assert result.empty

    def test_legalities_column_not_in_output(self, tmp_path):
        meta = _make_meta_history(
            [{"id": "s1", "snapshot_date": "2026-05-01", "legalities": {}}]
        )
        with _make_gold_storage(tmp_path, {"silver_meta_history": meta}) as g:
            result = g._signals.build_demand_signals()
        assert "legalities" not in result.columns

    def test_detects_commander_ban_transition(self, tmp_path):
        meta = _make_meta_history(
            [
                {
                    "id": "s1",
                    "snapshot_date": "2026-05-01",
                    "legalities": {"commander": "legal"},
                },
                {
                    "id": "s1",
                    "snapshot_date": "2026-05-02",
                    "legalities": {"commander": "banned"},
                },
            ]
        )
        with _make_gold_storage(tmp_path, {"silver_meta_history": meta}) as g:
            result = (
                g._signals.build_demand_signals()
                .sort_values("snapshot_date")
                .reset_index(drop=True)
            )
        # Row 0 (day 01): no previous → not banned on this day
        assert result.loc[0, "commander_banned"] == False  # noqa: E712
        # Row 1 (day 02): transitioned legal→banned
        assert result.loc[1, "commander_banned"] == True  # noqa: E712

    def test_detects_commander_unban_transition(self, tmp_path):
        meta = _make_meta_history(
            [
                {
                    "id": "s1",
                    "snapshot_date": "2026-05-01",
                    "legalities": {"commander": "banned"},
                },
                {
                    "id": "s1",
                    "snapshot_date": "2026-05-02",
                    "legalities": {"commander": "legal"},
                },
            ]
        )
        with _make_gold_storage(tmp_path, {"silver_meta_history": meta}) as g:
            result = (
                g._signals.build_demand_signals()
                .sort_values("snapshot_date")
                .reset_index(drop=True)
            )
        assert result.loc[1, "commander_unbanned"] == True  # noqa: E712

    def test_edhrec_rank_change_computed_correctly(self, tmp_path):
        meta = _make_meta_history(
            [
                {
                    "id": "s1",
                    "snapshot_date": "2026-05-01",
                    "legalities": {},
                    "edhrec_rank": 100,
                },
                {
                    "id": "s1",
                    "snapshot_date": "2026-05-02",
                    "legalities": {},
                    "edhrec_rank": 80,
                },
            ]
        )
        with _make_gold_storage(tmp_path, {"silver_meta_history": meta}) as g:
            result = (
                g._signals.build_demand_signals()
                .sort_values("snapshot_date")
                .reset_index(drop=True)
            )
        # Day 01: no prior rank → NULL
        assert pd.isna(result.loc[0, "edhrec_rank_change"])
        # Day 02: 80 - 100 = -20
        assert result.loc[1, "edhrec_rank_change"] == pytest.approx(-20)

    def test_returns_empty_when_meta_history_table_absent(self, tmp_path):
        with _make_gold_storage(tmp_path, {}) as g:  # no silver tables at all
            result = g._signals.build_demand_signals()
        assert result.empty


# ---------------------------------------------------------------------------
# Helpers for GoldSignalBuilders.build_tournament_signals tests
# ---------------------------------------------------------------------------


def _make_tournament_results(rows: list[dict]) -> pd.DataFrame:
    """Build a silver_tournament_results_history DataFrame."""
    return pd.DataFrame(
        [
            {
                "oracle_id": r.get("oracle_id"),
                "scryfall_id": r.get("scryfall_id", "s1"),
                "format": r.get("format", "modern"),
                "tournament_id": r.get("tournament_id", "t1"),
                "tournament_date": r.get("tournament_date", str(datetime.date.today())),
                "copies": r.get("copies", 4),
                "is_sideboard": r.get("is_sideboard", False),
            }
            for r in rows
        ]
    )


# ---------------------------------------------------------------------------
# GoldSignalBuilders.build_tournament_signals
# ---------------------------------------------------------------------------


class TestBuildTournamentSignals:
    def test_returns_expected_columns(self, tmp_path):
        df = _make_tournament_results([{"oracle_id": "o1"}])
        with _make_gold_storage(
            tmp_path, {"silver_tournament_results_history": df}
        ) as g:
            result = g._signals.build_tournament_signals()
        expected = {
            "oracle_id",
            "scryfall_id",
            "format",
            "top8_appearances_30d",
            "top8_appearances_90d",
            "top8_copies_avg",
            "sideboard_appearances_30d",
            "main_deck_pct",
            "last_top8_date",
        }
        assert expected.issubset(set(result.columns))

    def test_excludes_rows_with_null_oracle_id(self, tmp_path):
        df = _make_tournament_results(
            [
                {"oracle_id": "o1", "tournament_id": "t1"},
                {"oracle_id": None, "tournament_id": "t2"},  # must be excluded
            ]
        )
        with _make_gold_storage(
            tmp_path, {"silver_tournament_results_history": df}
        ) as g:
            result = g._signals.build_tournament_signals()
        assert len(result) == 1
        assert result.iloc[0]["oracle_id"] == "o1"

    def test_counts_recent_top8_appearances_within_30d(self, tmp_path):
        today = datetime.date.today()
        recent = str(today - datetime.timedelta(days=5))
        old = str(today - datetime.timedelta(days=60))
        df = _make_tournament_results(
            [
                {
                    "oracle_id": "o1",
                    "tournament_id": "t1",
                    "tournament_date": recent,
                    "is_sideboard": False,
                },
                {
                    "oracle_id": "o1",
                    "tournament_id": "t2",
                    "tournament_date": old,  # outside 30-day window
                    "is_sideboard": False,
                },
            ]
        )
        with _make_gold_storage(
            tmp_path, {"silver_tournament_results_history": df}
        ) as g:
            result = g._signals.build_tournament_signals()
        assert result.iloc[0]["top8_appearances_30d"] == 1
        assert result.iloc[0]["top8_appearances_90d"] == 2


# ---------------------------------------------------------------------------
# GoldFeatureBuilders.build_price_features — edhrec_rank JOIN branch
# ---------------------------------------------------------------------------


class TestBuildPriceFeaturesWithMeta:
    def test_edhrec_rank_populated_when_meta_history_present(self, tmp_path):
        """When silver_meta_history exists, edhrec_rank is joined by (scryfall_id, date)."""
        prices = _make_full_prices(
            [
                {
                    "uuid": "u1",
                    "scryfall_id": "s1",
                    "snapshot_date": "2026-05-01",
                    "eur": 5.0,
                }
            ]
        )
        meta = _make_meta_history(
            [
                {
                    "id": "s1",
                    "snapshot_date": "2026-05-01",
                    "legalities": {},
                    "edhrec_rank": 42,
                }
            ]
        )
        with _make_gold_storage(
            tmp_path,
            {"silver_prices_history": prices, "silver_meta_history": meta},
        ) as g:
            result = g._features.build_price_features()
        assert result.iloc[0]["edhrec_rank"] == 42

    def test_edhrec_rank_null_when_meta_history_absent(self, tmp_path):
        """When silver_meta_history is absent, edhrec_rank column is present but all NULL."""
        prices = _make_full_prices(
            [
                {
                    "uuid": "u1",
                    "scryfall_id": "s1",
                    "snapshot_date": "2026-05-01",
                    "eur": 5.0,
                }
            ]
        )
        with _make_gold_storage(tmp_path, {"silver_prices_history": prices}) as g:
            result = g._features.build_price_features()
        assert "edhrec_rank" in result.columns
        assert result["edhrec_rank"].isna().all()


# ---------------------------------------------------------------------------
# GoldFeatureBuilders.build_language_premiums
# ---------------------------------------------------------------------------


def _make_lang_prices(rows: list[dict]) -> pd.DataFrame:
    """Build a silver_language_prices_history DataFrame."""
    return pd.DataFrame(
        [
            {
                "scryfall_id": r["scryfall_id"],
                "canonical_uuid": r["canonical_uuid"],
                "lang": r.get("lang", "Japanese"),
                "snapshot_date": r["snapshot_date"],
                "eur": r.get("eur", None),
                "eur_foil": r.get("eur_foil", None),
                "usd": r.get("usd", None),
                "usd_foil": r.get("usd_foil", None),
            }
            for r in rows
        ]
    )


class TestBuildLanguagePremiums:
    def test_returns_empty_when_silver_language_prices_absent(self, tmp_path):
        prices = _make_full_prices(
            [
                {
                    "uuid": "u1",
                    "scryfall_id": "s1",
                    "snapshot_date": "2026-05-01",
                    "eur": 5.0,
                }
            ]
        )
        with _make_gold_storage(tmp_path, {"silver_prices_history": prices}) as g:
            result = g._features.build_language_premiums()
        assert result.empty

    def test_returns_empty_when_silver_prices_absent(self, tmp_path):
        lang = _make_lang_prices(
            [
                {
                    "scryfall_id": "s1-ja",
                    "canonical_uuid": "u1",
                    "snapshot_date": "2026-05-01",
                    "eur": 20.0,
                }
            ]
        )
        with _make_gold_storage(
            tmp_path, {"silver_language_prices_history": lang}
        ) as g:
            result = g._features.build_language_premiums()
        assert result.empty

    def test_premium_computed_correctly(self, tmp_path):
        prices = _make_full_prices(
            [
                {
                    "uuid": "u1",
                    "scryfall_id": "s1-en",
                    "snapshot_date": "2026-05-01",
                    "eur": 5.0,
                }
            ]
        )
        lang = _make_lang_prices(
            [
                {
                    "scryfall_id": "s1-ja",
                    "canonical_uuid": "u1",
                    "snapshot_date": "2026-05-01",
                    "eur": 20.0,
                }
            ]
        )
        with _make_gold_storage(
            tmp_path,
            {"silver_prices_history": prices, "silver_language_prices_history": lang},
        ) as g:
            result = g._features.build_language_premiums()

        assert len(result) == 1
        row = result.iloc[0]
        assert row["scryfall_id"] == "s1-ja"
        assert row["canonical_uuid"] == "u1"
        assert row["eur_lang_premium"] == pytest.approx(4.0)  # 20 / 5

    def test_premium_null_when_canonical_eur_is_null(self, tmp_path):
        prices = _make_full_prices(
            [
                {
                    "uuid": "u1",
                    "scryfall_id": "s1-en",
                    "snapshot_date": "2026-05-01",
                    "eur": None,
                }
            ]
        )
        lang = _make_lang_prices(
            [
                {
                    "scryfall_id": "s1-ja",
                    "canonical_uuid": "u1",
                    "snapshot_date": "2026-05-01",
                    "eur": 20.0,
                }
            ]
        )
        with _make_gold_storage(
            tmp_path,
            {"silver_prices_history": prices, "silver_language_prices_history": lang},
        ) as g:
            result = g._features.build_language_premiums()

        assert pd.isna(result.iloc[0]["eur_lang_premium"])

    def test_returns_expected_columns(self, tmp_path):
        prices = _make_full_prices(
            [
                {
                    "uuid": "u1",
                    "scryfall_id": "s1-en",
                    "snapshot_date": "2026-05-01",
                    "eur": 5.0,
                }
            ]
        )
        lang = _make_lang_prices(
            [
                {
                    "scryfall_id": "s1-ja",
                    "canonical_uuid": "u1",
                    "snapshot_date": "2026-05-01",
                    "eur": 10.0,
                }
            ]
        )
        with _make_gold_storage(
            tmp_path,
            {"silver_prices_history": prices, "silver_language_prices_history": lang},
        ) as g:
            result = g._features.build_language_premiums()

        expected = {
            "scryfall_id",
            "canonical_uuid",
            "lang",
            "snapshot_date",
            "lang_eur",
            "lang_eur_foil",
            "lang_usd",
            "lang_usd_foil",
            "canonical_eur",
            "canonical_eur_foil",
            "eur_lang_premium",
            "eur_foil_lang_premium",
        }
        assert set(result.columns) == expected

    def test_no_row_for_variant_without_matching_snapshot(self, tmp_path):
        # Canonical has price on day 01; variant price is on day 02 — no match.
        prices = _make_full_prices(
            [
                {
                    "uuid": "u1",
                    "scryfall_id": "s1-en",
                    "snapshot_date": "2026-05-01",
                    "eur": 5.0,
                }
            ]
        )
        lang = _make_lang_prices(
            [
                {
                    "scryfall_id": "s1-ja",
                    "canonical_uuid": "u1",
                    "snapshot_date": "2026-05-02",
                    "eur": 10.0,
                }
            ]
        )
        with _make_gold_storage(
            tmp_path,
            {"silver_prices_history": prices, "silver_language_prices_history": lang},
        ) as g:
            result = g._features.build_language_premiums()

        assert result.empty

    def test_pipeline_creates_gold_language_premiums(self, tmp_path):
        prices = _make_full_prices(
            [
                {
                    "uuid": "u1",
                    "scryfall_id": "s1-en",
                    "snapshot_date": "2026-05-01",
                    "eur": 5.0,
                }
            ]
        )
        lang = _make_lang_prices(
            [
                {
                    "scryfall_id": "s1-ja",
                    "canonical_uuid": "u1",
                    "snapshot_date": "2026-05-01",
                    "eur": 10.0,
                }
            ]
        )
        with _make_gold_storage(
            tmp_path,
            {"silver_prices_history": prices, "silver_language_prices_history": lang},
        ) as g:
            g.populate()
            tables = {r[0] for r in g._gold_con.execute("SHOW TABLES").fetchall()}
        assert "gold_language_premiums" in tables

    def test_pipeline_skips_gold_language_premiums_when_silver_language_absent(
        self, tmp_path
    ):
        prices = _make_full_prices(
            [
                {
                    "uuid": "u1",
                    "scryfall_id": "s1",
                    "snapshot_date": "2026-05-01",
                    "eur": 5.0,
                }
            ]
        )
        with _make_gold_storage(tmp_path, {"silver_prices_history": prices}) as g:
            g.populate()
            tables = {r[0] for r in g._gold_con.execute("SHOW TABLES").fetchall()}
        assert "gold_language_premiums" not in tables


# ---------------------------------------------------------------------------
# GoldStorage._pipeline — gold_ml_dataset horizon guard
# ---------------------------------------------------------------------------


class TestPipelineMLDatasetGuard:
    def test_ml_dataset_skipped_when_date_range_below_threshold(self, tmp_path, caplog):
        # Single snapshot → date range = 0 days → no t+7 targets → skip build.
        prices = _make_full_prices(
            [
                {
                    "uuid": "u1",
                    "scryfall_id": "s1",
                    "snapshot_date": "2026-05-01",
                    "eur": 1.0,
                }
            ]
        )
        with _make_gold_storage(tmp_path, {"silver_prices_history": prices}) as g:
            with caplog.at_level(
                logging.WARNING, logger="src.data.cards.storage.gold.storage"
            ):
                g.populate()
            tables = {r[0] for r in g._gold_con.execute("SHOW TABLES").fetchall()}

        assert "gold_ml_dataset" not in tables
        assert "skipping gold_ml_dataset" in caplog.text

    def test_ml_dataset_built_when_date_range_meets_threshold(self, tmp_path):
        # Two snapshots 7 days apart → date range = 7 → t+7 targets exist → build.
        prices = _make_full_prices(
            [
                {
                    "uuid": "u1",
                    "scryfall_id": "s1",
                    "snapshot_date": "2026-05-01",
                    "eur": 1.0,
                },
                {
                    "uuid": "u1",
                    "scryfall_id": "s1",
                    "snapshot_date": "2026-05-08",
                    "eur": 1.5,
                },
            ]
        )
        with _make_gold_storage(tmp_path, {"silver_prices_history": prices}) as g:
            g.populate()
            tables = {r[0] for r in g._gold_con.execute("SHOW TABLES").fetchall()}

        assert "gold_ml_dataset" in tables

    def test_ml_dataset_skipped_when_gold_price_features_absent(self, tmp_path):
        # No silver_prices_history → gold_price_features not built → guard skips.
        with _make_gold_storage(tmp_path, {}) as g:
            g.populate()
            tables = {r[0] for r in g._gold_con.execute("SHOW TABLES").fetchall()}

        assert "gold_ml_dataset" not in tables


# ---------------------------------------------------------------------------
# Helpers for GoldMLDatasetBuilder tests
# ---------------------------------------------------------------------------


def _build_ml(gold_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Run build_ml_dataset against an in-memory Gold DB pre-loaded with gold_tables."""
    con = duckdb.connect(":memory:")
    for name, df in gold_tables.items():
        con.register("_df", df)
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM _df")
        con.unregister("_df")
    result = GoldMLDatasetBuilder(con).build_ml_dataset()
    con.close()
    return result


def _gpf(rows: list[dict]) -> pd.DataFrame:
    """Minimal gold_price_features rows."""
    return pd.DataFrame(
        [
            {
                "uuid": r["uuid"],
                "scryfall_id": r.get("scryfall_id", f"s_{r['uuid']}"),
                "snapshot_date": r["snapshot_date"],
                "eur": r.get("eur", 1.0),
            }
            for r in rows
        ]
    )


def _gcf(rows: list[dict]) -> pd.DataFrame:
    """Minimal gold_card_features rows.

    Includes all columns referenced in build_ml_dataset's card_cols SELECT.
    """
    return pd.DataFrame(
        [
            {
                "uuid": r["uuid"],
                "scryfall_id": r.get("scryfall_id", f"s_{r['uuid']}"),
                "oracle_id": r.get("oracle_id", f"o_{r['uuid']}"),
                "name": r["name"],
                "rarity": r.get("rarity", "common"),
                "mana_value": r.get("mana_value", 2),
                "is_reserved": r.get("is_reserved", False),
                "is_reprint": r.get("is_reprint", False),
                "color_count": r.get("color_count", 1),
                "color_identity_count": r.get("color_identity_count", 1),
                "is_commander_legal": r.get("is_commander_legal", True),
                "is_modern_legal": r.get("is_modern_legal", False),
                "is_legacy_legal": r.get("is_legacy_legal", False),
                "is_standard_legal": r.get("is_standard_legal", False),
                "format_count": r.get("format_count", 1),
                "print_count": r.get("print_count", 1),
                "finish_count": r.get("finish_count", 1),
                "has_etched_finish": r.get("has_etched_finish", False),
                "edhrec_saltiness": r.get("edhrec_saltiness", None),
                "set_type": r.get("set_type", "expansion"),
            }
            for r in rows
        ]
    )


def _gfs(rows: list[dict]) -> pd.DataFrame:
    """Minimal gold_format_staples rows."""
    return pd.DataFrame(
        [
            {
                "id": f"{r['card_name']}__{r['format']}",
                "card_name": r["card_name"],
                "format": r["format"],
                "snapshot_date": r["snapshot_date"],
                "deck_pct": r.get("deck_pct", 10.0),
                "deck_pct_7d_avg": r.get("deck_pct_7d_avg", 10.0),
            }
            for r in rows
        ]
    )


# ---------------------------------------------------------------------------
# GoldMLDatasetBuilder.build_ml_dataset
# ---------------------------------------------------------------------------


class TestBuildMLDataset:
    def test_returns_empty_when_price_features_absent(self):
        result = _build_ml({})
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_returns_rows_for_each_price_feature(self):
        pf = _gpf(
            [
                {"uuid": "u1", "snapshot_date": "2026-05-01", "eur": 2.0},
                {"uuid": "u2", "snapshot_date": "2026-05-01", "eur": 3.0},
            ]
        )
        result = _build_ml({"gold_price_features": pf})
        assert len(result) == 2

    def test_excludes_rows_with_null_eur(self):
        pf = _gpf(
            [
                {"uuid": "u1", "snapshot_date": "2026-05-01", "eur": 2.0},
                {"uuid": "u2", "snapshot_date": "2026-05-01", "eur": None},
            ]
        )
        result = _build_ml({"gold_price_features": pf})
        assert len(result) == 1
        assert result.iloc[0]["uuid"] == "u1"

    def test_target_7d_joined_when_future_price_exists(self):
        pf = _gpf(
            [
                {"uuid": "u1", "snapshot_date": "2026-05-01", "eur": 2.0},
                {"uuid": "u1", "snapshot_date": "2026-05-08", "eur": 4.0},
            ]
        )
        result = (
            _build_ml({"gold_price_features": pf})
            .sort_values("snapshot_date")
            .reset_index(drop=True)
        )
        # Day 01 → day 08 is exactly 7 days later
        assert result.loc[0, "target_price_7d"] == pytest.approx(4.0)
        # Day 08 has no day 15 row → target_price_7d is NULL
        assert pd.isna(result.loc[1, "target_price_7d"])

    def test_target_7d_null_when_no_future_snapshot(self):
        pf = _gpf([{"uuid": "u1", "snapshot_date": "2026-05-01", "eur": 5.0}])
        result = _build_ml({"gold_price_features": pf})
        assert pd.isna(result.iloc[0]["target_price_7d"])

    def test_target_change_30d_up_when_price_rises_over_20pct(self):
        pf = _gpf(
            [
                {"uuid": "u1", "snapshot_date": "2026-05-01", "eur": 1.0},
                {"uuid": "u1", "snapshot_date": "2026-05-31", "eur": 1.3},
            ]
        )
        result = (
            _build_ml({"gold_price_features": pf})
            .sort_values("snapshot_date")
            .reset_index(drop=True)
        )
        assert result.loc[0, "target_change_30d"] == "up"

    def test_target_change_30d_down_when_price_falls_over_20pct(self):
        pf = _gpf(
            [
                {"uuid": "u1", "snapshot_date": "2026-05-01", "eur": 1.0},
                {"uuid": "u1", "snapshot_date": "2026-05-31", "eur": 0.7},
            ]
        )
        result = (
            _build_ml({"gold_price_features": pf})
            .sort_values("snapshot_date")
            .reset_index(drop=True)
        )
        assert result.loc[0, "target_change_30d"] == "down"

    def test_target_change_30d_flat_for_small_move(self):
        pf = _gpf(
            [
                {"uuid": "u1", "snapshot_date": "2026-05-01", "eur": 1.0},
                {"uuid": "u1", "snapshot_date": "2026-05-31", "eur": 1.1},
            ]
        )
        result = (
            _build_ml({"gold_price_features": pf})
            .sort_values("snapshot_date")
            .reset_index(drop=True)
        )
        assert result.loc[0, "target_change_30d"] == "flat"

    def test_target_change_30d_null_when_no_30d_price(self):
        pf = _gpf([{"uuid": "u1", "snapshot_date": "2026-05-01", "eur": 1.0}])
        result = _build_ml({"gold_price_features": pf})
        assert pd.isna(result.iloc[0]["target_change_30d"])

    def test_staple_pct_vintage_populated_from_format_staples(self):
        pf = _gpf([{"uuid": "u1", "snapshot_date": "2026-05-01", "eur": 5.0}])
        cf = _gcf([{"uuid": "u1", "name": "Black Lotus"}])
        gfs = _gfs(
            [
                {
                    "card_name": "Black Lotus",
                    "format": "vintage",
                    "snapshot_date": "2026-05-01",
                    "deck_pct": 42.0,
                    "deck_pct_7d_avg": 41.0,
                }
            ]
        )
        result = _build_ml(
            {
                "gold_price_features": pf,
                "gold_card_features": cf,
                "gold_format_staples": gfs,
            }
        )
        assert result.iloc[0]["staple_pct_vintage"] == pytest.approx(42.0)
        assert result.iloc[0]["staple_7d_vintage"] == pytest.approx(41.0)

    def test_staple_pct_commander_populated_from_format_staples(self):
        pf = _gpf([{"uuid": "u1", "snapshot_date": "2026-05-01", "eur": 1.0}])
        cf = _gcf([{"uuid": "u1", "name": "Sol Ring"}])
        gfs = _gfs(
            [
                {
                    "card_name": "Sol Ring",
                    "format": "commander",
                    "snapshot_date": "2026-05-01",
                    "deck_pct": 78.0,
                }
            ]
        )
        result = _build_ml(
            {
                "gold_price_features": pf,
                "gold_card_features": cf,
                "gold_format_staples": gfs,
            }
        )
        assert result.iloc[0]["staple_pct_commander"] == pytest.approx(78.0)

    def test_staple_columns_null_when_format_staples_absent(self):
        pf = _gpf([{"uuid": "u1", "snapshot_date": "2026-05-01", "eur": 1.0}])
        result = _build_ml({"gold_price_features": pf})
        for col in (
            "staple_pct_commander",
            "staple_7d_commander",
            "staple_pct_modern",
            "staple_pct_legacy",
            "staple_pct_vintage",
            "staple_7d_vintage",
        ):
            assert col in result.columns, f"Column {col} missing"
            assert pd.isna(result.iloc[0][col]), f"{col} should be NULL"

    def test_staple_pct_vintage_null_for_non_vintage_card(self):
        pf = _gpf([{"uuid": "u1", "snapshot_date": "2026-05-01", "eur": 1.0}])
        cf = _gcf([{"uuid": "u1", "name": "Sol Ring"}])
        # Only commander data — no vintage row for Sol Ring
        gfs = _gfs(
            [
                {
                    "card_name": "Sol Ring",
                    "format": "commander",
                    "snapshot_date": "2026-05-01",
                    "deck_pct": 78.0,
                }
            ]
        )
        result = _build_ml(
            {
                "gold_price_features": pf,
                "gold_card_features": cf,
                "gold_format_staples": gfs,
            }
        )
        assert pd.isna(result.iloc[0]["staple_pct_vintage"])

    def test_staple_join_is_date_exact(self):
        """Staple from a different snapshot_date must not bleed into earlier rows."""
        pf = _gpf(
            [
                {"uuid": "u1", "snapshot_date": "2026-05-01", "eur": 1.0},
                {"uuid": "u1", "snapshot_date": "2026-05-02", "eur": 1.0},
            ]
        )
        cf = _gcf([{"uuid": "u1", "name": "Black Lotus"}])
        # Only vintage data for day 02 — day 01 row must get NULL
        gfs = _gfs(
            [
                {
                    "card_name": "Black Lotus",
                    "format": "vintage",
                    "snapshot_date": "2026-05-02",
                    "deck_pct": 55.0,
                }
            ]
        )
        result = (
            _build_ml(
                {
                    "gold_price_features": pf,
                    "gold_card_features": cf,
                    "gold_format_staples": gfs,
                }
            )
            .sort_values("snapshot_date")
            .reset_index(drop=True)
        )
        assert pd.isna(result.loc[0, "staple_pct_vintage"])  # day 01 — no staple
        assert result.loc[1, "staple_pct_vintage"] == pytest.approx(55.0)  # day 02

    def test_static_columns_null_when_card_features_absent(self):
        pf = _gpf([{"uuid": "u1", "snapshot_date": "2026-05-01", "eur": 1.0}])
        result = _build_ml({"gold_price_features": pf})
        for col in ("rarity", "mana_value", "is_reserved", "set_type"):
            assert col in result.columns, f"Column {col} missing"
            assert pd.isna(result.iloc[0][col]), f"{col} should be NULL"
