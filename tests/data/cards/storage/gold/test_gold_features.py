import math

import duckdb
import pytest

from src.data.cards.storage.gold.features import GoldFeatureBuilders


# ── helpers ───────────────────────────────────────────────────────────────────


def _card_con(rows: list[dict]) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")
    con.execute("""
        CREATE TABLE silver_cards (
            uuid VARCHAR, scryfall_id VARCHAR, oracle_id VARCHAR, name VARCHAR,
            set_code VARCHAR, rarity VARCHAR, mana_value DOUBLE,
            is_reserved BOOLEAN, is_reprint BOOLEAN, is_promo BOOLEAN,
            is_full_art BOOLEAN, is_textless BOOLEAN,
            edhrec_saltiness DOUBLE, set_type VARCHAR,
            finishes VARCHAR[], colors VARCHAR[],
            color_identity VARCHAR[], variations VARCHAR[],
            original_supertypes VARCHAR[],
            is_commander_legal BOOLEAN, is_standard_legal BOOLEAN,
            is_modern_legal BOOLEAN, is_legacy_legal BOOLEAN,
            format_count INTEGER
        )
    """)
    for r in rows:
        con.execute(
            """INSERT INTO silver_cards VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )""",
            [
                r.get("uuid"),
                r.get("scryfall_id", "sf"),
                r.get("oracle_id", "o1"),
                r.get("name", "Card"),
                r.get("set_code", "M21"),
                r.get("rarity", "common"),
                r.get("mana_value"),
                r.get("is_reserved", False),
                r.get("is_reprint", False),
                r.get("is_promo", False),
                r.get("is_full_art", False),
                r.get("is_textless", False),
                r.get("edhrec_saltiness"),
                r.get("set_type", "core"),
                r.get("finishes", ["nonfoil"]),
                r.get("colors", []),
                r.get("color_identity", []),
                r.get("variations", []),
                r.get("original_supertypes", []),
                r.get("is_commander_legal", False),
                r.get("is_standard_legal", False),
                r.get("is_modern_legal", False),
                r.get("is_legacy_legal", False),
                r.get("format_count", 0),
            ],
        )
    return con


def _price_con(rows: list[dict]) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")
    con.execute("""
        CREATE TABLE silver_prices_history (
            uuid VARCHAR, scryfall_id VARCHAR, snapshot_date DATE,
            eur DOUBLE, eur_foil DOUBLE, usd DOUBLE, usd_foil DOUBLE,
            cardmarket_eur DOUBLE, cardmarket_eur_foil DOUBLE,
            tcgplayer_usd DOUBLE, tcgplayer_usd_foil DOUBLE
        )
    """)
    for r in rows:
        con.execute(
            "INSERT INTO silver_prices_history VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                r["uuid"],
                r.get("scryfall_id", "sf"),
                r["snapshot_date"],
                r.get("eur"),
                r.get("eur_foil"),
                r.get("usd"),
                r.get("usd_foil"),
                r.get("cardmarket_eur"),
                r.get("cardmarket_eur_foil"),
                r.get("tcgplayer_usd"),
                r.get("tcgplayer_usd_foil"),
            ],
        )
    return con


# ── build_card_features ───────────────────────────────────────────────────────


class TestBuildCardFeatures:
    def test_excludes_null_uuid_rows(self) -> None:
        con = _card_con([
            {"uuid": None, "scryfall_id": "sf1", "name": "Scryfall Only"},
            {"uuid": "u1", "scryfall_id": "sf2", "name": "Real Card"},
        ])
        result = GoldFeatureBuilders(con).build_card_features()
        assert list(result["uuid"]) == ["u1"]

    def test_mana_value_above_20_becomes_null(self) -> None:
        con = _card_con([{"uuid": "u1", "mana_value": 1_000_000}])
        result = GoldFeatureBuilders(con).build_card_features()
        assert math.isnan(result.iloc[0]["mana_value"])

    def test_mana_value_at_20_is_kept(self) -> None:
        con = _card_con([{"uuid": "u1", "mana_value": 20.0}])
        result = GoldFeatureBuilders(con).build_card_features()
        assert result.iloc[0]["mana_value"] == 20.0

    def test_is_legendary_from_supertypes(self) -> None:
        con = _card_con([
            {"uuid": "u1", "original_supertypes": ["Legendary", "Basic"]},
            {"uuid": "u2", "original_supertypes": ["Basic"]},
            {"uuid": "u3", "original_supertypes": []},
        ])
        result = GoldFeatureBuilders(con).build_card_features().set_index("uuid")
        assert result.loc["u1", "is_legendary"]
        assert not result.loc["u2", "is_legendary"]
        assert not result.loc["u3", "is_legendary"]

    def test_print_count_per_oracle_id(self) -> None:
        con = _card_con([
            {"uuid": "u1", "oracle_id": "oracle_a"},
            {"uuid": "u2", "oracle_id": "oracle_a"},
            {"uuid": "u3", "oracle_id": "oracle_b"},
        ])
        result = GoldFeatureBuilders(con).build_card_features().set_index("uuid")
        assert result.loc["u1", "print_count"] == 2
        assert result.loc["u2", "print_count"] == 2
        assert result.loc["u3", "print_count"] == 1

    def test_finish_count_and_has_etched(self) -> None:
        con = _card_con([
            {"uuid": "u1", "finishes": ["nonfoil", "etched"]},
            {"uuid": "u2", "finishes": ["nonfoil"]},
        ])
        result = GoldFeatureBuilders(con).build_card_features().set_index("uuid")
        assert result.loc["u1", "finish_count"] == 2
        assert result.loc["u1", "has_etched_finish"]
        assert not result.loc["u2", "has_etched_finish"]


# ── build_language_premiums ───────────────────────────────────────────────────


class TestBuildLanguagePremiums:
    def test_returns_empty_when_tables_absent(self) -> None:
        con = duckdb.connect(":memory:")
        result = GoldFeatureBuilders(con).build_language_premiums()
        assert result.empty


# ── build_price_features ──────────────────────────────────────────────────────


class TestBuildPriceFeatures:
    def test_lag_1d_is_previous_row_price(self) -> None:
        con = _price_con([
            {"uuid": "u1", "snapshot_date": "2026-01-01", "eur": 1.0},
            {"uuid": "u1", "snapshot_date": "2026-01-02", "eur": 2.0},
            {"uuid": "u1", "snapshot_date": "2026-01-03", "eur": 3.0},
        ])
        result = GoldFeatureBuilders(con).build_price_features()
        rows = result[result["uuid"] == "u1"].sort_values("snapshot_date").reset_index(drop=True)
        assert math.isnan(rows.loc[0, "price_change_1d_abs"])
        assert rows.loc[1, "price_change_1d_abs"] == pytest.approx(1.0)
        assert rows.loc[2, "price_change_1d_abs"] == pytest.approx(1.0)

    def test_is_price_spike_flags_over_300_pct_change(self) -> None:
        con = _price_con([
            {"uuid": "u1", "snapshot_date": "2026-01-01", "eur": 1.0},
            {"uuid": "u1", "snapshot_date": "2026-01-02", "eur": 5.0},  # +400%
            {"uuid": "u1", "snapshot_date": "2026-01-03", "eur": 5.5},  # +10%
        ])
        result = GoldFeatureBuilders(con).build_price_features()
        rows = result[result["uuid"] == "u1"].sort_values("snapshot_date").reset_index(drop=True)
        assert rows.loc[1, "is_price_spike"]
        assert not rows.loc[2, "is_price_spike"]

    def test_price_ath_does_not_use_future_rows(self) -> None:
        # w_hist is bounded to ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW.
        # On 2026-01-02 the ATH must be 2.0, not 5.0 (which is a future row).
        con = _price_con([
            {"uuid": "u1", "snapshot_date": "2026-01-01", "eur": 2.0},
            {"uuid": "u1", "snapshot_date": "2026-01-02", "eur": 1.0},
            {"uuid": "u1", "snapshot_date": "2026-01-03", "eur": 5.0},
        ])
        result = GoldFeatureBuilders(con).build_price_features()
        rows = result[result["uuid"] == "u1"].sort_values("snapshot_date").reset_index(drop=True)
        assert rows.loc[1, "price_ath"] == pytest.approx(2.0)

    def test_null_edhrec_rank_when_meta_history_absent(self) -> None:
        con = _price_con([
            {"uuid": "u1", "snapshot_date": "2026-01-01", "eur": 1.0},
        ])
        result = GoldFeatureBuilders(con).build_price_features()
        import pandas as pd

        assert "edhrec_rank" in result.columns
        assert pd.isna(result.iloc[0]["edhrec_rank"])
