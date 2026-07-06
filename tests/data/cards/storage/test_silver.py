import json
import logging
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from src.data.cards.storage.silver import SilverStorage
from src.data.cards.storage.base.writers import DuckDBWriter as SilverWriter
from src.data.cards.storage.errors import StorageWriteError

MINIMAL_CONFIG = {
    "language_map": {"en": "English", "es": "Spanish"},
    "legality_map": {"Legal": "legal", "Not Legal": "not_legal"},
    "supertypes": ["Legendary", "Basic"],
    "card_types": ["Creature", "Instant"],
    "sources": {},
}


def _make_storage(tmp_path):
    """Return a SilverStorage backed by a temp bronze file and in-memory silver."""
    config_path = tmp_path / "silver_config.json"
    config_path.write_text(json.dumps(MINIMAL_CONFIG))
    bronze_path = str(tmp_path / "bronze.duckdb")
    # DuckDB requires the file to exist before opening it read-only.
    duckdb.connect(bronze_path).close()
    return SilverStorage(bronze_path, ":memory:", str(config_path))


@pytest.fixture
def storage(tmp_path):
    s = _make_storage(tmp_path)
    yield s
    s.close()


# ---------------------------------------------------------------------------
# populate / update (integration smoke tests)
# ---------------------------------------------------------------------------


def _make_storage_with_meta_bronze(
    tmp_path: Path,
    rows: list[tuple[str, str]],
) -> SilverStorage:
    """SilverStorage with Bronze `bronze_scryfall_meta_history` populated.

    Creates the full production Bronze schema; callers supply only
    (id, snapshot_date) — remaining columns get sensible defaults.
    """
    config = {**MINIMAL_CONFIG, "sources": {}}
    config_path = tmp_path / "silver_config.json"
    config_path.write_text(json.dumps(config))

    bronze_path = str(tmp_path / "bronze.duckdb")
    con = duckdb.connect(bronze_path)
    con.execute("""
        CREATE TABLE bronze_scryfall_meta_history (
            id            VARCHAR,
            snapshot_date VARCHAR,
            legalities    VARCHAR,
            edhrec_rank   DOUBLE,
            reserved      BOOLEAN,
            promo_types   VARCHAR,
            finishes      VARCHAR
        )
    """)
    for id_, snap in rows:
        con.execute(
            "INSERT INTO bronze_scryfall_meta_history"
            " VALUES (?, ?, NULL, NULL, false, '[]', '[]')",
            [id_, snap],
        )
    con.close()
    return SilverStorage(bronze_path, ":memory:", str(config_path))


def _make_minimal_mtgjson_bronze(con: duckdb.DuckDBPyConnection) -> None:
    """Create bronze_mtgjson_cards with the columns needed by _build_silver_cards_sql."""
    con.execute("""
        CREATE TABLE bronze_mtgjson_cards (
            uuid           VARCHAR,
            name           VARCHAR,
            ascii_name     VARCHAR,
            set_code       VARCHAR,
            number         VARCHAR,
            language       VARCHAR,
            layout         VARCHAR,
            mana_cost      VARCHAR,
            mana_value     DOUBLE,
            face_mana_value DOUBLE,
            text           VARCHAR,
            original_text  VARCHAR,
            original_type  VARCHAR,
            power          VARCHAR,
            toughness      VARCHAR,
            loyalty        VARCHAR,
            defense        VARCHAR,
            rarity         VARCHAR,
            border_color   VARCHAR,
            frame_version  VARCHAR,
            watermark      VARCHAR,
            security_stamp VARCHAR,
            flavor_text    VARCHAR,
            flavor_name    VARCHAR,
            artist         VARCHAR,
            printed_name   VARCHAR,
            printed_text   VARCHAR,
            face_name      VARCHAR,
            side           VARCHAR,
            edhrec_rank    DOUBLE,
            edhrec_saltiness DOUBLE,
            is_online_only BOOLEAN,
            is_funny       BOOLEAN,
            is_oversized   BOOLEAN,
            is_reprint     BOOLEAN,
            is_reserved    BOOLEAN,
            is_promo       BOOLEAN,
            is_full_art    BOOLEAN,
            is_textless    BOOLEAN,
            is_alternative BOOLEAN,
            is_story_spotlight BOOLEAN,
            is_timeshifted BOOLEAN,
            is_rebalanced  BOOLEAN,
            is_game_changer BOOLEAN,
            has_alternative_deck_limit BOOLEAN,
            has_content_warning BOOLEAN,
            legalities     VARCHAR,
            identifiers    VARCHAR,
            colors         VARCHAR,
            color_identity VARCHAR,
            color_indicator VARCHAR,
            produced_mana  VARCHAR,
            printings      VARCHAR,
            keywords       VARCHAR,
            finishes       VARCHAR,
            availability   VARCHAR,
            frame_effects  VARCHAR,
            booster_types  VARCHAR,
            promo_types    VARCHAR,
            rulings        VARCHAR,
            artist_ids     VARCHAR,
            other_face_ids VARCHAR,
            card_parts     VARCHAR,
            variations     VARCHAR
        )
    """)


def _make_minimal_scryfall_bronze(con: duckdb.DuckDBPyConnection) -> None:
    """Create bronze_scryfall_cards with the columns needed by _build_silver_cards_sql."""
    con.execute("""
        CREATE TABLE bronze_scryfall_cards (
            id             VARCHAR,
            oracle_id      VARCHAR,
            name           VARCHAR,
            lang           VARCHAR,
            layout         VARCHAR,
            mana_cost      VARCHAR,
            cmc            DOUBLE,
            oracle_text    VARCHAR,
            type_line      VARCHAR,
            power          VARCHAR,
            toughness      VARCHAR,
            loyalty        VARCHAR,
            defense        VARCHAR,
            artist         VARCHAR,
            illustration_id VARCHAR,
            border_color   VARCHAR,
            collector_number VARCHAR,
            flavor_name    VARCHAR,
            flavor_text    VARCHAR,
            frame          VARCHAR,
            printed_name   VARCHAR,
            printed_text   VARCHAR,
            printed_type_line VARCHAR,
            rarity         VARCHAR,
            set            VARCHAR,
            set_id         VARCHAR,
            set_name       VARCHAR,
            set_type       VARCHAR,
            security_stamp VARCHAR,
            watermark      VARCHAR,
            scryfall_uri   VARCHAR,
            tcgplayer_id   DOUBLE,
            cardmarket_id  DOUBLE,
            edhrec_rank    DOUBLE,
            penny_rank     DOUBLE,
            digital        BOOLEAN,
            oversized      BOOLEAN,
            reserved       BOOLEAN,
            reprint        BOOLEAN,
            promo          BOOLEAN,
            full_art       BOOLEAN,
            textless       BOOLEAN,
            variation      BOOLEAN,
            booster        BOOLEAN,
            story_spotlight BOOLEAN,
            game_changer   BOOLEAN,
            legalities     VARCHAR,
            color_identity VARCHAR,
            color_indicator VARCHAR,
            colors         VARCHAR,
            produced_mana  VARCHAR,
            keywords       VARCHAR,
            finishes       VARCHAR,
            frame_effects  VARCHAR,
            games          VARCHAR,
            promo_types    VARCHAR,
            artist_ids     VARCHAR,
            multiverse_ids VARCHAR,
            all_parts      VARCHAR,
            card_faces     VARCHAR
        )
    """)


def _make_storage_with_cards_bronze(
    tmp_path: Path,
    mtgjson_rows: list[dict],
    scryfall_rows: list[dict],
) -> SilverStorage:
    """SilverStorage with Bronze card tables populated for SQL path tests."""
    config = {**MINIMAL_CONFIG, "sources": {}}
    config_path = tmp_path / "silver_config.json"
    config_path.write_text(json.dumps(config))

    bronze_path = str(tmp_path / "bronze.duckdb")
    con = duckdb.connect(bronze_path)
    _make_minimal_mtgjson_bronze(con)
    _make_minimal_scryfall_bronze(con)

    for row in mtgjson_rows:
        cols = ", ".join(row.keys())
        placeholders = ", ".join(["?"] * len(row))
        con.execute(
            f"INSERT INTO bronze_mtgjson_cards ({cols}) VALUES ({placeholders})",
            list(row.values()),
        )
    for row in scryfall_rows:
        cols = ", ".join(row.keys())
        placeholders = ", ".join(["?"] * len(row))
        con.execute(
            f"INSERT INTO bronze_scryfall_cards ({cols}) VALUES ({placeholders})",
            list(row.values()),
        )
    con.close()
    return SilverStorage(bronze_path, ":memory:", str(config_path))


# Minimal dicts for test rows — only required columns; rest default to NULL/false
_MTGJSON_ROW = {
    "uuid": "uuid-a",
    "name": "Lightning Bolt",
    "set_code": "M10",
    "number": "141",
    "language": "English",
    "layout": "normal",
    "rarity": "common",
    "is_online_only": False,
    "is_funny": False,
    "is_oversized": False,
    "identifiers": '{"scryfall_id": "scryfall-a"}',
    "legalities": '{"commander": "legal", "standard": "not_legal"}',
    "colors": '["R"]',
    "color_identity": '["R"]',
    "finishes": '["nonfoil"]',
}

_SCRYFALL_ROW = {
    "id": "scryfall-a",
    "name": "Lightning Bolt",
    "lang": "en",
    "layout": "normal",
    "rarity": "common",
    "set": "m10",
    "digital": False,
    "oversized": False,
    "legalities": '{"commander": "legal", "standard": "not_legal"}',
    "colors": '["R"]',
    "color_identity": '["R"]',
    "finishes": '["nonfoil"]',
}


class TestPopulateUpdate:
    def test_populate_with_empty_sources_does_not_raise(self, tmp_path):
        with _make_storage(tmp_path) as s:
            s.populate()

    def test_update_with_empty_sources_does_not_raise(self, tmp_path):
        with _make_storage(tmp_path) as s:
            s.update()

    def test_pipeline_does_not_create_silver_scryfall_meta_history(self, tmp_path):
        with _make_storage_with_meta_bronze(tmp_path, [("abc", "2026-05-11")]) as s:
            s.populate()
            tables = {r[0] for r in s._silver_con.execute("SHOW TABLES").fetchall()}
            assert "silver_scryfall_meta_history" not in tables


class TestAppendMetaHistorySql:
    """Unit tests for SilverStorage._append_meta_history_sql."""

    def test_creates_silver_meta_history_table(self, tmp_path):
        with _make_storage_with_meta_bronze(tmp_path, [("abc", "2026-06-20")]) as s:
            s.populate()
            tables = {r[0] for r in s._silver_con.execute("SHOW TABLES").fetchall()}
            assert "silver_meta_history" in tables

    def test_trims_id_and_snapshot_date(self, tmp_path):
        bronze_path = str(tmp_path / "bronze.duckdb")
        con = duckdb.connect(bronze_path)
        con.execute("""
            CREATE TABLE bronze_scryfall_meta_history (
                id VARCHAR, snapshot_date VARCHAR, legalities VARCHAR,
                edhrec_rank DOUBLE, reserved BOOLEAN, promo_types VARCHAR, finishes VARCHAR
            )
        """)
        con.execute(
            "INSERT INTO bronze_scryfall_meta_history VALUES (?, ?, NULL, NULL, false, '[]', '[]')",
            ["  abc  ", " 2026-06-20 "],
        )
        con.close()
        config_path = tmp_path / "cfg.json"
        config_path.write_text(json.dumps({**MINIMAL_CONFIG, "sources": {}}))
        with SilverStorage(bronze_path, ":memory:", str(config_path)) as s:
            s._append_meta_history_sql()
            row = s._silver_con.execute(
                "SELECT id, snapshot_date FROM silver_meta_history"
            ).fetchone()
            assert row == ("abc", "2026-06-20")

    def test_casts_edhrec_rank_to_integer(self, tmp_path):
        bronze_path = str(tmp_path / "bronze.duckdb")
        con = duckdb.connect(bronze_path)
        con.execute("""
            CREATE TABLE bronze_scryfall_meta_history (
                id VARCHAR, snapshot_date VARCHAR, legalities VARCHAR,
                edhrec_rank DOUBLE, reserved BOOLEAN, promo_types VARCHAR, finishes VARCHAR
            )
        """)
        con.execute(
            "INSERT INTO bronze_scryfall_meta_history VALUES (?, ?, NULL, ?, false, '[]', '[]')",
            ["x", "2026-06-20", 42.0],
        )
        con.close()
        config_path = tmp_path / "cfg.json"
        config_path.write_text(json.dumps({**MINIMAL_CONFIG, "sources": {}}))
        with SilverStorage(bronze_path, ":memory:", str(config_path)) as s:
            s._append_meta_history_sql()
            row = s._silver_con.execute(
                "SELECT edhrec_rank FROM silver_meta_history"
            ).fetchone()
            assert row is not None and row[0] == 42

    def test_renames_reserved_to_is_reserved_and_fills_nulls(self, tmp_path):
        bronze_path = str(tmp_path / "bronze.duckdb")
        con = duckdb.connect(bronze_path)
        con.execute("""
            CREATE TABLE bronze_scryfall_meta_history (
                id VARCHAR, snapshot_date VARCHAR, legalities VARCHAR,
                edhrec_rank DOUBLE, reserved BOOLEAN, promo_types VARCHAR, finishes VARCHAR
            )
        """)
        con.execute(
            "INSERT INTO bronze_scryfall_meta_history VALUES (?, ?, NULL, NULL, ?, '[]', '[]')",
            ["a", "2026-06-20", None],
        )
        con.execute(
            "INSERT INTO bronze_scryfall_meta_history VALUES (?, ?, NULL, NULL, ?, '[]', '[]')",
            ["b", "2026-06-20", True],
        )
        con.close()
        config_path = tmp_path / "cfg.json"
        config_path.write_text(json.dumps({**MINIMAL_CONFIG, "sources": {}}))
        with SilverStorage(bronze_path, ":memory:", str(config_path)) as s:
            s._append_meta_history_sql()
            cols = [
                r[0]
                for r in s._silver_con.execute(
                    "DESCRIBE silver_meta_history"
                ).fetchall()
            ]
            assert "is_reserved" in cols
            assert "reserved" not in cols
            rows = dict(
                s._silver_con.execute(
                    "SELECT id, is_reserved FROM silver_meta_history ORDER BY id"
                ).fetchall()
            )
            assert rows == {"a": False, "b": True}

    def test_lowercases_promo_types_and_fills_null_finishes(self, tmp_path):
        bronze_path = str(tmp_path / "bronze.duckdb")
        con = duckdb.connect(bronze_path)
        con.execute("""
            CREATE TABLE bronze_scryfall_meta_history (
                id VARCHAR, snapshot_date VARCHAR, legalities VARCHAR,
                edhrec_rank DOUBLE, reserved BOOLEAN, promo_types VARCHAR, finishes VARCHAR
            )
        """)
        con.execute(
            "INSERT INTO bronze_scryfall_meta_history VALUES (?, ?, NULL, NULL, false, ?, ?)",
            ["x", "2026-06-20", '["Nonfoil","Foil"]', None],
        )
        con.close()
        config_path = tmp_path / "cfg.json"
        config_path.write_text(json.dumps({**MINIMAL_CONFIG, "sources": {}}))
        with SilverStorage(bronze_path, ":memory:", str(config_path)) as s:
            s._append_meta_history_sql()
            row = s._silver_con.execute(
                "SELECT promo_types, finishes FROM silver_meta_history"
            ).fetchone()
            assert row is not None
            assert row[0] == '["nonfoil","foil"]'
            assert row[1] == "[]"

    def test_skips_duplicate_id_snapshot_date_pairs(self, tmp_path):
        with _make_storage_with_meta_bronze(tmp_path, [("abc", "2026-06-20")]) as s:
            s.populate()
            s._append_meta_history_sql()  # second call must not insert duplicate
            row = s._silver_con.execute(
                "SELECT count(*) FROM silver_meta_history"
            ).fetchone()
            assert row is not None and row[0] == 1

    def test_filters_to_ids_in_silver_cards(self, tmp_path):
        s = _make_storage_with_meta_bronze(
            tmp_path,
            [("id-keep", "2026-06-20"), ("id-drop", "2026-06-20")],
        )
        with s:
            # Pre-seed silver_cards with only "id-keep" so that _append_meta_history_sql
            # filters meta rows to ids present in silver_cards.
            s._silver_con.execute("CREATE TABLE silver_cards (scryfall_id VARCHAR)")
            s._silver_con.execute("INSERT INTO silver_cards VALUES ('id-keep')")
            s._append_meta_history_sql()
            row = s._silver_con.execute(
                "SELECT count(*) FROM silver_meta_history"
            ).fetchone()
            assert row is not None and row[0] == 1
            kept = s._silver_con.execute(
                "SELECT id FROM silver_meta_history"
            ).fetchone()
            assert kept is not None and kept[0] == "id-keep"

    def test_writes_all_rows_when_silver_cards_absent(self, tmp_path):
        with _make_storage_with_meta_bronze(
            tmp_path, [("abc", "2026-06-20"), ("def", "2026-06-20")]
        ) as s:
            s._append_meta_history_sql()
            row = s._silver_con.execute(
                "SELECT count(*) FROM silver_meta_history"
            ).fetchone()
            assert row is not None and row[0] == 2


# ---------------------------------------------------------------------------
# Helpers for SilverPriceBuilder tests
# ---------------------------------------------------------------------------

_SCRYFALL_PRICE_COLS = {
    "eur": 3.20,
    "eur_foil": 8.50,
    "usd": 3.50,
    "usd_foil": 9.00,
    "tix": 0.05,
}


def _scryfall_hist(*ids_and_dates: tuple[str, str]) -> pd.DataFrame:
    """Build a bronze_scryfall_prices_history DataFrame with scalar price columns."""
    ids, dates = zip(*ids_and_dates) if ids_and_dates else ([], [])
    n = len(ids)
    return pd.DataFrame(
        {
            "id": list(ids),
            "snapshot_date": list(dates),
            "eur": [3.20] * n,
            "eur_foil": [8.50] * n,
            "usd": [3.50] * n,
            "usd_foil": [9.00] * n,
            "tix": [0.05] * n,
        }
    )


def _mtgjson_eav_hist(*uuids_and_dates: tuple[str, str]) -> pd.DataFrame:
    """Build a bronze_mtgjson_prices_history EAV DataFrame with 6 price rows per card."""
    rows: list[dict] = []
    combos = [
        ("cardmarket", "retail", "normal", 3.20),
        ("cardmarket", "retail", "foil", 8.50),
        ("cardmarket", "buylist", "normal", 1.80),
        ("tcgplayer", "retail", "normal", 3.50),
        ("tcgplayer", "retail", "foil", 9.00),
        ("tcgplayer", "buylist", "normal", 2.10),
    ]
    for uuid, date in uuids_and_dates:
        for retailer, tx_type, finish, price in combos:
            rows.append(
                {
                    "uuid": uuid,
                    "snapshot_date": date,
                    "retailer": retailer,
                    "tx_type": tx_type,
                    "finish": finish,
                    "price": price,
                }
            )
    return (
        pd.DataFrame(rows)
        if rows
        else pd.DataFrame(
            columns=["uuid", "snapshot_date", "retailer", "tx_type", "finish", "price"]
        )
    )


def _make_storage_with_bronze(
    tmp_path: Path, bronze_tables: dict[str, pd.DataFrame]
) -> SilverStorage:
    """Create SilverStorage backed by a pre-populated Bronze DuckDB file."""
    config_path = tmp_path / "silver_config.json"
    config_path.write_text(json.dumps(MINIMAL_CONFIG))
    bronze_path = str(tmp_path / "bronze.duckdb")

    con = duckdb.connect(bronze_path)
    for table_name, df in bronze_tables.items():
        con.register("_df", df)
        con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM _df")
        con.unregister("_df")
    con.close()

    return SilverStorage(bronze_path, ":memory:", str(config_path))


def _seed_silver_cards(storage: SilverStorage, rows: list[tuple]) -> None:
    """Insert rows into the in-memory silver_cards table as (uuid, scryfall_id[, canonical_uuid, language])."""
    storage._silver_con.execute(
        "CREATE TABLE IF NOT EXISTS silver_cards "
        "(uuid VARCHAR, scryfall_id VARCHAR, canonical_uuid VARCHAR, language VARCHAR)"
    )
    for row in rows:
        uuid = row[0]
        scryfall_id = row[1]
        canonical_uuid = row[2] if len(row) > 2 else uuid
        language = row[3] if len(row) > 3 else "English"
        storage._silver_con.execute(
            "INSERT INTO silver_cards VALUES (?, ?, ?, ?)",
            [uuid, scryfall_id, canonical_uuid, language],
        )


# ---------------------------------------------------------------------------
# SilverPriceBuilder.build
# ---------------------------------------------------------------------------


class TestSilverPriceBuilder:
    def test_returns_empty_dataframe_when_silver_cards_missing(self, tmp_path):
        with _make_storage_with_bronze(
            tmp_path,
            {"bronze_scryfall_prices_history": _scryfall_hist(("s1", "2026-05-11"))},
        ) as s:
            result = s._prices.build("2026-05-11")
            assert result.empty

    def test_returns_empty_dataframe_when_bronze_scryfall_history_missing(
        self, tmp_path
    ):
        with _make_storage_with_bronze(tmp_path, {}) as s:
            _seed_silver_cards(s, [("u1", "s1")])
            result = s._prices.build("2026-05-11")
            assert result.empty

    def test_happy_path_both_sources_present(self, tmp_path):
        with _make_storage_with_bronze(
            tmp_path,
            {
                "bronze_scryfall_prices_history": _scryfall_hist(("s1", "2026-05-11")),
                "bronze_mtgjson_prices_history": _mtgjson_eav_hist(
                    ("u1", "2026-05-11")
                ),
            },
        ) as s:
            _seed_silver_cards(s, [("u1", "s1")])
            result = s._prices.build("2026-05-11")

            assert len(result) == 1
            row = result.iloc[0]
            assert row["uuid"] == "u1"
            assert row["scryfall_id"] == "s1"
            assert row["eur"] == pytest.approx(3.20)
            assert row["cardmarket_eur"] == pytest.approx(3.20)
            assert row["cardmarket_buylist_eur"] == pytest.approx(1.80)
            assert row["tcgplayer_usd"] == pytest.approx(3.50)

    def test_happy_path_has_all_expected_columns(self, tmp_path):
        with _make_storage_with_bronze(
            tmp_path,
            {"bronze_scryfall_prices_history": _scryfall_hist(("s1", "2026-05-11"))},
        ) as s:
            _seed_silver_cards(s, [("u1", "s1")])
            result = s._prices.build("2026-05-11")

            expected_columns = [
                "uuid",
                "scryfall_id",
                "snapshot_date",
                "eur",
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
            assert list(result.columns) == expected_columns

    def test_mtgjson_missing_fills_columns_with_none(self, tmp_path):
        with _make_storage_with_bronze(
            tmp_path,
            {"bronze_scryfall_prices_history": _scryfall_hist(("s1", "2026-05-11"))},
        ) as s:
            _seed_silver_cards(s, [("u1", "s1")])
            result = s._prices.build("2026-05-11")

            assert len(result) == 1
            assert pd.isna(result.iloc[0]["cardmarket_eur"])
            assert pd.isna(result.iloc[0]["tcgplayer_usd"])

    def test_scryfall_card_with_no_silver_match_is_dropped(self, tmp_path):
        with _make_storage_with_bronze(
            tmp_path,
            {
                "bronze_scryfall_prices_history": _scryfall_hist(
                    ("s1", "2026-05-11"), ("s-no-match", "2026-05-11")
                )
            },
        ) as s:
            _seed_silver_cards(s, [("u1", "s1")])
            result = s._prices.build("2026-05-11")

            assert len(result) == 1
            assert result.iloc[0]["scryfall_id"] == "s1"

    def test_mtgjson_card_with_no_scryfall_history_row_is_excluded(self, tmp_path):
        with _make_storage_with_bronze(
            tmp_path,
            {
                "bronze_scryfall_prices_history": _scryfall_hist(("s1", "2026-05-11")),
                "bronze_mtgjson_prices_history": _mtgjson_eav_hist(
                    ("u1", "2026-05-11"), ("u-no-scryfall", "2026-05-11")
                ),
            },
        ) as s:
            _seed_silver_cards(s, [("u1", "s1"), ("u-no-scryfall", None)])
            result = s._prices.build("2026-05-11")

            assert len(result) == 1
            assert result.iloc[0]["uuid"] == "u1"

    def test_build_ignores_bronze_rows_from_other_dates(self, tmp_path):
        with _make_storage_with_bronze(
            tmp_path,
            {
                "bronze_scryfall_prices_history": _scryfall_hist(
                    ("s1", "2026-05-10"), ("s1", "2026-05-11")
                )
            },
        ) as s:
            _seed_silver_cards(s, [("u1", "s1")])
            result = s._prices.build("2026-05-11")

            assert len(result) == 1
            assert result.iloc[0]["snapshot_date"] == "2026-05-11"

    def test_english_card_with_stale_scryfall_id_uses_canonical_uuid(self, tmp_path):
        # Simulate an English paper card where MTGJson holds a stale scryfall_id:
        # the direct scryfall_id→uuid join in card_join missed, leaving uuid=NULL,
        # but (set_code, collector_number) resolved canonical_uuid="u1".
        # The card's current Scryfall ID "s-stale" has real prices and must be
        # included in silver_prices_history under canonical_uuid.
        with _make_storage_with_bronze(
            tmp_path,
            {
                "bronze_scryfall_prices_history": _scryfall_hist(
                    ("s-stale", "2026-05-11")
                )
            },
        ) as s:
            # uuid=None forces COALESCE path; canonical_uuid resolves to "u1"
            _seed_silver_cards(s, [(None, "s-stale", "u1", "English")])
            result = s._prices.build("2026-05-11")

            assert len(result) == 1
            assert result.iloc[0]["uuid"] == "u1"
            assert result.iloc[0]["scryfall_id"] == "s-stale"
            assert result.iloc[0]["eur"] == pytest.approx(3.20)

    def test_non_english_canonical_uuid_card_excluded_from_main_prices(self, tmp_path):
        # Non-English language variants (uuid=NULL, canonical_uuid=NOT NULL, language≠English)
        # must NOT appear in the main price history — they are handled by
        # build_language_prices to avoid duplicating the canonical card's prices.
        with _make_storage_with_bronze(
            tmp_path,
            {
                "bronze_scryfall_prices_history": _scryfall_hist(
                    ("s1-en", "2026-05-11"), ("s1-ja", "2026-05-11")
                )
            },
        ) as s:
            _seed_silver_cards(s, [("u1", "s1-en")])
            _seed_silver_language_variant_cards(s, [("s1-ja", "u1", "Japanese")])
            result = s._prices.build("2026-05-11")

            assert len(result) == 1
            assert result.iloc[0]["scryfall_id"] == "s1-en"


# ---------------------------------------------------------------------------
# SilverPriceBuilder._fill_price_history
# ---------------------------------------------------------------------------


def _make_price_df(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal silver_prices_history DataFrame for fill tests."""
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
                "uuid": r.get("uuid"),
                "scryfall_id": r.get("scryfall_id"),
                "snapshot_date": r["snapshot_date"],
                "eur": r.get("eur"),
                **{c: r.get(c) for c in null_cols},
            }
            for r in rows
        ]
    )


def _seed_silver_prices_history(storage: SilverStorage, rows: list[dict]) -> None:
    """Insert rows into silver_prices_history to simulate prior-day snapshots."""
    df = _make_price_df(rows)
    storage._silver_con.register("_ph", df)
    storage._silver_con.execute(
        "CREATE TABLE IF NOT EXISTS silver_prices_history AS SELECT * FROM _ph"
    )
    storage._silver_con.unregister("_ph")


class TestFillPriceHistory:
    def test_empty_df_returned_unchanged(self, storage):
        df = _make_price_df([])
        result = storage._prices._fill_price_history(df, "2026-05-11")
        assert result.empty

    def test_no_null_prices_returns_df_unchanged(self, storage):
        df = _make_price_df(
            [
                {
                    "uuid": "u1",
                    "scryfall_id": "s1",
                    "snapshot_date": "2026-05-11",
                    "eur": 5.0,
                }
            ]
        )
        result = storage._prices._fill_price_history(df, "2026-05-11")
        assert len(result) == 1
        assert result.iloc[0]["eur"] == pytest.approx(5.0)

    def test_no_prior_silver_table_returns_df_unchanged(self, storage):
        df = _make_price_df(
            [
                {
                    "uuid": "u1",
                    "scryfall_id": "s1",
                    "snapshot_date": "2026-05-11",
                    "eur": None,
                }
            ]
        )
        result = storage._prices._fill_price_history(df, "2026-05-11")
        assert len(result) == 1
        assert pd.isna(result.iloc[0]["eur"])

    def test_null_prices_filled_from_prior_silver_row(self, storage):
        _seed_silver_prices_history(
            storage,
            [
                {
                    "uuid": "u1",
                    "scryfall_id": "s1",
                    "snapshot_date": "2026-05-10",
                    "eur": 5.0,
                }
            ],
        )
        df = _make_price_df(
            [
                {
                    "uuid": "u1",
                    "scryfall_id": "s1",
                    "snapshot_date": "2026-05-11",
                    "eur": None,
                }
            ]
        )
        result = storage._prices._fill_price_history(df, "2026-05-11")
        assert result.iloc[0]["eur"] == pytest.approx(5.0)

    def test_non_null_prices_not_overwritten_by_prior_silver(self, storage):
        _seed_silver_prices_history(
            storage,
            [
                {
                    "uuid": "u1",
                    "scryfall_id": "s1",
                    "snapshot_date": "2026-05-10",
                    "eur": 99.0,
                }
            ],
        )
        df = _make_price_df(
            [
                {
                    "uuid": "u1",
                    "scryfall_id": "s1",
                    "snapshot_date": "2026-05-11",
                    "eur": 5.0,
                }
            ]
        )
        result = storage._prices._fill_price_history(df, "2026-05-11")
        assert result.iloc[0]["eur"] == pytest.approx(5.0)

    def test_no_prior_silver_row_leaves_null(self, storage):
        # silver_prices_history exists but has no row for this card
        _seed_silver_prices_history(
            storage,
            [
                {
                    "uuid": "u2",
                    "scryfall_id": "s2",
                    "snapshot_date": "2026-05-10",
                    "eur": 99.0,
                }
            ],
        )
        df = _make_price_df(
            [
                {
                    "uuid": "u1",
                    "scryfall_id": "s1",
                    "snapshot_date": "2026-05-11",
                    "eur": None,
                }
            ]
        )
        result = storage._prices._fill_price_history(df, "2026-05-11")
        assert pd.isna(result.iloc[0]["eur"])

    def test_two_cards_do_not_mix_fills(self, storage):
        _seed_silver_prices_history(
            storage,
            [
                {
                    "uuid": "u1",
                    "scryfall_id": "s1",
                    "snapshot_date": "2026-05-10",
                    "eur": 5.0,
                },
                {
                    "uuid": "u2",
                    "scryfall_id": "s2",
                    "snapshot_date": "2026-05-10",
                    "eur": 99.0,
                },
            ],
        )
        df = _make_price_df(
            [
                {
                    "uuid": "u1",
                    "scryfall_id": "s1",
                    "snapshot_date": "2026-05-11",
                    "eur": None,
                },
                {
                    "uuid": "u2",
                    "scryfall_id": "s2",
                    "snapshot_date": "2026-05-11",
                    "eur": None,
                },
            ]
        )
        result = storage._prices._fill_price_history(df, "2026-05-11")
        assert result[result["scryfall_id"] == "s1"].iloc[0]["eur"] == pytest.approx(
            5.0
        )
        assert result[result["scryfall_id"] == "s2"].iloc[0]["eur"] == pytest.approx(
            99.0
        )

    def test_most_recent_prior_row_used_when_multiple_exist(self, storage):
        _seed_silver_prices_history(
            storage,
            [
                {
                    "uuid": "u1",
                    "scryfall_id": "s1",
                    "snapshot_date": "2026-05-08",
                    "eur": 1.0,
                },
                {
                    "uuid": "u1",
                    "scryfall_id": "s1",
                    "snapshot_date": "2026-05-10",
                    "eur": 7.0,
                },
            ],
        )
        df = _make_price_df(
            [
                {
                    "uuid": "u1",
                    "scryfall_id": "s1",
                    "snapshot_date": "2026-05-11",
                    "eur": None,
                }
            ]
        )
        result = storage._prices._fill_price_history(df, "2026-05-11")
        assert result.iloc[0]["eur"] == pytest.approx(7.0)

    def test_same_date_row_not_used_as_fill_source(self, storage):
        # snapshot_date = today must not be used as the fill source (WHERE < today)
        _seed_silver_prices_history(
            storage,
            [
                {
                    "uuid": "u1",
                    "scryfall_id": "s1",
                    "snapshot_date": "2026-05-11",
                    "eur": 99.0,
                }
            ],
        )
        df = _make_price_df(
            [
                {
                    "uuid": "u1",
                    "scryfall_id": "s1",
                    "snapshot_date": "2026-05-11",
                    "eur": None,
                }
            ]
        )
        result = storage._prices._fill_price_history(df, "2026-05-11")
        assert pd.isna(result.iloc[0]["eur"])

    def test_column_order_preserved(self, storage):
        df = _make_price_df(
            [
                {
                    "uuid": "u1",
                    "scryfall_id": "s1",
                    "snapshot_date": "2026-05-11",
                    "eur": 5.0,
                }
            ]
        )
        result = storage._prices._fill_price_history(df, "2026-05-11")
        assert list(result.columns) == list(df.columns)


# ---------------------------------------------------------------------------
# SilverPriceBuilder.build_language_prices
# ---------------------------------------------------------------------------


def _seed_silver_language_variant_cards(
    storage: SilverStorage, rows: list[tuple]
) -> None:
    """Insert language-variant rows into silver_cards as (uuid=NULL, scryfall_id, canonical_uuid, language)."""
    storage._silver_con.execute(
        "CREATE TABLE IF NOT EXISTS silver_cards "
        "(uuid VARCHAR, scryfall_id VARCHAR, canonical_uuid VARCHAR, language VARCHAR)"
    )
    for scryfall_id, canonical_uuid, language in rows:
        storage._silver_con.execute(
            "INSERT INTO silver_cards VALUES (NULL, ?, ?, ?)",
            [scryfall_id, canonical_uuid, language],
        )


class TestBuildLanguagePrices:
    def test_returns_empty_when_silver_cards_missing(self, tmp_path):
        with _make_storage_with_bronze(
            tmp_path,
            {"bronze_scryfall_prices_history": _scryfall_hist(("s1-ja", "2026-05-11"))},
        ) as s:
            result = s._prices.build_language_prices("2026-05-11")
            assert result.empty

    def test_returns_empty_when_no_language_variant_cards(self, tmp_path):
        with _make_storage_with_bronze(
            tmp_path,
            {
                "bronze_scryfall_prices_history": _scryfall_hist(("s1", "2026-05-11")),
                "bronze_scryfall_cards": pd.DataFrame({"id": ["s1"], "lang": ["en"]}),
            },
        ) as s:
            # Only English card (uuid is not NULL) — no language variants
            _seed_silver_cards(s, [("u1", "s1")])
            result = s._prices.build_language_prices("2026-05-11")
            assert result.empty

    def test_happy_path_language_variant_gets_prices(self, tmp_path):
        with _make_storage_with_bronze(
            tmp_path,
            {
                "bronze_scryfall_prices_history": _scryfall_hist(
                    ("s1-ja", "2026-05-11")
                ),
                "bronze_scryfall_cards": pd.DataFrame(
                    {"id": ["s1-ja"], "lang": ["ja"]}
                ),
            },
        ) as s:
            _seed_silver_language_variant_cards(s, [("s1-ja", "u1", "Japanese")])
            result = s._prices.build_language_prices("2026-05-11")

            assert len(result) == 1
            row = result.iloc[0]
            assert row["scryfall_id"] == "s1-ja"
            assert row["canonical_uuid"] == "u1"
            assert row["lang"] == "ja"
            assert row["eur"] == pytest.approx(3.20)

    def test_has_expected_columns(self, tmp_path):
        with _make_storage_with_bronze(
            tmp_path,
            {
                "bronze_scryfall_prices_history": _scryfall_hist(
                    ("s1-ja", "2026-05-11")
                ),
                "bronze_scryfall_cards": pd.DataFrame(
                    {"id": ["s1-ja"], "lang": ["ja"]}
                ),
            },
        ) as s:
            _seed_silver_language_variant_cards(s, [("s1-ja", "u1", "Japanese")])
            result = s._prices.build_language_prices("2026-05-11")

            expected = [
                "scryfall_id",
                "canonical_uuid",
                "lang",
                "snapshot_date",
                "eur",
                "eur_foil",
                "usd",
                "usd_foil",
            ]
            assert list(result.columns) == expected

    def test_english_card_scryfall_id_not_included(self, tmp_path):
        # English card has uuid NOT NULL — must not appear in language prices
        with _make_storage_with_bronze(
            tmp_path,
            {
                "bronze_scryfall_prices_history": _scryfall_hist(
                    ("s1-en", "2026-05-11"), ("s1-ja", "2026-05-11")
                ),
                "bronze_scryfall_cards": pd.DataFrame(
                    {"id": ["s1-en", "s1-ja"], "lang": ["en", "ja"]}
                ),
            },
        ) as s:
            _seed_silver_cards(s, [("u1", "s1-en")])
            _seed_silver_language_variant_cards(s, [("s1-ja", "u1", "Japanese")])
            result = s._prices.build_language_prices("2026-05-11")

            assert len(result) == 1
            assert result.iloc[0]["scryfall_id"] == "s1-ja"


# ---------------------------------------------------------------------------
# SilverStorage._check_oracle_id_conflicts — oracle ID name conflict check (EDA-01 §7)
# ---------------------------------------------------------------------------


class TestOracleIdConflictCheck:
    def _seed_silver_cards_with_oracle(
        self, storage: SilverStorage, rows: list[dict]
    ) -> None:
        """Create silver_cards with name and oracle_id columns for conflict tests."""
        storage._silver_con.execute(
            "CREATE TABLE silver_cards (scryfall_id VARCHAR, name VARCHAR, oracle_id VARCHAR)"
        )
        for row in rows:
            storage._silver_con.execute(
                "INSERT INTO silver_cards VALUES (?, ?, ?)",
                [row["scryfall_id"], row["name"], row["oracle_id"]],
            )

    def test_no_warning_when_all_names_have_unique_oracle_id(self, tmp_path, caplog):
        with _make_storage(tmp_path) as s:
            self._seed_silver_cards_with_oracle(
                s,
                [
                    {"scryfall_id": "s1", "name": "CardA", "oracle_id": "o1"},
                    {"scryfall_id": "s2", "name": "CardB", "oracle_id": "o2"},
                ],
            )
            with caplog.at_level(
                logging.INFO,
                logger="src.data.cards.storage.silver.storage",
            ):
                s._check_oracle_id_conflicts()

        warning_records = [
            r
            for r in caplog.records
            if r.levelno == logging.WARNING and "Oracle ID conflict" in r.message
        ]
        assert warning_records == []

    def test_warning_logged_when_name_maps_to_multiple_oracle_ids(
        self, tmp_path, caplog
    ):
        # "Fire // Ice" appears with two different oracle_ids — split card regression.
        with _make_storage(tmp_path) as s:
            self._seed_silver_cards_with_oracle(
                s,
                [
                    {"scryfall_id": "s1", "name": "Fire // Ice", "oracle_id": "o1"},
                    {"scryfall_id": "s2", "name": "Fire // Ice", "oracle_id": "o2"},
                ],
            )
            with caplog.at_level(
                logging.WARNING,
                logger="src.data.cards.storage.silver.storage",
            ):
                s._check_oracle_id_conflicts()

        warning_records = [
            r
            for r in caplog.records
            if r.levelno == logging.WARNING and "Oracle ID conflict" in r.message
        ]
        assert len(warning_records) == 1
        assert "Fire // Ice" in warning_records[0].message

    def test_warning_reports_true_count_beyond_five_examples(self, tmp_path, caplog):
        # 7 conflicting names exist; the old LIMIT 5 in the query truncated the
        # list before Python counted it, so the log always said "5" regardless
        # of the true total.
        with _make_storage(tmp_path) as s:
            rows = []
            for i in range(7):
                rows.append(
                    {"scryfall_id": f"s{i}a", "name": f"Card{i}", "oracle_id": f"o{i}a"}
                )
                rows.append(
                    {"scryfall_id": f"s{i}b", "name": f"Card{i}", "oracle_id": f"o{i}b"}
                )
            self._seed_silver_cards_with_oracle(s, rows)
            with caplog.at_level(
                logging.WARNING,
                logger="src.data.cards.storage.silver.storage",
            ):
                s._check_oracle_id_conflicts()

        warning_records = [
            r
            for r in caplog.records
            if r.levelno == logging.WARNING and "Oracle ID conflict" in r.message
        ]
        assert len(warning_records) == 1
        assert "7 name(s)" in warning_records[0].message


# ---------------------------------------------------------------------------
# SilverWriter (silver/persistence.py)
# ---------------------------------------------------------------------------


@pytest.fixture
def silver_con():
    con = duckdb.connect(":memory:")
    yield con
    con.close()


class TestSilverWriterAppendLoad:
    def test_append_creates_table_on_first_call(self, silver_con):
        writer = SilverWriter(silver_con)
        df = pd.DataFrame(
            {"uuid": ["a"], "snapshot_date": ["2026-01-01"], "val": [1.0]}
        )
        writer.append(df, "hist", "uuid")
        count = silver_con.execute("SELECT count(*) FROM hist").fetchone()[0]
        assert count == 1

    def test_append_inserts_new_rows_into_existing_table(self, silver_con):
        writer = SilverWriter(silver_con)
        df1 = pd.DataFrame(
            {"uuid": ["a"], "snapshot_date": ["2026-01-01"], "val": [1.0]}
        )
        df2 = pd.DataFrame(
            {"uuid": ["b"], "snapshot_date": ["2026-01-02"], "val": [2.0]}
        )
        writer.append(df1, "hist", "uuid")
        writer.append(df2, "hist", "uuid")
        count = silver_con.execute("SELECT count(*) FROM hist").fetchone()[0]
        assert count == 2

    def test_append_skips_duplicate_key_snapshot_pair(self, silver_con):
        writer = SilverWriter(silver_con)
        df = pd.DataFrame(
            {"uuid": ["a"], "snapshot_date": ["2026-01-01"], "val": [1.0]}
        )
        writer.append(df, "hist", "uuid")
        writer.append(df, "hist", "uuid")  # same key + date — must be skipped
        count = silver_con.execute("SELECT count(*) FROM hist").fetchone()[0]
        assert count == 1

    def test_append_skips_empty_dataframe(self, silver_con):
        writer = SilverWriter(silver_con)
        empty = pd.DataFrame(
            {
                "uuid": pd.Series([], dtype=str),
                "snapshot_date": pd.Series([], dtype=str),
            }
        )
        writer.append(empty, "hist", "uuid")
        tables = silver_con.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_name='hist'"
        ).fetchall()
        assert tables == []

    def test_append_raises_storage_write_error_on_duckdb_error(self):
        from unittest.mock import MagicMock

        mock_con = MagicMock()
        # _table_exists → execute().fetchone() → (1,) means table exists
        table_exists_result = MagicMock()
        table_exists_result.fetchone.return_value = (1,)
        # second execute (the INSERT) raises duckdb.Error
        mock_con.execute.side_effect = [table_exists_result, duckdb.Error("fail")]
        writer = SilverWriter(mock_con)
        df = pd.DataFrame({"uuid": ["a"], "snapshot_date": ["2026-01-01"]})
        with pytest.raises(StorageWriteError):
            writer.append(df, "hist", "uuid")


class TestSilverWriterFullLoad:
    def test_full_load_creates_table(self, silver_con):
        writer = SilverWriter(silver_con)
        df = pd.DataFrame({"uuid": ["x"], "val": [5]})
        writer.full_load(df, "cards")
        count = silver_con.execute("SELECT count(*) FROM cards").fetchone()[0]
        assert count == 1

    def test_full_load_replaces_existing_table(self, silver_con):
        writer = SilverWriter(silver_con)
        df1 = pd.DataFrame({"uuid": ["x"], "val": [1]})
        df2 = pd.DataFrame({"uuid": ["y"], "val": [2]})
        writer.full_load(df1, "cards")
        writer.full_load(df2, "cards")
        result = silver_con.execute("SELECT uuid FROM cards").df()
        assert list(result["uuid"]) == ["y"]

    def test_full_load_skips_empty_dataframe(self, silver_con):
        writer = SilverWriter(silver_con)
        empty = pd.DataFrame({"uuid": pd.Series([], dtype=str)})
        writer.full_load(empty, "cards")
        tables = silver_con.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_name='cards'"
        ).fetchall()
        assert tables == []

    def test_full_load_raises_storage_write_error_on_duckdb_error(self):
        from unittest.mock import MagicMock

        mock_con = MagicMock()
        mock_con.execute.side_effect = duckdb.Error("fail")
        writer = SilverWriter(mock_con)
        df = pd.DataFrame({"uuid": ["x"]})
        with pytest.raises(StorageWriteError):
            writer.full_load(df, "cards")


class TestSilverWriterIncremental:
    def test_incremental_creates_table_on_first_call(self, silver_con):
        writer = SilverWriter(silver_con)
        df = pd.DataFrame({"uuid": ["a"], "val": [1]})
        writer.upsert(df, "cards", "uuid")
        count = silver_con.execute("SELECT count(*) FROM cards").fetchone()[0]
        assert count == 1

    def test_incremental_upserts_existing_key(self, silver_con):
        writer = SilverWriter(silver_con)
        df1 = pd.DataFrame({"uuid": ["a"], "val": [1]})
        writer.upsert(df1, "cards", "uuid")
        df2 = pd.DataFrame({"uuid": ["a"], "val": [99]})
        writer.upsert(df2, "cards", "uuid")
        result = silver_con.execute("SELECT val FROM cards WHERE uuid='a'").fetchone()[
            0
        ]
        assert result == 99

    def test_incremental_leaves_untouched_rows(self, silver_con):
        writer = SilverWriter(silver_con)
        df1 = pd.DataFrame({"uuid": ["a", "b"], "val": [1, 2]})
        writer.upsert(df1, "cards", "uuid")
        df2 = pd.DataFrame({"uuid": ["a"], "val": [10]})
        writer.upsert(df2, "cards", "uuid")
        result = silver_con.execute("SELECT val FROM cards WHERE uuid='b'").fetchone()[
            0
        ]
        assert result == 2

    def test_incremental_skips_empty_dataframe(self, silver_con):
        writer = SilverWriter(silver_con)
        empty = pd.DataFrame(
            {"uuid": pd.Series([], dtype=str), "val": pd.Series([], dtype=int)}
        )
        writer.upsert(empty, "cards", "uuid")
        tables = silver_con.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_name='cards'"
        ).fetchall()
        assert tables == []

    def test_incremental_raises_storage_write_error_on_duckdb_error(self):
        from unittest.mock import MagicMock

        mock_con = MagicMock()
        # _table_exists → execute().fetchone() → (1,) means table exists
        table_exists_result = MagicMock()
        table_exists_result.fetchone.return_value = (1,)
        # second execute (the DELETE) raises duckdb.Error
        mock_con.execute.side_effect = [table_exists_result, duckdb.Error("fail")]
        writer = SilverWriter(mock_con)
        df = pd.DataFrame({"uuid": ["a"], "val": [1]})
        with pytest.raises(StorageWriteError):
            writer.upsert(df, "cards", "uuid")


class TestBuildSilverCardsSql:
    """Unit tests for SilverStorage._build_silver_cards_sql."""

    def test_creates_silver_cards_table(self, tmp_path):
        with _make_storage_with_cards_bronze(
            tmp_path, [_MTGJSON_ROW], [_SCRYFALL_ROW]
        ) as s:
            s._build_silver_cards_sql()
            tables = {r[0] for r in s._silver_con.execute("SHOW TABLES").fetchall()}
            assert "silver_cards" in tables

    def test_filters_out_is_online_only_mtgjson_rows(self, tmp_path):
        online_row = {
            **_MTGJSON_ROW,
            "uuid": "uuid-online",
            "is_online_only": True,
            "identifiers": '{"scryfall_id": "scryfall-online"}',
        }
        scryfall_online = {**_SCRYFALL_ROW, "id": "scryfall-online"}
        with _make_storage_with_cards_bronze(
            tmp_path,
            [_MTGJSON_ROW, online_row],
            [_SCRYFALL_ROW, scryfall_online],
        ) as s:
            s._build_silver_cards_sql()
            uuids = {
                r[0]
                for r in s._silver_con.execute(
                    "SELECT uuid FROM silver_cards WHERE uuid IS NOT NULL"
                ).fetchall()
            }
            assert "uuid-a" in uuids
            assert "uuid-online" not in uuids

    def test_filters_out_digital_scryfall_rows(self, tmp_path):
        digital_scryfall = {**_SCRYFALL_ROW, "id": "scryfall-digital", "digital": True}
        with _make_storage_with_cards_bronze(
            tmp_path, [_MTGJSON_ROW], [_SCRYFALL_ROW, digital_scryfall]
        ) as s:
            s._build_silver_cards_sql()
            ids = {
                r[0]
                for r in s._silver_con.execute(
                    "SELECT scryfall_id FROM silver_cards"
                ).fetchall()
            }
            assert "scryfall-digital" not in ids

    def test_filters_out_token_layout_scryfall_rows(self, tmp_path):
        token_scryfall = {**_SCRYFALL_ROW, "id": "scryfall-token", "layout": "token"}
        with _make_storage_with_cards_bronze(
            tmp_path, [_MTGJSON_ROW], [_SCRYFALL_ROW, token_scryfall]
        ) as s:
            s._build_silver_cards_sql()
            ids = {
                r[0]
                for r in s._silver_con.execute(
                    "SELECT scryfall_id FROM silver_cards"
                ).fetchall()
            }
            assert "scryfall-token" not in ids

    def test_filters_out_funny_and_memorabilia_set_type_scryfall_rows(self, tmp_path):
        funny_scryfall = {
            **_SCRYFALL_ROW,
            "id": "scryfall-funny",
            "set_type": "funny",
        }
        memorabilia_scryfall = {
            **_SCRYFALL_ROW,
            "id": "scryfall-memorabilia",
            "set_type": "memorabilia",
        }
        with _make_storage_with_cards_bronze(
            tmp_path,
            [_MTGJSON_ROW],
            [_SCRYFALL_ROW, funny_scryfall, memorabilia_scryfall],
        ) as s:
            s._build_silver_cards_sql()
            ids = {
                r[0]
                for r in s._silver_con.execute(
                    "SELECT scryfall_id FROM silver_cards"
                ).fetchall()
            }
            assert "scryfall-funny" not in ids
            assert "scryfall-memorabilia" not in ids

    def test_joined_row_has_uuid_and_oracle_id(self, tmp_path):
        scryfall_with_oracle = {**_SCRYFALL_ROW, "oracle_id": "oracle-a"}
        with _make_storage_with_cards_bronze(
            tmp_path, [_MTGJSON_ROW], [scryfall_with_oracle]
        ) as s:
            s._build_silver_cards_sql()
            row = s._silver_con.execute(
                "SELECT uuid, oracle_id FROM silver_cards WHERE scryfall_id = 'scryfall-a'"
            ).fetchone()
            assert row is not None
            assert row[0] == "uuid-a"
            assert row[1] == "oracle-a"

    def test_scryfall_only_row_has_null_uuid(self, tmp_path):
        scryfall_only = {**_SCRYFALL_ROW, "id": "scryfall-only", "lang": "ja"}
        with _make_storage_with_cards_bronze(
            tmp_path, [_MTGJSON_ROW], [_SCRYFALL_ROW, scryfall_only]
        ) as s:
            s._build_silver_cards_sql()
            row = s._silver_con.execute(
                "SELECT uuid FROM silver_cards WHERE scryfall_id = 'scryfall-only'"
            ).fetchone()
            assert row is not None and row[0] is None

    def test_dfc_scryfall_id_deduplicated_to_one_row(self, tmp_path):
        # DFC: two MTGJson rows share the same scryfall_id (front/back face)
        back_face = {
            **_MTGJSON_ROW,
            "uuid": "uuid-back",
            "name": "Lightning Bolt // Back",
        }
        with _make_storage_with_cards_bronze(
            tmp_path, [_MTGJSON_ROW, back_face], [_SCRYFALL_ROW]
        ) as s:
            s._build_silver_cards_sql()
            count = s._silver_con.execute(
                "SELECT count(*) FROM silver_cards WHERE scryfall_id = 'scryfall-a'"
            ).fetchone()
            assert count is not None and count[0] == 1

    def test_is_commander_legal_extracted_from_legalities(self, tmp_path):
        with _make_storage_with_cards_bronze(
            tmp_path, [_MTGJSON_ROW], [_SCRYFALL_ROW]
        ) as s:
            s._build_silver_cards_sql()
            row = s._silver_con.execute(
                "SELECT is_commander_legal FROM silver_cards WHERE scryfall_id = 'scryfall-a'"
            ).fetchone()
            assert row is not None and row[0] is True

    def test_rarity_lowercased(self, tmp_path):
        row = {**_MTGJSON_ROW, "rarity": "  RARE  "}
        with _make_storage_with_cards_bronze(tmp_path, [row], [_SCRYFALL_ROW]) as s:
            s._build_silver_cards_sql()
            r = s._silver_con.execute(
                "SELECT rarity FROM silver_cards WHERE uuid = 'uuid-a'"
            ).fetchone()
            assert r is not None and r[0] == "rare"

    def test_sentinel_replaced_with_null(self, tmp_path):
        row = {**_MTGJSON_ROW, "artist": "_"}
        with _make_storage_with_cards_bronze(tmp_path, [row], [_SCRYFALL_ROW]) as s:
            s._build_silver_cards_sql()
            r = s._silver_con.execute(
                "SELECT artist FROM silver_cards WHERE uuid = 'uuid-a'"
            ).fetchone()
            assert r is not None and r[0] is None

    def test_colors_stored_as_uppercase_array(self, tmp_path):
        row = {**_MTGJSON_ROW, "colors": '["r"]'}
        with _make_storage_with_cards_bronze(tmp_path, [row], [_SCRYFALL_ROW]) as s:
            s._build_silver_cards_sql()
            r = s._silver_con.execute(
                "SELECT colors FROM silver_cards WHERE uuid = 'uuid-a'"
            ).fetchone()
            assert r is not None and r[0] == ["R"]

    def test_original_type_parsed_into_supertypes_and_subtypes(self, tmp_path):
        row = {**_MTGJSON_ROW, "original_type": "Legendary Creature — Human Wizard"}
        with _make_storage_with_cards_bronze(tmp_path, [row], [_SCRYFALL_ROW]) as s:
            s._build_silver_cards_sql()
            r = s._silver_con.execute(
                "SELECT original_supertypes, original_types, original_subtypes"
                " FROM silver_cards WHERE uuid = 'uuid-a'"
            ).fetchone()
            assert r is not None
            assert "Legendary" in r[0]
            assert "Creature" in r[1]
            assert set(r[2]) == {"Human", "Wizard"}

    def test_errata_true_when_text_differs_from_original(self, tmp_path):
        row = {**_MTGJSON_ROW, "text": "New text", "original_text": "Old text"}
        with _make_storage_with_cards_bronze(tmp_path, [row], [_SCRYFALL_ROW]) as s:
            s._build_silver_cards_sql()
            r = s._silver_con.execute(
                "SELECT errata FROM silver_cards WHERE uuid = 'uuid-a'"
            ).fetchone()
            assert r is not None and r[0] is True

    def test_scryfall_lang_code_mapped_to_full_language_name(self, tmp_path):
        scryfall_ja = {**_SCRYFALL_ROW, "id": "scryfall-ja", "lang": "ja"}
        with _make_storage_with_cards_bronze(tmp_path, [], [scryfall_ja]) as s:
            s._build_silver_cards_sql()
            r = s._silver_con.execute(
                "SELECT language FROM silver_cards WHERE scryfall_id = 'scryfall-ja'"
            ).fetchone()
            assert r is not None and r[0] == "Japanese"

    def test_format_count_is_non_null_and_counts_legal_formats(self, tmp_path):
        # legalities has commander=legal and standard=not_legal → format_count should be 1
        with _make_storage_with_cards_bronze(
            tmp_path, [_MTGJSON_ROW], [_SCRYFALL_ROW]
        ) as s:
            s._build_silver_cards_sql()
            r = s._silver_con.execute(
                "SELECT format_count FROM silver_cards WHERE scryfall_id = 'scryfall-a'"
            ).fetchone()
            assert r is not None and r[0] == 1


# ---------------------------------------------------------------------------
# Task 12: dead code removal verification
# ---------------------------------------------------------------------------


def test_extract_all_prices_removed():
    import src.data.cards.storage.silver.prices as mod

    assert not hasattr(mod.SilverPriceBuilder, "_extract_all_prices"), (
        "_extract_all_prices should have been removed"
    )
