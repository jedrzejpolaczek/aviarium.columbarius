"""Unit tests for src/data/cards/sources/extractors.py."""

from typing import Any

from src.data.cards.sources.extractors import (
    extract_format_staples,
    extract_mtgjson_cards,
    extract_mtgjson_prices,
    extract_mtgtop8_decklist,
    extract_mtgtop8_event_decks,
    extract_mtgtop8_tournament_list,
)


class TestExtractMtgjsonCards:
    def test_flattens_cards_from_multiple_sets(self):
        raw = {
            "data": {
                "LEB": {"cards": [{"name": "Black Lotus"}, {"name": "Mox Pearl"}]},
                "NEO": {"cards": [{"name": "Boseiju"}]},
            }
        }
        result = extract_mtgjson_cards(raw)
        names = [c["name"] for c in result]
        assert len(result) == 3
        assert "Black Lotus" in names
        assert "Boseiju" in names

    def test_set_without_cards_key_is_skipped(self):
        raw = {
            "data": {
                "LEB": {"cards": [{"name": "Lightning Bolt"}]},
                "TST": {},
            }
        }
        result = extract_mtgjson_cards(raw)
        assert len(result) == 1

    def test_empty_data_returns_empty_list(self):
        assert extract_mtgjson_cards({"data": {}}) == []


class TestExtractMtgjsonPrices:
    def test_injects_uuid_from_key(self):
        uuid = "a1b2c3d4-0000-0000-0000-000000000000"
        raw: dict[str, Any] = {"data": {uuid: {"paper": {}}}}
        result = extract_mtgjson_prices(raw)
        assert len(result) == 1
        assert result[0]["uuid"] == uuid

    def test_price_data_is_preserved(self):
        raw = {"data": {"uuid-1": {"paper": {"ck": {"currency": "USD"}}}}}
        result = extract_mtgjson_prices(raw)
        assert result[0]["paper"] == {"ck": {"currency": "USD"}}

    def test_multiple_entries(self):
        raw: dict[str, Any] = {"data": {"uuid-1": {}, "uuid-2": {}, "uuid-3": {}}}
        result = extract_mtgjson_prices(raw)
        uuids = [r["uuid"] for r in result]
        assert len(result) == 3
        assert "uuid-1" in uuids
        assert "uuid-3" in uuids

    def test_empty_data_returns_empty_list(self):
        assert extract_mtgjson_prices({"data": {}}) == []


class TestExtractMtgtop8TournamentList:
    def _html(self, rows: list[tuple[str, str, str]]) -> str:
        """Build minimal mtgtop8 format-page HTML with given (event_id, name, date) rows."""
        row_html = "".join(
            f'<tr class="hover_tr">'
            f"<td></td>"
            f'<td><a href="event?e={eid}&f=MO">{name}</a></td>'
            f"<td></td>"
            f"<td>{date}</td>"
            f"</tr>"
            for eid, name, date in rows
        )
        return f'<html><body><table class="Stable"><tbody>{row_html}</tbody></table></body></html>'

    def test_returns_empty_for_page_with_no_stable_table(self):
        assert extract_mtgtop8_tournament_list("<html></html>", "modern") == []

    def test_parses_single_tournament(self):
        html = self._html([("12345", "MTGO Challenge", "18/05/26")])
        result = extract_mtgtop8_tournament_list(html, "modern")
        assert len(result) == 1
        r = result[0]
        assert r["tournament_id"] == "mtgtop8_12345"
        assert r["event_name"] == "MTGO Challenge"
        assert r["format"] == "modern"
        assert r["tournament_date"] == "2026-05-18"
        assert r["_event_id"] == "12345"

    def test_parses_multiple_tournaments(self):
        html = self._html(
            [("1001", "Event A", "18/05/26"), ("1002", "Event B", "17/05/26")]
        )
        result = extract_mtgtop8_tournament_list(html, "legacy")
        assert len(result) == 2
        assert result[0]["tournament_id"] == "mtgtop8_1001"
        assert result[1]["tournament_id"] == "mtgtop8_1002"

    def test_format_field_matches_argument(self):
        html = self._html([("1", "Event", "18/05/26")])
        result = extract_mtgtop8_tournament_list(html, "pioneer")
        assert result[0]["format"] == "pioneer"

    def test_skips_row_without_event_link(self):
        html = (
            '<html><body><table class="Stable"><tbody>'
            '<tr class="hover_tr"><td></td><td>No link</td><td></td><td>18/05/26</td></tr>'
            "</tbody></table></body></html>"
        )
        assert extract_mtgtop8_tournament_list(html, "modern") == []

    def test_skips_row_with_unparseable_date(self):
        html = self._html([("999", "Bad Date Event", "not-a-date")])
        assert extract_mtgtop8_tournament_list(html, "modern") == []

    def test_rows_outside_stable_table_ignored(self):
        html = (
            "<html><body>"
            '<tr class="hover_tr"><td></td><td><a href="event?e=99&f=MO">Stray</a></td><td></td><td>18/05/26</td></tr>'
            '<table class="Stable"><tbody></tbody></table>'
            "</body></html>"
        )
        assert extract_mtgtop8_tournament_list(html, "modern") == []


class TestExtractMtgtop8Decklist:
    def _call(self, html: str) -> list[dict[str, Any]]:
        return extract_mtgtop8_decklist(
            html,
            tournament_id="mtgtop8_1",
            tournament_date="2026-05-16",
            event_name="MTGO Challenge",
            fmt="modern",
            placement=1,
            player="bokk",
            deck_name="Dimir Control",
        )

    def _html(
        self, md_cards: list[tuple[str, int, str]], sb_cards: list[tuple[str, int, str]]
    ) -> str:
        """Build minimal deck HTML. Each tuple is (ref, copies, name)."""
        md = "".join(
            f'<div id="md{ref}" class="deck_line hover_tr">{copies} <span class=L14>{name}</span></div>'
            for ref, copies, name in md_cards
        )
        sb = "".join(
            f'<div id="sb{ref}" class="deck_line hover_tr">{copies} <span class=L14>{name}</span></div>'
            for ref, copies, name in sb_cards
        )
        return f"<html><body>{md}{sb}</body></html>"

    def test_parses_main_deck_cards(self):
        html = self._html([("abc1", 4, "Lightning Bolt")], [])
        result = self._call(html)
        assert len(result) == 1
        r = result[0]
        assert r["card_name"] == "Lightning Bolt"
        assert r["copies"] == 4
        assert r["is_sideboard"] is False

    def test_parses_sideboard_cards(self):
        html = self._html([], [("abc2", 2, "Chalice of the Void")])
        result = self._call(html)
        assert len(result) == 1
        assert result[0]["is_sideboard"] is True
        assert result[0]["card_name"] == "Chalice of the Void"

    def test_main_and_sideboard_together(self):
        html = self._html(
            [("md1", 4, "Thoughtseize")],
            [("sb1", 3, "Dauthi Voidwalker")],
        )
        result = self._call(html)
        assert len(result) == 2
        main = [r for r in result if not r["is_sideboard"]]
        side = [r for r in result if r["is_sideboard"]]
        assert len(main) == 1 and main[0]["card_name"] == "Thoughtseize"
        assert len(side) == 1 and side[0]["card_name"] == "Dauthi Voidwalker"

    def test_id_contains_tournament_id_and_card_and_sideboard_flag(self):
        html = self._html([("x1", 1, "Island")], [])
        result = self._call(html)
        assert result[0]["id"] == "mtgtop8_1__Island__False"

    def test_metadata_propagated_to_every_row(self):
        html = self._html([("a1", 4, "Fatal Push"), ("a2", 4, "Grief")], [])
        result = self._call(html)
        for r in result:
            assert r["tournament_id"] == "mtgtop8_1"
            assert r["tournament_date"] == "2026-05-16"
            assert r["format"] == "modern"
            assert r["placement"] == 1
            assert r["player"] == "bokk"

    def test_empty_html_returns_empty_list(self):
        assert self._call("<html></html>") == []

    def test_rows_without_count_prefix_are_skipped(self):
        html = '<html><body><div id="md1" class="deck_line hover_tr">LANDS</div></body></html>'
        assert self._call(html) == []


class TestExtractFormatStaples:
    def _row(
        self, rank: int, name: str, deck_pct: str, played: str, has_link: bool = True
    ) -> str:
        name_td = (
            f'<td><a href="/cards/{name}">{name}</a></td>'
            if has_link
            else f"<td>{name}</td>"
        )
        return f"<tr><td>{rank}</td>{name_td}<td>R</td><td>{deck_pct}</td><td>{played}</td></tr>"

    def _html(self, rows: list[tuple]) -> str:
        row_html = "".join(
            self._row(rank, name, deck_pct, played)
            for rank, name, deck_pct, played in rows
        )
        return f'<html><body><table class="table-staples"><tbody>{row_html}</tbody></table></body></html>'

    def test_returns_empty_when_no_table_staples(self):
        assert extract_format_staples("<html></html>", "modern") == []

    def test_parses_single_row(self):
        html = self._html([(1, "Lightning Bolt", "95.5%", "3.8")])
        result = extract_format_staples(html, "modern")
        assert len(result) == 1
        r = result[0]
        assert r["card_name"] == "Lightning Bolt"
        assert r["format"] == "modern"
        assert r["top"] == 1
        assert r["deck_pct"] == 95.5
        assert r["percentage_in_decks"] == 95
        assert r["played"] == 3.8
        assert r["id"] == "Lightning Bolt__modern"

    def test_played_with_comma_separator(self):
        html = self._html([(1, "Island", "100%", "1,000")])
        result = extract_format_staples(html, "pauper")
        assert result[0]["played"] == 1000.0

    def test_multiple_rows(self):
        html = self._html(
            [(1, "Lightning Bolt", "95%", "3.8"), (2, "Thoughtseize", "80%", "2.1")]
        )
        result = extract_format_staples(html, "modern")
        assert len(result) == 2
        assert result[0]["top"] == 1
        assert result[1]["top"] == 2

    def test_row_with_fewer_than_five_cols_skipped(self):
        html = (
            '<html><body><table class="table-staples">'
            "<tr><td>1</td><td>Card</td></tr>"
            "</table></body></html>"
        )
        assert extract_format_staples(html, "modern") == []

    def test_row_with_unparseable_rank_skipped(self):
        html = (
            '<html><body><table class="table-staples">'
            "<tr><td>N/A</td><td>Card</td><td>R</td><td>95%</td><td>3.8</td></tr>"
            "</table></body></html>"
        )
        assert extract_format_staples(html, "modern") == []

    def test_name_fallback_without_link(self):
        row = self._row(1, "DirectName", "50%", "1.5", has_link=False)
        html = f'<html><body><table class="table-staples"><tbody>{row}</tbody></table></body></html>'
        result = extract_format_staples(html, "legacy")
        assert len(result) == 1
        assert result[0]["card_name"] == "DirectName"


class TestExtractMtgtop8EventDecks:
    def _html(self, decks: list[tuple[str, str, str]]) -> str:
        """Build event HTML. Each tuple is (deck_id, player, deck_name)."""
        rows = "".join(
            f'<div class="hover_tr">'
            f'<a href="event?e=1&d={deck_id}&f=MO">{deck_name}</a>'
            f'<span class="G11">{player}</span>'
            f'<span class="S14">{deck_name}</span>'
            f"</div>"
            for deck_id, player, deck_name in decks
        )
        return f"<html><body>{rows}</body></html>"

    def test_returns_empty_for_page_with_no_decks(self):
        assert extract_mtgtop8_event_decks("<html></html>") == []

    def test_parses_single_deck(self):
        html = self._html([("12345", "bokk", "Dimir Control")])
        result = extract_mtgtop8_event_decks(html)
        assert len(result) == 1
        r = result[0]
        assert r["deck_id"] == "12345"
        assert r["player"] == "bokk"
        assert r["deck_name"] == "Dimir Control"
        assert r["placement"] == 1

    def test_placement_increments_per_deck(self):
        html = self._html(
            [("1", "p1", "Deck A"), ("2", "p2", "Deck B"), ("3", "p3", "Deck C")]
        )
        result = extract_mtgtop8_event_decks(html)
        assert [r["placement"] for r in result] == [1, 2, 3]

    def test_caps_at_8_decks(self):
        html = self._html([(str(i), f"p{i}", f"Deck {i}") for i in range(12)])
        assert len(extract_mtgtop8_event_decks(html)) == 8

    def test_skips_rows_without_deck_link(self):
        html = (
            "<html><body>"
            '<div class="hover_tr"><a href="event?e=1&f=MO">No deck id</a></div>'
            '<div class="hover_tr"><a href="event?e=1&d=99&f=MO">Deck</a>'
            '<span class="G11">Player</span><span class="S14">Archetype</span></div>'
            "</body></html>"
        )
        result = extract_mtgtop8_event_decks(html)
        assert len(result) == 1
        assert result[0]["deck_id"] == "99"

    def test_missing_player_element_gives_empty_string(self):
        html = (
            "<html><body>"
            '<div class="hover_tr"><a href="event?e=1&d=55&f=MO">Arch</a>'
            '<span class="S14">Arch</span></div>'
            "</body></html>"
        )
        assert extract_mtgtop8_event_decks(html)[0]["player"] == ""

    def test_deck_name_falls_back_to_link_text(self):
        html = (
            "<html><body>"
            '<div class="hover_tr"><a href="event?e=1&d=77&f=MO">FallbackName</a>'
            '<span class="G11">Player</span></div>'
            "</body></html>"
        )
        assert extract_mtgtop8_event_decks(html)[0]["deck_name"] == "FallbackName"
