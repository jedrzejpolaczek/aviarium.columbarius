"""Unit tests for src/data/cards/sources/{pipeline,scrapers,registry}.py."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from pydantic import BaseModel

from src.data.cards.sources import (
    SourceDownloadError,
    SourceLoadError,
    _ingest_format_staples_async,
    _ingest_tournament_results_async,
    _save_to_json,
    ingesting_pipeline,
    load_from_json,
)
from src.data.cards.sources.scrapers import _cleanup_html_files


class _Simple(BaseModel):
    name: str
    value: int


# ---------------------------------------------------------------------------
# load_from_json
# ---------------------------------------------------------------------------


class TestLoadFromJson:
    def test_valid_file_returns_records(self, tmp_path):
        f = tmp_path / "data.json"
        f.write_text(
            json.dumps([{"name": "Alpha", "value": 1}, {"name": "Beta", "value": 2}])
        )
        records, errors = load_from_json(str(f), _Simple)
        assert len(records) == 2
        assert errors == []
        assert records[0].name == "Alpha"

    def test_invalid_records_go_to_errors(self, tmp_path):
        f = tmp_path / "data.json"
        f.write_text(
            json.dumps(
                [
                    {"name": "Good", "value": 10},
                    {"name": "Bad"},
                    {"name": "AlsoGood", "value": 20},
                ]
            )
        )
        records, errors = load_from_json(str(f), _Simple)
        assert len(records) == 2
        assert len(errors) == 1
        assert errors[0]["name"] == "Bad"

    def test_missing_file_raises_source_load_error(self, tmp_path):
        with pytest.raises(SourceLoadError, match="File not found"):
            load_from_json(str(tmp_path / "missing.json"), _Simple)

    def test_invalid_json_raises_source_load_error(self, tmp_path):
        f = tmp_path / "bad.json"
        f.write_text("{ not valid json }")
        with pytest.raises(SourceLoadError, match="Invalid JSON"):
            load_from_json(str(f), _Simple)

    def test_custom_extractor_is_applied(self, tmp_path):
        f = tmp_path / "data.json"
        f.write_text(json.dumps({"items": [{"name": "X", "value": 99}]}))
        records, errors = load_from_json(
            str(f), _Simple, extractor=lambda raw: raw["items"]
        )
        assert len(records) == 1
        assert records[0].value == 99

    def test_all_invalid_returns_empty_records(self, tmp_path):
        f = tmp_path / "data.json"
        f.write_text(json.dumps([{"name": "No value"}, {"name": "Also no value"}]))
        records, errors = load_from_json(str(f), _Simple)
        assert records == []
        assert len(errors) == 2


# ---------------------------------------------------------------------------
# _save_to_json
# ---------------------------------------------------------------------------


class TestSaveToJson:
    def test_creates_file_with_records(self, tmp_path):
        path = str(tmp_path / "out.json")
        _save_to_json([{"a": 1}, {"a": 2}], path)
        data = json.loads((tmp_path / "out.json").read_text())
        assert data == [{"a": 1}, {"a": 2}]

    def test_overwrites_existing_file(self, tmp_path):
        path = str(tmp_path / "out.json")
        _save_to_json([{"a": 1}], path)
        _save_to_json([{"b": 2}], path)
        data = json.loads((tmp_path / "out.json").read_text())
        assert data == [{"b": 2}]

    def test_creates_parent_directories(self, tmp_path):
        path = str(tmp_path / "nested" / "dir" / "out.json")
        _save_to_json([], path)
        assert (tmp_path / "nested" / "dir" / "out.json").exists()

    def test_empty_list_writes_empty_array(self, tmp_path):
        path = str(tmp_path / "out.json")
        _save_to_json([], path)
        assert json.loads((tmp_path / "out.json").read_text()) == []

    def test_os_error_raises_source_load_error(self, tmp_path):
        path = str(tmp_path / "out.json")
        with patch("pathlib.Path.open", side_effect=OSError("disk full")):
            with pytest.raises(SourceLoadError, match="Failed to write"):
                _save_to_json([{"a": 1}], path)


# ---------------------------------------------------------------------------
# ingesting_pipeline  (public sync wrapper — uses asyncio.run internally)
# ---------------------------------------------------------------------------


def _mock_httpx_client() -> AsyncMock:
    """Return an AsyncMock that behaves like httpx.AsyncClient as context manager."""
    mock = AsyncMock(spec=httpx.AsyncClient)
    mock.__aenter__ = AsyncMock(return_value=mock)
    mock.__aexit__ = AsyncMock(return_value=False)
    return mock


class TestIngestingPipeline:
    def _source(self, source_type="mtgjson_cards", flag=False, path="/tmp/x.json"):
        return {
            "type": source_type,
            "url": "http://example.com",
            "path": path,
            "flag": flag,
        }

    def _config(self, *sources):
        return {"sources": list(sources)}

    @pytest.mark.asyncio
    async def test_flag_false_skips_download(self):
        with (
            patch("src.data.cards.sources.scrapers.download_json_from_url") as mock_dl,
            patch(
                "src.data.cards.sources.scrapers.load_from_json", return_value=([], [])
            ),
            patch(
                "src.data.cards.sources.pipeline.httpx.AsyncClient",
                return_value=_mock_httpx_client(),
            ),
        ):
            await ingesting_pipeline(self._config(self._source(flag=False)))
            mock_dl.assert_not_called()

    @pytest.mark.asyncio
    async def test_flag_false_still_loads_file(self):
        with (
            patch("src.data.cards.sources.scrapers.download_json_from_url"),
            patch(
                "src.data.cards.sources.scrapers.load_from_json",
                return_value=(["r"], []),
            ) as mock_load,
            patch(
                "src.data.cards.sources.pipeline.httpx.AsyncClient",
                return_value=_mock_httpx_client(),
            ),
        ):
            result = await ingesting_pipeline(self._config(self._source(flag=False)))
            mock_load.assert_called_once()
            assert result["mtgjson_cards"] == (["r"], [])

    @pytest.mark.asyncio
    async def test_flag_true_downloads_then_loads(self):
        with (
            patch("src.data.cards.sources.scrapers.download_json_from_url") as mock_dl,
            patch(
                "src.data.cards.sources.scrapers.load_from_json", return_value=([], [])
            ),
            patch(
                "src.data.cards.sources.pipeline.httpx.AsyncClient",
                return_value=_mock_httpx_client(),
            ),
        ):
            await ingesting_pipeline(
                self._config(self._source(source_type="mtgjson_cards", flag=True))
            )
            mock_dl.assert_called_once()

    @pytest.mark.asyncio
    async def test_scryfall_resolves_download_uri_before_download(self):
        meta_response = MagicMock()
        meta_response.raise_for_status.return_value = None
        meta_response.json.return_value = {
            "download_uri": "http://cdn.scryfall.com/cards.json"
        }
        mock_client = _mock_httpx_client()
        mock_client.get.return_value = meta_response

        with (
            patch(
                "src.data.cards.sources.pipeline.httpx.AsyncClient",
                return_value=mock_client,
            ),
            patch("src.data.cards.sources.scrapers.download_json_from_url") as mock_dl,
            patch(
                "src.data.cards.sources.scrapers.load_from_json", return_value=([], [])
            ),
        ):
            await ingesting_pipeline(
                self._config(self._source(source_type="scryfall", flag=True))
            )

        # args: (client, url, path) — check the resolved URL is the CDN URI
        assert mock_dl.call_args[0][1] == "http://cdn.scryfall.com/cards.json"

    @pytest.mark.asyncio
    async def test_unknown_source_type_is_skipped(self):
        with patch(
            "src.data.cards.sources.pipeline.httpx.AsyncClient",
            return_value=_mock_httpx_client(),
        ):
            result = await ingesting_pipeline(
                self._config(self._source(source_type="unknown_source"))
            )
        assert "unknown_source" not in result

    @pytest.mark.asyncio
    async def test_failed_source_does_not_stop_other_sources(self):
        with (
            patch("src.data.cards.sources.scrapers.download_json_from_url"),
            patch(
                "src.data.cards.sources.scrapers.load_from_json",
                side_effect=[SourceLoadError("file missing"), (["r"], [])],
            ),
            patch(
                "src.data.cards.sources.pipeline.httpx.AsyncClient",
                return_value=_mock_httpx_client(),
            ),
        ):
            result = await ingesting_pipeline(
                self._config(
                    self._source(source_type="mtgjson_cards", flag=False),
                    self._source(source_type="mtgjson_prices", flag=False),
                )
            )
            assert "mtgjson_cards" not in result
            assert "mtgjson_prices" in result

    @pytest.mark.asyncio
    async def test_returns_dict_keyed_by_source_type(self):
        with (
            patch("src.data.cards.sources.scrapers.download_json_from_url"),
            patch(
                "src.data.cards.sources.scrapers.load_from_json", return_value=([], [])
            ),
            patch(
                "src.data.cards.sources.pipeline.httpx.AsyncClient",
                return_value=_mock_httpx_client(),
            ),
        ):
            result = await ingesting_pipeline(
                self._config(
                    self._source(source_type="mtgjson_cards"),
                    self._source(source_type="mtgjson_prices"),
                )
            )
            assert set(result.keys()) == {"mtgjson_cards", "mtgjson_prices"}

    @pytest.mark.asyncio
    async def test_scryfall_meta_http_error_is_caught_and_source_skipped(self):
        meta_response = MagicMock()
        meta_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "403",
            request=MagicMock(spec=httpx.Request),
            response=MagicMock(spec=httpx.Response, status_code=403),
        )
        mock_client = _mock_httpx_client()
        mock_client.get.return_value = meta_response

        with patch(
            "src.data.cards.sources.pipeline.httpx.AsyncClient",
            return_value=mock_client,
        ):
            result = await ingesting_pipeline(
                self._config(self._source(source_type="scryfall", flag=True))
            )
        assert "scryfall" not in result

    @pytest.mark.asyncio
    async def test_scryfall_meta_missing_key_is_caught_and_source_skipped(self):
        meta_response = MagicMock()
        meta_response.raise_for_status.return_value = None
        meta_response.json.return_value = {}  # no "download_uri" key → KeyError
        mock_client = _mock_httpx_client()
        mock_client.get.return_value = meta_response

        with patch(
            "src.data.cards.sources.pipeline.httpx.AsyncClient",
            return_value=mock_client,
        ):
            result = await ingesting_pipeline(
                self._config(self._source(source_type="scryfall", flag=True))
            )
        assert "scryfall" not in result


# ---------------------------------------------------------------------------
# _ingest_format_staples_async
# ---------------------------------------------------------------------------


class TestIngestFormatStaplesAsync:
    def _config(
        self,
        formats: list[str] | None = None,
        base_url: str = "http://example.com/{format}",
        path: str = "data/raw/fs.json",
    ) -> dict:
        return {
            "format_staples": {
                "formats": formats if formats is not None else ["modern"],
                "base_url": base_url,
                "path": path,
            }
        }

    @pytest.mark.asyncio
    async def test_empty_config_returns_empty_dict(self):
        client = AsyncMock(spec=httpx.AsyncClient)
        assert await _ingest_format_staples_async(client, {}) == {}

    @pytest.mark.asyncio
    async def test_missing_base_url_returns_empty_dict(self):
        client = AsyncMock(spec=httpx.AsyncClient)
        assert (
            await _ingest_format_staples_async(
                client, {"format_staples": {"formats": ["modern"]}}
            )
            == {}
        )

    @pytest.mark.asyncio
    async def test_successful_scrape_returns_format_staples_key(self):
        client = AsyncMock(spec=httpx.AsyncClient)
        with (
            patch("src.data.cards.sources.scrapers.download_html_page"),
            patch("pathlib.Path.read_text", return_value="<html></html>"),
            patch(
                "src.data.cards.sources.scrapers.extract_format_staples",
                return_value=[],
            ),
            patch("src.data.cards.sources.scrapers._save_to_json"),
            patch(
                "src.data.cards.sources.scrapers.load_from_json", return_value=([], [])
            ),
        ):
            result = await _ingest_format_staples_async(client, self._config())
        assert "format_staples" in result

    @pytest.mark.asyncio
    async def test_snapshot_date_added_to_each_record(self):
        client = AsyncMock(spec=httpx.AsyncClient)
        fake_rec = {"card_name": "Lightning Bolt", "format": "modern"}
        with (
            patch("src.data.cards.sources.scrapers.download_html_page"),
            patch("pathlib.Path.read_text", return_value="<html></html>"),
            patch(
                "src.data.cards.sources.scrapers.extract_format_staples",
                return_value=[fake_rec],
            ),
            patch("src.data.cards.sources.scrapers._save_to_json") as mock_save,
            patch(
                "src.data.cards.sources.scrapers.load_from_json", return_value=([], [])
            ),
        ):
            await _ingest_format_staples_async(client, self._config())
        saved = mock_save.call_args[0][0]
        assert len(saved) == 1
        assert "snapshot_date" in saved[0]

    @pytest.mark.asyncio
    async def test_download_error_is_skipped_gracefully(self):
        client = AsyncMock(spec=httpx.AsyncClient)
        with (
            patch(
                "src.data.cards.sources.scrapers.download_html_page",
                side_effect=SourceDownloadError("404"),
            ),
            patch("src.data.cards.sources.scrapers._save_to_json"),
            patch(
                "src.data.cards.sources.scrapers.load_from_json", return_value=([], [])
            ),
        ):
            result = await _ingest_format_staples_async(client, self._config())
        assert "format_staples" in result

    @pytest.mark.asyncio
    async def test_records_from_multiple_formats_are_combined(self):
        client = AsyncMock(spec=httpx.AsyncClient)

        def fake_extract(html, fmt):
            return [{"card_name": f"Card_{fmt}", "format": fmt}]

        with (
            patch("src.data.cards.sources.scrapers.download_html_page"),
            patch("pathlib.Path.read_text", return_value="<html></html>"),
            patch(
                "src.data.cards.sources.scrapers.extract_format_staples",
                side_effect=fake_extract,
            ),
            patch("src.data.cards.sources.scrapers._save_to_json") as mock_save,
            patch(
                "src.data.cards.sources.scrapers.load_from_json", return_value=([], [])
            ),
        ):
            await _ingest_format_staples_async(
                client, self._config(formats=["modern", "legacy"])
            )
        saved = mock_save.call_args[0][0]
        assert len(saved) == 2
        formats_in_saved = {r["format"] for r in saved}
        assert formats_in_saved == {"modern", "legacy"}


# ---------------------------------------------------------------------------
# _ingest_tournament_results_async
# ---------------------------------------------------------------------------


class TestIngestTournamentResultsAsync:
    def _config(self, max_n: int = 1) -> dict:
        return {
            "tournament_results": {
                "formats": ["modern"],
                "format_codes": {"modern": "MO"},
                "list_url": "http://example.com/format?f={code}",
                "deck_url_prefix": "http://example.com",
                "max_tournaments_per_format": max_n,
                "path": "data/raw/tr.json",
            }
        }

    def _fake_tournament(self, event_id: str = "1001") -> dict:
        return {
            "_event_id": event_id,
            "tournament_id": f"mtgtop8_{event_id}",
            "tournament_date": "2026-05-18",
            "event_name": "MTGO Challenge",
            "format": "modern",
        }

    def _fake_deck_meta(self, deck_id: str = "5001") -> dict:
        return {
            "deck_id": deck_id,
            "player": "bokk",
            "deck_name": "Dimir Control",
            "placement": 1,
        }

    @pytest.mark.asyncio
    async def test_empty_config_returns_empty_dict(self):
        client = AsyncMock(spec=httpx.AsyncClient)
        assert await _ingest_tournament_results_async(client, {}) == {}

    @pytest.mark.asyncio
    async def test_missing_format_code_skips_format(self):
        client = AsyncMock(spec=httpx.AsyncClient)
        config = {
            "tournament_results": {
                "formats": ["pioneer"],
                "format_codes": {},
                "list_url": "http://example.com/format?f={code}",
                "deck_url_prefix": "http://example.com",
                "path": "data/raw/tr.json",
            }
        }
        with (
            patch("src.data.cards.sources.scrapers._save_to_json"),
            patch(
                "src.data.cards.sources.scrapers.load_from_json", return_value=([], [])
            ),
        ):
            result = await _ingest_tournament_results_async(client, config)
        assert result["tournament_results"] == ([], [])

    @pytest.mark.asyncio
    async def test_successful_scrape_returns_tournament_results_key(self):
        client = AsyncMock(spec=httpx.AsyncClient)
        with (
            patch("src.data.cards.sources.scrapers.download_html_page"),
            patch("pathlib.Path.read_text", return_value="<html></html>"),
            patch(
                "src.data.cards.sources.scrapers.extract_mtgtop8_tournament_list",
                return_value=[self._fake_tournament()],
            ),
            patch(
                "src.data.cards.sources.scrapers.extract_mtgtop8_event_decks",
                return_value=[self._fake_deck_meta()],
            ),
            patch(
                "src.data.cards.sources.scrapers.extract_mtgtop8_decklist",
                return_value=[{"card_name": "Lightning Bolt"}],
            ),
            patch("src.data.cards.sources.scrapers._save_to_json"),
            patch(
                "src.data.cards.sources.scrapers.load_from_json", return_value=([], [])
            ),
        ):
            result = await _ingest_tournament_results_async(client, self._config())
        assert "tournament_results" in result

    @pytest.mark.asyncio
    async def test_max_tournaments_per_format_is_respected(self):
        client = AsyncMock(spec=httpx.AsyncClient)
        tournaments = [self._fake_tournament(str(i)) for i in range(5)]
        download_calls: list[str] = []

        def track_download(_client, url, path):
            download_calls.append(url)

        with (
            patch(
                "src.data.cards.sources.scrapers.download_html_page",
                side_effect=track_download,
            ),
            patch("pathlib.Path.read_text", return_value="<html></html>"),
            patch(
                "src.data.cards.sources.scrapers.extract_mtgtop8_tournament_list",
                return_value=tournaments,
            ),
            patch(
                "src.data.cards.sources.scrapers.extract_mtgtop8_event_decks",
                return_value=[],
            ),
            patch("src.data.cards.sources.scrapers._save_to_json"),
            patch(
                "src.data.cards.sources.scrapers.load_from_json", return_value=([], [])
            ),
        ):
            await _ingest_tournament_results_async(client, self._config(max_n=2))
        # 1 list page + 2 event pages (no deck pages since event_decks returns [])
        assert len(download_calls) == 3

    @pytest.mark.asyncio
    async def test_tournament_list_download_error_skips_format(self):
        client = AsyncMock(spec=httpx.AsyncClient)
        with (
            patch(
                "src.data.cards.sources.scrapers.download_html_page",
                side_effect=SourceDownloadError("500"),
            ),
            patch("src.data.cards.sources.scrapers._save_to_json"),
            patch(
                "src.data.cards.sources.scrapers.load_from_json", return_value=([], [])
            ),
        ):
            result = await _ingest_tournament_results_async(client, self._config())
        assert result["tournament_results"] == ([], [])

    @pytest.mark.asyncio
    async def test_event_page_error_skips_tournament(self):
        client = AsyncMock(spec=httpx.AsyncClient)

        def download_side_effect(_client, url, path):
            if "event" in url:
                raise SourceDownloadError("Event page down")

        with (
            patch(
                "src.data.cards.sources.scrapers.download_html_page",
                side_effect=download_side_effect,
            ),
            patch("pathlib.Path.read_text", return_value="<html></html>"),
            patch(
                "src.data.cards.sources.scrapers.extract_mtgtop8_tournament_list",
                return_value=[self._fake_tournament()],
            ),
            patch("src.data.cards.sources.scrapers._save_to_json") as mock_save,
            patch(
                "src.data.cards.sources.scrapers.load_from_json", return_value=([], [])
            ),
        ):
            await _ingest_tournament_results_async(client, self._config())
        # Error on event page → no card rows saved
        assert mock_save.call_args[0][0] == []


class TestIngestTournamentResultsAsyncRealParsing:
    """Exercises the real extract_mtgtop8_* parsers — only the HTTP call is mocked.

    Unlike TestIngestTournamentResultsAsync above (which mocks every extractor),
    this class lets extract_mtgtop8_tournament_list, extract_mtgtop8_event_decks,
    and extract_mtgtop8_decklist run against real inline HTML fixtures, built in
    the same style as tests/data/cards/sources/test_extractors.py's _html()
    helpers. This is what actually catches a real HTML/selector regression.
    """

    # Level 1: format list page — table.Stable > tr.hover_tr, cols[1] has the
    # event link "event?e={id}&f={code}", cols[-1] the DD/MM/YY date.
    _LIST_HTML = (
        '<html><body><table class="Stable"><tbody>'
        '<tr class="hover_tr"><td></td>'
        '<td><a href="event?e=111&f=MO">Modern Open</a></td>'
        "<td></td><td>01/06/26</td></tr>"
        "</tbody></table></body></html>"
    )

    # Level 2: event page — div.hover_tr with the deck link "...&d={deck_id}",
    # span.G11 for player, span.S14 for deck (archetype) name. Four rows so
    # the ">=4 decks" sanity check in _fetch_event doesn't warn.
    _EVENT_HTML = (
        "<html><body>"
        '<div class="hover_tr"><a href="event?e=111&d=1&f=MO">Player A Deck</a>'
        '<span class="G11">Player A</span><span class="S14">Boros Aggro</span></div>'
        '<div class="hover_tr"><a href="event?e=111&d=2&f=MO">Player B Deck</a>'
        '<span class="G11">Player B</span><span class="S14">Mono Green</span></div>'
        '<div class="hover_tr"><a href="event?e=111&d=3&f=MO">Player C Deck</a>'
        '<span class="G11">Player C</span><span class="S14">Dimir Control</span></div>'
        '<div class="hover_tr"><a href="event?e=111&d=4&f=MO">Player D Deck</a>'
        '<span class="G11">Player D</span><span class="S14">Amulet Titan</span></div>'
        "</body></html>"
    )

    # Level 3: deck page — div.deck_line id="md{ref}" (main deck) / "sb{ref}"
    # (sideboard), text "{copies} {card name}". One main-deck card per deck.
    _DECK_HTML = (
        '<html><body><div id="md1" class="deck_line hover_tr">'
        "4 <span class=L14>Lightning Bolt</span></div></body></html>"
    )

    @pytest.fixture(autouse=True)
    def _patch_downloads(self, monkeypatch, tmp_path):
        """Replace the network boundary only: download_html_page writes an
        inline HTML fixture to output_path instead of making an HTTP call.
        Everything downstream (Path.read_text, the extract_* calls,
        _save_to_json, load_from_json) runs for real.
        """

        async def fake_download(client, url, output_path):
            path_str = str(output_path)
            if "tournament_list_" in path_str:
                html = self._LIST_HTML
            elif "tournament_event_" in path_str:
                html = self._EVENT_HTML
            else:
                html = self._DECK_HTML
            Path(output_path).write_text(html, encoding="utf-8")

        monkeypatch.setattr(
            "src.data.cards.sources.scrapers.download_html_page", fake_download
        )
        monkeypatch.chdir(tmp_path)
        (tmp_path / "data" / "raw").mkdir(parents=True)

    @pytest.mark.asyncio
    async def test_real_extractors_produce_deck_records(self):
        config = {
            "tournament_results": {
                "formats": ["modern"],
                "format_codes": {"modern": "MO"},
                "list_url": "https://www.mtgtop8.com/format?f={code}",
                "deck_url_prefix": "https://www.mtgtop8.com",
                "max_tournaments_per_format": 5,
                "path": "data/raw/tournament_results.json",
            }
        }
        async with httpx.AsyncClient() as client:
            result = await _ingest_tournament_results_async(client, config)

        records, errors = result["tournament_results"]
        assert errors == []
        # 1 tournament -> 4 decks (_EVENT_HTML) -> 1 card row per deck page
        # (_DECK_HTML) => 4 TournamentResult records total.
        assert len(records) == 4

        assert {r.player for r in records} == {
            "Player A",
            "Player B",
            "Player C",
            "Player D",
        }
        assert {r.deck_name for r in records} == {
            "Boros Aggro",
            "Mono Green",
            "Dimir Control",
            "Amulet Titan",
        }
        assert {r.placement for r in records} == {1, 2, 3, 4}
        for r in records:
            assert r.tournament_id == "mtgtop8_111"
            assert r.tournament_date == "2026-06-01"
            assert r.event_name == "Modern Open"
            assert r.format == "modern"
            assert r.card_name == "Lightning Bolt"
            assert r.copies == 4
            assert r.is_sideboard is False


# ---------------------------------------------------------------------------
# _cleanup_html_files
# ---------------------------------------------------------------------------


class TestCleanupHtmlFiles:
    def test_cleanup_html_files_removes_existing_files(self, tmp_path):
        f1 = tmp_path / "a.html"
        f2 = tmp_path / "b.html"
        f1.write_text("x")
        f2.write_text("y")

        _cleanup_html_files([str(f1), str(f2)])

        assert not f1.exists()
        assert not f2.exists()

    def test_cleanup_html_files_ignores_missing_files(self, tmp_path):
        missing = tmp_path / "does_not_exist.html"
        # Must not raise.
        _cleanup_html_files([str(missing)])

    def test_cleanup_html_files_ignores_permission_error(self, tmp_path):
        f1 = tmp_path / "locked.html"
        f1.write_text("x")

        with patch.object(Path, "unlink", side_effect=PermissionError):
            # Must not raise — this is the Windows "file still held" case.
            _cleanup_html_files([str(f1)])
