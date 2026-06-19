"""Unit tests for src/data/cards/sources/{pipeline,scrapers,registry}.py."""

import json
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
