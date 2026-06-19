"""Unit tests for src/data/cards/sources/http.py."""

import json
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from src.data.cards.sources import (
    SourceDownloadError,
    _is_retryable_http_error,
    download_html_page,
    download_json_from_url,
)


def _http_error(status_code: int) -> httpx.HTTPStatusError:
    request = MagicMock(spec=httpx.Request)
    response = MagicMock(spec=httpx.Response)
    response.status_code = status_code
    return httpx.HTTPStatusError("error", request=request, response=response)


class TestIsRetryableHttpError:
    def test_429_is_retryable(self):
        assert _is_retryable_http_error(_http_error(429)) is True

    def test_503_is_retryable(self):
        assert _is_retryable_http_error(_http_error(503)) is True

    def test_500_is_retryable(self):
        assert _is_retryable_http_error(_http_error(500)) is True

    def test_404_is_not_retryable(self):
        assert _is_retryable_http_error(_http_error(404)) is False

    def test_401_is_not_retryable(self):
        assert _is_retryable_http_error(_http_error(401)) is False

    def test_non_http_error_is_not_retryable(self):
        assert _is_retryable_http_error(ValueError("oops")) is False


class TestDownloadJsonFromUrl:
    @pytest.mark.asyncio
    async def test_successful_download_writes_json_file(self, tmp_path):
        out = tmp_path / "out.json"
        mock_response = MagicMock()
        mock_response.json.return_value = {"key": "value"}
        mock_response.raise_for_status.return_value = None
        client = AsyncMock(spec=httpx.AsyncClient)
        client.get.return_value = mock_response

        await download_json_from_url(client, "http://example.com/data.json", str(out))

        assert out.exists()
        assert json.loads(out.read_text()) == {"key": "value"}

    @pytest.mark.asyncio
    async def test_http_error_raises_source_download_error(self, tmp_path):
        out = tmp_path / "out.json"
        client = AsyncMock(spec=httpx.AsyncClient)
        client.get.return_value = MagicMock(
            raise_for_status=MagicMock(side_effect=_http_error(404))
        )

        with pytest.raises(SourceDownloadError, match="HTTP error"):
            await download_json_from_url(
                client, "http://example.com/data.json", str(out)
            )


class TestDownloadRetry:
    def _ok_json(self, data: dict) -> MagicMock:
        r = MagicMock()
        r.raise_for_status.return_value = None
        r.json.return_value = data
        return r

    def _ok_html(self, text: str) -> MagicMock:
        r = MagicMock()
        r.raise_for_status.return_value = None
        r.text = text
        return r

    def _err(self, status_code: int) -> MagicMock:
        r = MagicMock()
        r.raise_for_status.side_effect = _http_error(status_code)
        return r

    @pytest.mark.asyncio
    async def test_retries_on_429_then_succeeds(self, tmp_path):
        out = tmp_path / "out.json"
        client = AsyncMock(spec=httpx.AsyncClient)
        client.get.side_effect = [self._err(429), self._ok_json({"k": 1})]

        await download_json_from_url(client, "http://example.com/data.json", str(out))

        assert json.loads(out.read_text()) == {"k": 1}

    @pytest.mark.asyncio
    async def test_permanent_404_raises_immediately(self, tmp_path):
        out = tmp_path / "out.json"
        client = AsyncMock(spec=httpx.AsyncClient)
        client.get.return_value = self._err(404)

        with pytest.raises(SourceDownloadError):
            await download_json_from_url(
                client, "http://example.com/data.json", str(out)
            )
        assert client.get.call_count == 1

    @pytest.mark.asyncio
    async def test_html_retries_on_503_then_succeeds(self, tmp_path):
        out = tmp_path / "out.html"
        client = AsyncMock(spec=httpx.AsyncClient)
        client.get.side_effect = [self._err(503), self._ok_html("<html>ok</html>")]

        await download_html_page(client, "http://example.com/page.html", str(out))

        assert out.read_text() == "<html>ok</html>"

    @pytest.mark.asyncio
    async def test_html_permanent_error_raises_immediately(self, tmp_path):
        out = tmp_path / "out.html"
        client = AsyncMock(spec=httpx.AsyncClient)
        client.get.return_value = self._err(404)

        with pytest.raises(SourceDownloadError):
            await download_html_page(client, "http://example.com/page.html", str(out))
        assert client.get.call_count == 1
