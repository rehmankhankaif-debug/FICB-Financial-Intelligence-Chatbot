from __future__ import annotations

import pytest
import requests

from src.ingestion import url_loader
from src.ingestion.url_loader import load_url
from src.utils.errors import IngestionError


class FakeResponse:
    def __init__(self, text: str, status_code: int = 200) -> None:
        self.text = text
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError("bad status")


def test_load_url_cleans_html(monkeypatch) -> None:
    html = """
    <html>
      <head><title>Market Report</title><style>.x{}</style></head>
      <body>
        <nav>Menu</nav>
        <script>alert(1)</script>
        <main><h1>Revenue Growth</h1><p>Profit improved this quarter.</p></main>
      </body>
    </html>
    """

    def fake_get(url, timeout, headers):
        return FakeResponse(html)

    monkeypatch.setattr(url_loader.requests, "get", fake_get)

    sources = load_url("https://example.com/report", source_id="url_source")

    assert len(sources) == 1
    assert sources[0].source_id == "url_source"
    assert sources[0].source_type == "url"
    assert sources[0].metadata["title"] == "Market Report"
    assert "Revenue Growth" in sources[0].content
    assert "Menu" not in sources[0].content
    assert "alert" not in sources[0].content


def test_load_url_rejects_invalid_url() -> None:
    with pytest.raises(IngestionError):
        load_url("not-a-url")


def test_load_url_handles_timeout(monkeypatch) -> None:
    def fake_get(url, timeout, headers):
        raise requests.Timeout("timed out")

    monkeypatch.setattr(url_loader.requests, "get", fake_get)

    with pytest.raises(IngestionError):
        load_url("https://example.com/report")


def test_load_url_rejects_private_network_url() -> None:
    with pytest.raises(IngestionError):
        load_url("http://127.0.0.1/internal")


def test_load_url_retries_transient_request_failure(monkeypatch) -> None:
    attempts = {"count": 0}
    html = "<html><head><title>Retry Report</title></head><body>Revenue recovered.</body></html>"
    monkeypatch.setattr(url_loader.settings, "url_retry_max_attempts", 2)
    monkeypatch.setattr(url_loader.settings, "retry_backoff_seconds", 0)

    def fake_get(url, timeout, headers):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise requests.ConnectionError("temporary")
        return FakeResponse(html)

    monkeypatch.setattr(url_loader.requests, "get", fake_get)

    sources = load_url("https://example.com/retry", source_id="retry_source", timeout=1)

    assert attempts["count"] == 2
    assert sources[0].source_id == "retry_source"
    assert "Revenue recovered" in sources[0].content
