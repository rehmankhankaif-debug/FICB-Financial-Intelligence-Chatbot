from __future__ import annotations

from src.llm.gemini_client import GeminiClient
from src.llm.json_utils import parse_json_safely


class FakeResponse:
    def __init__(self, text: str) -> None:
        self.text = text


class FakeClient:
    def __init__(self, text: str) -> None:
        self.text = text

    def generate_content(self, prompt: str) -> FakeResponse:
        return FakeResponse(self.text)


class FlakyClient:
    def __init__(self) -> None:
        self.calls = 0

    def generate_content(self, prompt: str) -> FakeResponse:
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("temporary")
        return FakeResponse("recovered")


class QuotaClient:
    def __init__(self) -> None:
        self.calls = 0

    def generate_content(self, prompt: str) -> FakeResponse:
        self.calls += 1
        raise RuntimeError("429 RESOURCE_EXHAUSTED: quota exceeded")


def test_missing_api_key_does_not_crash_and_is_unavailable() -> None:
    client = GeminiClient(api_key="", client=None)

    assert client.is_available() is False
    assert client.generate("hello") == ""


def test_generate_json_returns_fallback_if_unavailable() -> None:
    client = GeminiClient(api_key="", client=None)

    payload = client.generate_json("return json", fallback={"safe": True})

    assert payload == {"safe": True}


def test_markdown_fenced_json_is_parsed() -> None:
    client = GeminiClient(api_key="", client=FakeClient('```json\n{"confidence": 0.9, "intent": "table_analysis"}\n```'))

    payload = client.generate_json("return json")

    assert payload["confidence"] == 0.9
    assert payload["intent"] == "table_analysis"


def test_malformed_json_returns_fallback_safely() -> None:
    client = GeminiClient(api_key="", client=FakeClient("{not-json"))

    payload = client.generate_json("return json", fallback={"fallback": "ok"})

    assert payload == {"fallback": "ok"}


def test_parse_json_safely_repairs_trailing_comma() -> None:
    payload = parse_json_safely('{"a": 1,}', fallback={})

    assert payload == {"a": 1}


def test_generate_retries_transient_provider_failure(monkeypatch) -> None:
    flaky = FlakyClient()
    monkeypatch.setattr("src.llm.gemini_client.settings.retry_max_attempts", 2)
    monkeypatch.setattr("src.llm.gemini_client.settings.retry_backoff_seconds", 0)
    client = GeminiClient(api_key="", client=flaky)

    assert client.generate("hello") == "recovered"
    assert flaky.calls == 2


def test_generate_does_not_waste_retries_when_quota_is_exhausted(monkeypatch) -> None:
    quota = QuotaClient()
    monkeypatch.setattr("src.llm.gemini_client.settings.retry_max_attempts", 3)
    monkeypatch.setattr("src.llm.gemini_client.settings.retry_backoff_seconds", 0)
    client = GeminiClient(api_key="", client=quota)

    assert client.generate("hello") == ""
    assert quota.calls == 1
    assert client.last_error_type == "QuotaExceededError"
    assert "quota is exhausted" in client.last_error
