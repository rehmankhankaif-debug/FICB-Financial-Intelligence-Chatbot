from __future__ import annotations

from src.utils.language import detect_language, language_code_for_label, language_options, normalize_language_code, should_respond_in_same_language


def test_empty_text_returns_en() -> None:
    assert detect_language("") == "en"
    assert detect_language("   ") == "en"


def test_malformed_text_returns_en() -> None:
    assert detect_language("...") == "en"


def test_english_text_returns_safe_normalized_language() -> None:
    assert detect_language("Show me the revenue trend") in {"en", "hi-en", "es", "hi"}


def test_hinglish_like_text_does_not_crash() -> None:
    assert detect_language("manual aur automatic cars kitni hain") == "hi-en"


def test_language_normalization() -> None:
    assert normalize_language_code("hi") == "hi"
    assert normalize_language_code("en-US") == "en"
    assert normalize_language_code("es") == "es"
    assert normalize_language_code("unknown") == "en"


def test_should_respond_in_same_language() -> None:
    assert should_respond_in_same_language("hi-en") is True
    assert should_respond_in_same_language("unknown") is True


def test_configurable_language_options_include_spanish() -> None:
    options = language_options(["en", "hi-en", "es"])

    assert ("Spanish", "es") in options
    assert language_code_for_label("Spanish", ["en", "hi-en", "es"]) == "es"
    assert language_code_for_label("Auto detect", ["en", "es"]) is None
