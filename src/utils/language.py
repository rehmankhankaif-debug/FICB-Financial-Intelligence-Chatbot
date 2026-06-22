from __future__ import annotations

import re

try:
    from langdetect import DetectorFactory, LangDetectException, detect

    DetectorFactory.seed = 0
except Exception:
    DetectorFactory = None
    LangDetectException = Exception
    detect = None


HINGLISH_WORDS = {
    "aur",
    "batao",
    "bnao",
    "banao",
    "kitna",
    "kitni",
    "kitne",
    "ke",
    "ka",
    "ki",
    "kya",
    "hai",
    "hain",
    "jeeti",
    "haari",
    "dikhao",
    "karo",
}

LANGUAGE_LABELS = {
    "en": "English",
    "hi": "Hindi",
    "hi-en": "Hinglish",
    "es": "Spanish",
}


def normalize_language_code(lang: str) -> str:
    normalized = (lang or "en").lower().strip()
    if normalized in {"hi-en", "hinglish"}:
        return "hi-en"
    if normalized.startswith("hi"):
        return "hi"
    if normalized.startswith("en"):
        return "en"
    if normalized.startswith("es"):
        return "es"
    return "en"


def _looks_hinglish(text: str) -> bool:
    tokens = set(re.findall(r"[A-Za-z]+", text.lower()))
    return bool(tokens.intersection(HINGLISH_WORDS))


def detect_language(text: str) -> str:
    if not text or not text.strip():
        return "en"

    if _looks_hinglish(text):
        return "hi-en"

    if detect is None:
        return "en"

    try:
        return normalize_language_code(detect(text))
    except LangDetectException:
        return "en"
    except Exception:
        return "en"


def should_respond_in_same_language(user_language: str) -> bool:
    return normalize_language_code(user_language) in {"en", "hi", "hi-en", "es"}


def language_options(codes) -> list:
    options = [("Auto detect", None)]
    seen = set()
    for code in codes or []:
        normalized = normalize_language_code(code)
        if normalized in seen:
            continue
        seen.add(normalized)
        options.append((LANGUAGE_LABELS.get(normalized, normalized), normalized))
    return options


def language_code_for_label(label: str, codes=None):
    for option_label, code in language_options(codes or LANGUAGE_LABELS.keys()):
        if option_label == label:
            return code
    return None
