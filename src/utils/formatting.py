from __future__ import annotations


def format_confidence(confidence: float) -> str:
    bounded = max(0.0, min(1.0, confidence))
    return f"{bounded:.0%}"


def safe_truncate(text: str, limit: int = 500) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}...[truncated]"
