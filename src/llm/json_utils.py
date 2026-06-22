from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional


def strip_markdown_fences(text: str) -> str:
    clean = (text or "").strip()
    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", clean, flags=re.IGNORECASE | re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()
    return clean


def extract_json_candidate(text: str) -> str:
    clean = strip_markdown_fences(text)
    if not clean:
        return ""
    first_object = clean.find("{")
    last_object = clean.rfind("}")
    if first_object >= 0 and last_object > first_object:
        return clean[first_object : last_object + 1]
    return clean


def lightweight_json_repair(text: str) -> str:
    repaired = extract_json_candidate(text)
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    return repaired


def parse_json_safely(text: str, fallback: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    safe_fallback = dict(fallback or {})
    candidate = extract_json_candidate(text)
    if not candidate:
        return safe_fallback

    for value in (candidate, lightweight_json_repair(candidate)):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else safe_fallback
        except Exception:
            continue
    return safe_fallback
