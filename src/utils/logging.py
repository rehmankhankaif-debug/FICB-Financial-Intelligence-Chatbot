from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Union
from uuid import uuid4

from src.config import LOGS_DIR, settings


SENSITIVE_KEYS = {"api_key", "gemini_api_key", "password", "token", "secret", "authorization"}
RAW_CONTENT_KEYS = {
    "content",
    "contents",
    "raw_content",
    "raw_document",
    "document_text",
    "financial_document",
    "uploaded_file_contents",
    "dataframe",
    "df",
    "retriever",
    "vector_store",
    "embedding_service",
}
MAX_STRING_LENGTH = 500
MAX_LIST_ITEMS = 50
LOG_LEVELS = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40, "CRITICAL": 50}


def _normalize_level(level: str) -> str:
    normalized = str(level or "INFO").upper()
    return normalized if normalized in LOG_LEVELS else "INFO"


def _should_log(level: str) -> bool:
    configured = _normalize_level(settings.log_level)
    current = _normalize_level(level)
    return LOG_LEVELS[current] >= LOG_LEVELS[configured]


def _is_sensitive_key(key: Any) -> bool:
    normalized = str(key or "").lower()
    return any(sensitive in normalized for sensitive in SENSITIVE_KEYS)


def _is_raw_content_key(key: Any) -> bool:
    normalized = str(key or "").lower()
    return normalized in RAW_CONTENT_KEYS or normalized.endswith("_content") or normalized.endswith("_document")


def sanitize_for_logging(value: Any) -> Any:
    return _sanitize_value(value)


def _sanitize_value(value: Any, key: Any = None) -> Any:
    if key is not None:
        if _is_sensitive_key(key):
            return "[REDACTED]"
        if _is_raw_content_key(key):
            return "[REDACTED_RAW_CONTENT]"

    if isinstance(value, dict):
        return {
            str(item_key): _sanitize_value(item_value, item_key)
            for item_key, item_value in value.items()
        }
    if isinstance(value, list):
        return [_sanitize_value(item) for item in value[:MAX_LIST_ITEMS]]
    if isinstance(value, tuple):
        return [_sanitize_value(item) for item in value[:MAX_LIST_ITEMS]]
    if isinstance(value, str):
        if len(value) > MAX_STRING_LENGTH:
            return f"{value[:MAX_STRING_LENGTH]}...[truncated]"
        return value
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return str(value)


def log_event(
    event_type: str,
    payload: Dict[str, Any],
    log_path: Optional[Union[str, Path]] = None,
    level: str = "info",
) -> None:
    try:
        normalized_level = _normalize_level(level)
        if not _should_log(normalized_level):
            return
        target_path = Path(log_path) if log_path is not None else LOGS_DIR / "app.log"
        target_path.parent.mkdir(parents=True, exist_ok=True)
        safe_payload = _sanitize_value(payload or {})
        entry = {
            "event_id": uuid4().hex,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": normalized_level,
            "environment": settings.environment,
            "event_type": event_type,
            "trace_id": safe_payload.get("trace_id") if isinstance(safe_payload, dict) else None,
            "payload": safe_payload,
        }
        with target_path.open("a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(entry, ensure_ascii=True) + "\n")
    except Exception:
        return


def log_error(
    error: Union[Exception, str],
    context: Optional[Dict[str, Any]] = None,
    log_path: Optional[Union[str, Path]] = None,
) -> None:
    payload = {
        "error": str(error),
        "error_type": error.__class__.__name__ if isinstance(error, Exception) else "str",
        "context": context or {},
    }
    log_event("error", payload, log_path=log_path, level="error")


def log_tool_call(
    tool_name: str,
    payload: Dict[str, Any],
    log_path: Optional[Union[str, Path]] = None,
) -> None:
    log_event("tool_call", {"tool_name": tool_name, **(payload or {})}, log_path=log_path, level="info")


def log_query(
    query: str,
    metadata: Optional[Dict[str, Any]] = None,
    log_path: Optional[Union[str, Path]] = None,
) -> None:
    log_event("query", {"query": query, "metadata": metadata or {}}, log_path=log_path, level="info")
