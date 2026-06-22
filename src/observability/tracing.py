from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional
from uuid import uuid4

from src.models.tool import ToolResult
from src.utils.logging import log_error, log_event, log_tool_call, sanitize_for_logging


def _now_ms() -> float:
    return time.perf_counter() * 1000.0


def _dump_model(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    return {"value": str(value)}


def summarize_tool_result(result: ToolResult) -> Dict[str, Any]:
    payload = _dump_model(result)
    return {
        "tool_name": payload.get("tool_name"),
        "success": bool(payload.get("success")),
        "confidence": float(payload.get("confidence") or 0.0),
        "warning_count": len(payload.get("warnings") or []),
        "error_msg": payload.get("error_msg"),
        "has_table": payload.get("table") is not None,
        "has_chart": payload.get("chart") is not None,
        "citation_count": len(payload.get("citations") or []),
        "metadata": sanitize_for_logging(payload.get("metadata") or {}),
    }


@dataclass
class TraceSpan:
    trace_id: str
    name: str
    start_ms: float
    metadata: Dict[str, Any]

    def elapsed_ms(self) -> float:
        return max(0.0, _now_ms() - self.start_ms)


class TraceRecorder:
    def __init__(self, log_path: Optional[str] = None) -> None:
        self.log_path = log_path

    def start_trace(self, name: str, metadata: Optional[Dict[str, Any]] = None, trace_id: Optional[str] = None) -> TraceSpan:
        span = TraceSpan(
            trace_id=trace_id or uuid4().hex,
            name=name,
            start_ms=_now_ms(),
            metadata=sanitize_for_logging(metadata or {}),
        )
        log_event(
            "trace_start",
            {"trace_id": span.trace_id, "name": name, "metadata": span.metadata},
            log_path=self.log_path,
        )
        return span

    def record_event(self, trace_id: str, event_type: str, payload: Optional[Dict[str, Any]] = None) -> None:
        log_event(
            "trace_event",
            {"trace_id": trace_id, "event_type": event_type, "payload": sanitize_for_logging(payload or {})},
            log_path=self.log_path,
        )

    def record_tool_call(
        self,
        trace_id: str,
        tool_name: str,
        result: Optional[ToolResult] = None,
        elapsed_ms: Optional[float] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        safe_payload = {
            "trace_id": trace_id,
            "elapsed_ms": elapsed_ms,
            "payload": sanitize_for_logging(payload or {}),
        }
        if result is not None:
            safe_payload["result"] = summarize_tool_result(result)
        log_tool_call(tool_name, safe_payload, log_path=self.log_path)

    def record_error(self, trace_id: str, error: Exception | str, context: Optional[Dict[str, Any]] = None) -> None:
        log_error(error, {"trace_id": trace_id, **(context or {})}, log_path=self.log_path)

    def end_trace(
        self,
        span: TraceSpan,
        status: str = "success",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> float:
        elapsed_ms = span.elapsed_ms()
        log_event(
            "trace_end",
            {
                "trace_id": span.trace_id,
                "name": span.name,
                "status": status,
                "elapsed_ms": elapsed_ms,
                "metadata": sanitize_for_logging(metadata or {}),
            },
            log_path=self.log_path,
        )
        return elapsed_ms

    @contextmanager
    def span(self, name: str, metadata: Optional[Dict[str, Any]] = None) -> Iterator[TraceSpan]:
        span = self.start_trace(name, metadata)
        try:
            yield span
        except Exception as exc:
            self.record_error(span.trace_id, exc, {"span": name})
            self.end_trace(span, status="error")
            raise
        else:
            self.end_trace(span, status="success")
