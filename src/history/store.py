from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from src.config import HISTORY_DIR
from src.models.history import ChatHistoryRecord
from src.models.tool import ToolResult
from src.utils.logging import sanitize_for_logging


HISTORY_FILENAME = "chat_history.jsonl"
MAX_ANSWER_LENGTH = 4000
MAX_SNIPPET_LENGTH = 500


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


def _safe_text(value: Any, limit: int = MAX_SNIPPET_LENGTH) -> str:
    text = str(value or "")
    if len(text) > limit:
        return text[:limit] + "...[truncated]"
    return text


def _tool_result_summary(result: Any) -> Dict[str, Any]:
    payload = _dump_model(result)
    table = payload.get("table")
    citations = payload.get("citations") or []
    return {
        "success": bool(payload.get("success")),
        "tool_name": payload.get("tool_name", ""),
        "answer": _safe_text(payload.get("answer"), limit=1000),
        "confidence": float(payload.get("confidence") or 0.0),
        "warnings": list(payload.get("warnings") or []),
        "error_msg": payload.get("error_msg"),
        "metadata": sanitize_for_logging(payload.get("metadata") or {}),
        "table_summary": _table_summary(table),
        "chart_available": payload.get("chart") is not None,
        "citation_count": len(citations),
    }


def _table_summary(table: Any) -> Dict[str, Any]:
    if isinstance(table, list):
        columns = sorted({str(key) for row in table[:3] if isinstance(row, dict) for key in row.keys()})
        return {"row_count": len(table), "columns": columns}
    if isinstance(table, dict):
        return {"row_count": 1, "columns": sorted(str(key) for key in table.keys())}
    if table is not None:
        return {"row_count": None, "columns": [], "type": type(table).__name__}
    return {"row_count": 0, "columns": []}


def _citation_payloads(record: ChatHistoryRecord) -> List[Dict[str, Any]]:
    citations = []
    for citation in record.citations or []:
        payload = _dump_model(citation)
        payload["text_snippet"] = _safe_text(payload.get("text_snippet"), limit=MAX_SNIPPET_LENGTH)
        citations.append(payload)
    return citations


def _document_ids(record: ChatHistoryRecord, payload: Dict[str, Any]) -> List[str]:
    ids = set(record.document_source_ids or [])
    selected_source = payload.get("selected_source") or {}
    if selected_source.get("selected_source_id"):
        ids.add(str(selected_source["selected_source_id"]))
    for citation in payload.get("citations") or []:
        if citation.get("source_id"):
            ids.add(str(citation["source_id"]))
    return sorted(ids)


def sanitize_history_record(record: ChatHistoryRecord) -> Dict[str, Any]:
    payload = _dump_model(record)
    payload["final_answer"] = _safe_text(payload.get("final_answer"), limit=MAX_ANSWER_LENGTH)
    payload["selected_source"] = sanitize_for_logging(payload.get("selected_source") or {})
    payload["execution_plan"] = _execution_plan_summary(payload.get("execution_plan") or {})
    payload["tool_results"] = [_tool_result_summary(result) for result in record.tool_results]
    payload["citations"] = _citation_payloads(record)
    payload["warnings"] = list(record.warnings or [])
    payload["errors"] = sanitize_for_logging(record.errors or [])
    payload["metadata"] = sanitize_for_logging(record.metadata or {})
    payload["document_source_ids"] = _document_ids(record, payload)
    return sanitize_for_logging(payload)


def _execution_plan_summary(payload: Dict[str, Any]) -> Dict[str, Any]:
    tool_calls = []
    for tool_call in payload.get("tool_calls") or []:
        call_payload = dict(tool_call or {})
        input_payload = call_payload.get("input_payload") or {}
        call_payload["input_payload"] = {
            "source_id": input_payload.get("source_id"),
            "source_selection": sanitize_for_logging(input_payload.get("source_selection") or {}),
            "metadata_filter": sanitize_for_logging(input_payload.get("metadata_filter") or {}),
            "top_k": input_payload.get("top_k"),
        }
        tool_calls.append(sanitize_for_logging(call_payload))
    return {
        "selected_sources": sanitize_for_logging(payload.get("selected_sources") or []),
        "tool_calls": tool_calls,
        "requires_tool_chain": bool(payload.get("requires_tool_chain")),
        "confidence": float(payload.get("confidence") or 0.0),
        "warnings": list(payload.get("warnings") or []),
    }


def record_from_payload(payload: Dict[str, Any]) -> ChatHistoryRecord:
    safe_payload = dict(payload or {})
    tool_results = []
    for item in safe_payload.get("tool_results") or []:
        if isinstance(item, ToolResult):
            tool_results.append(item)
        elif isinstance(item, dict):
            tool_results.append(
                ToolResult(
                    success=bool(item.get("success")),
                    tool_name=str(item.get("tool_name") or ""),
                    answer=item.get("answer"),
                    confidence=float(item.get("confidence") or 0.0),
                    warnings=list(item.get("warnings") or []),
                    error_msg=item.get("error_msg"),
                    metadata=item.get("metadata") or {},
                )
            )
    safe_payload["tool_results"] = tool_results
    return ChatHistoryRecord(**safe_payload)


def export_records_to_markdown(records: Iterable[ChatHistoryRecord]) -> str:
    lines = ["# Financial Intelligence Chatbot History", ""]
    count = 0
    for count, record in enumerate(records or [], start=1):
        lines.extend(
            [
                "## Turn {0}".format(count),
                "",
                "**Timestamp:** {0}".format(record.timestamp.isoformat()),
                "",
                "**User:** {0}".format(record.user_query),
                "",
                "**Rewritten Query:** {0}".format(record.rewritten_query.rewritten_query),
                "",
                "**Intent:** {0}".format(record.query_plan.intent),
                "",
                "**Selected Tools:** {0}".format(", ".join(record.selected_tools) or "none"),
                "",
                "**Execution Time:** {0:.2f} ms".format(record.execution_time_ms),
                "",
                "**Confidence:** {0}".format(record.confidence_scores),
                "",
                "**Assistant:** {0}".format(record.final_answer),
                "",
            ]
        )
        if record.citations:
            lines.append("**Citations:**")
            for citation in record.citations:
                payload = _dump_model(citation)
                label = payload.get("filename") or payload.get("source_id") or "source"
                page = payload.get("page")
                chunk_id = payload.get("chunk_id")
                if page is not None:
                    label = "{0}, page {1}".format(label, page)
                if chunk_id:
                    label = "{0}, chunk {1}".format(label, chunk_id)
                lines.append("- {0}: {1}".format(label, _safe_text(payload.get("text_snippet"))))
            lines.append("")
        if record.warnings:
            lines.append("**Warnings:**")
            for warning in record.warnings:
                lines.append("- {0}".format(warning))
            lines.append("")
        if record.errors:
            lines.append("**Errors:**")
            for error in record.errors:
                lines.append("- {0}".format(error))
            lines.append("")
    if count == 0:
        lines.append("No chat history yet.")
    return "\n".join(lines).strip() + "\n"


class HistoryStore:
    def __init__(self, history_dir: Optional[Path] = None, filename: str = HISTORY_FILENAME) -> None:
        self.history_dir = Path(history_dir or HISTORY_DIR)
        self.history_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.history_dir / filename

    def append(self, record: ChatHistoryRecord) -> None:
        try:
            payload = sanitize_history_record(record)
            with self.path.open("a", encoding="utf-8") as history_file:
                history_file.write(json.dumps(payload, ensure_ascii=True, default=str) + "\n")
        except Exception:
            return

    def save_record(self, record: ChatHistoryRecord) -> None:
        self.append(record)

    def load_records(
        self,
        session_id: Optional[str] = None,
        source_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[ChatHistoryRecord]:
        records: List[ChatHistoryRecord] = []
        if not self.path.exists():
            return records
        try:
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                payload = json.loads(line)
                record = record_from_payload(payload)
                if session_id and record.session_id != session_id:
                    continue
                if source_id and source_id not in set(record.document_source_ids):
                    continue
                records.append(record)
        except Exception:
            return records
        if limit is not None:
            return records[-max(0, int(limit)) :]
        return records

    def get_session_history(self, session_id: str, limit: Optional[int] = None) -> List[ChatHistoryRecord]:
        return self.load_records(session_id=session_id, limit=limit)

    def get_document_history(self, source_id: str, limit: Optional[int] = None) -> List[ChatHistoryRecord]:
        return self.load_records(source_id=source_id, limit=limit)

    def export_markdown(
        self,
        session_id: Optional[str] = None,
        source_id: Optional[str] = None,
    ) -> str:
        return export_records_to_markdown(self.load_records(session_id=session_id, source_id=source_id))
