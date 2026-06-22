from __future__ import annotations

import json

from src.history.store import HistoryStore, export_records_to_markdown
from src.models import ChatHistoryRecord, Citation, ExecutionPlan, QueryPlan, RewrittenQuery, ToolCall, ToolResult


def _record(session_id: str = "session_1", source_id: str = "doc_1") -> ChatHistoryRecord:
    return ChatHistoryRecord(
        session_id=session_id,
        user_query="outline this report",
        rewritten_query=RewrittenQuery(
            original_query="outline this report",
            rewritten_query="Create an outline of the report.",
            confidence=0.9,
        ),
        query_plan=QueryPlan(intent="summarize_document", confidence=0.88),
        selected_source={"selected_source_id": source_id, "source_type": "document", "confidence": 0.87},
        selected_tools=["summarize_tool"],
        execution_plan=ExecutionPlan(tool_calls=[ToolCall(tool_name="summarize_tool", input_payload={"raw_content": "secret document text"})]),
        execution_time_ms=123.45,
        confidence_scores={"planner": 0.88, "final_response": 0.86},
        tool_results=[
            ToolResult(
                success=True,
                tool_name="summarize_tool",
                answer="Revenue improved.",
                data={"retrieved_chunks": [{"content": "raw financial document text"}]},
                table=[{"sensitive_row": 100}],
                confidence=0.86,
                metadata={"gemini_api_key": "do-not-store", "row_count": 1},
            )
        ],
        final_answer="Revenue improved.",
        citations=[Citation(source_id=source_id, filename="report.pdf", page=2, chunk_id="c1", text_snippet="Revenue improved.")],
        warnings=["Partial summary."],
        errors=[],
        document_source_ids=[source_id],
        trace_id="trace_1",
        metadata={"gemini_api_key": "secret", "raw_document": "full confidential filing"},
    )


def test_history_store_saves_and_loads_session_records(tmp_path) -> None:
    store = HistoryStore(history_dir=tmp_path)
    store.save_record(_record())

    records = store.get_session_history("session_1")

    assert len(records) == 1
    assert records[0].user_query == "outline this report"
    assert records[0].selected_tools == ["summarize_tool"]
    assert records[0].execution_time_ms == 123.45
    assert records[0].confidence_scores["planner"] == 0.88


def test_history_store_supports_document_specific_history(tmp_path) -> None:
    store = HistoryStore(history_dir=tmp_path)
    store.save_record(_record(session_id="session_1", source_id="doc_1"))
    store.save_record(_record(session_id="session_2", source_id="doc_2"))

    records = store.get_document_history("doc_2")

    assert len(records) == 1
    assert records[0].document_source_ids == ["doc_2"]


def test_history_export_to_markdown_contains_traceable_details(tmp_path) -> None:
    store = HistoryStore(history_dir=tmp_path)
    record = _record()
    store.save_record(record)

    markdown = store.export_markdown(session_id="session_1")

    assert "# Financial Intelligence Chatbot History" in markdown
    assert "**User:** outline this report" in markdown
    assert "**Intent:** summarize_document" in markdown
    assert "**Selected Tools:** summarize_tool" in markdown
    assert "report.pdf, page 2, chunk c1" in markdown
    assert "Partial summary." in markdown


def test_history_store_sanitizes_raw_documents_and_secrets(tmp_path) -> None:
    store = HistoryStore(history_dir=tmp_path)
    store.save_record(_record())

    raw_text = store.path.read_text(encoding="utf-8")
    payload = json.loads(raw_text.splitlines()[0])

    assert "do-not-store" not in raw_text
    assert "full confidential filing" not in raw_text
    assert "raw financial document text" not in raw_text
    assert payload["metadata"]["gemini_api_key"] == "[REDACTED]"
    assert payload["execution_plan"]["tool_calls"][0]["input_payload"].get("raw_content") is None


def test_export_records_to_markdown_handles_empty_history() -> None:
    markdown = export_records_to_markdown([])

    assert "No chat history yet." in markdown
