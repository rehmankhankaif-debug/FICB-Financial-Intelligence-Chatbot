from __future__ import annotations

import json

from src.models import ToolResult
from src.observability.tracing import TraceRecorder, summarize_tool_result
from src.utils.logging import log_error, log_event, log_tool_call


def _json_lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_log_event_redacts_secrets_and_raw_content(tmp_path) -> None:
    log_path = tmp_path / "app.log"

    log_event(
        "test_event",
        {
            "gemini_api_key": "secret-key",
            "raw_document": "confidential filing body",
            "query": "average profit",
        },
        log_path=log_path,
    )

    text = log_path.read_text(encoding="utf-8")
    payload = _json_lines(log_path)[0]["payload"]
    entry = _json_lines(log_path)[0]
    assert "secret-key" not in text
    assert "confidential filing body" not in text
    assert entry["level"] == "INFO"
    assert entry["event_id"]
    assert payload["gemini_api_key"] == "[REDACTED]"
    assert payload["raw_document"] == "[REDACTED_RAW_CONTENT]"
    assert payload["query"] == "average profit"


def test_log_tool_call_and_error_write_json_lines(tmp_path) -> None:
    log_path = tmp_path / "app.log"

    log_tool_call("table_analysis_tool", {"success": True, "confidence": 0.9}, log_path=log_path)
    log_error(RuntimeError("boom"), {"stage": "unit_test", "token": "secret-token"}, log_path=log_path)

    entries = _json_lines(log_path)

    assert entries[0]["event_type"] == "tool_call"
    assert entries[0]["payload"]["tool_name"] == "table_analysis_tool"
    assert entries[1]["event_type"] == "error"
    assert entries[1]["level"] == "ERROR"
    assert entries[1]["payload"]["context"]["token"] == "[REDACTED]"


def test_trace_recorder_records_start_event_tool_and_end(tmp_path) -> None:
    log_path = tmp_path / "trace.log"
    recorder = TraceRecorder(log_path=str(log_path))
    span = recorder.start_trace("query_pipeline", {"raw_content": "do not log"})
    result = ToolResult(success=True, tool_name="general_finance_tool", answer="ok", confidence=0.8)

    recorder.record_event(span.trace_id, "planned", {"intent": "general_finance"})
    recorder.record_tool_call(span.trace_id, "general_finance_tool", result=result, elapsed_ms=3.5)
    elapsed = recorder.end_trace(span, metadata={"final_confidence": 0.8})

    entries = _json_lines(log_path)
    event_types = [entry["event_type"] for entry in entries]
    assert event_types == ["trace_start", "trace_event", "tool_call", "trace_end"]
    assert entries[0]["payload"]["metadata"]["raw_content"] == "[REDACTED_RAW_CONTENT]"
    assert entries[2]["payload"]["result"]["tool_name"] == "general_finance_tool"
    assert entries[3]["payload"]["status"] == "success"
    assert elapsed >= 0.0


def test_trace_recorder_context_manager_logs_errors(tmp_path) -> None:
    log_path = tmp_path / "trace.log"
    recorder = TraceRecorder(log_path=str(log_path))
    caught = False

    try:
        with recorder.span("failing_stage"):
            raise ValueError("bad")
    except ValueError:
        caught = True

    entries = _json_lines(log_path)

    assert caught is True
    assert entries[0]["event_type"] == "trace_start"
    assert entries[1]["event_type"] == "error"
    assert entries[2]["event_type"] == "trace_end"
    assert entries[2]["payload"]["status"] == "error"


def test_summarize_tool_result_omits_table_values() -> None:
    summary = summarize_tool_result(
        ToolResult(
            success=True,
            tool_name="table_analysis_tool",
            table=[{"profit": 100}],
            citations=[],
            confidence=0.9,
        )
    )

    assert summary["has_table"] is True
    assert "profit" not in json.dumps(summary)
