from __future__ import annotations

from src.agents.validator_agent import ValidatorAgent
from src.models.citation import Citation
from src.models.execution import ExecutionPlan, ToolCall
from src.models.query import QueryPlan
from src.models.source import SourceSelection
from src.models.tool import ToolResult


def _source(source_type: str = "table") -> SourceSelection:
    return SourceSelection(
        selected_source_id="source_1",
        source_type=source_type,
        confidence=0.9,
        reason="test source",
    )


def test_validator_accepts_valid_table_result() -> None:
    query_plan = QueryPlan(intent="table_analysis", confidence=0.9)
    execution_plan = ExecutionPlan(
        query_plan=query_plan,
        tool_calls=[ToolCall(tool_name="table_analysis_tool")],
        confidence=0.88,
    )
    result = ToolResult(
        success=True,
        tool_name="table_analysis_tool",
        answer="sum of profit: 600",
        table=[{"sum_profit": 600.0}],
        data={"value": 600.0},
        confidence=0.95,
    )

    validation = ValidatorAgent().validate(query_plan, execution_plan, [result], _source("table"))

    assert validation.is_valid is True
    assert validation.issues == []
    assert validation.requires_retry is False


def test_validator_rejects_empty_table_result() -> None:
    query_plan = QueryPlan(intent="table_analysis", confidence=0.9)
    execution_plan = ExecutionPlan(
        query_plan=query_plan,
        tool_calls=[ToolCall(tool_name="table_analysis_tool")],
        confidence=0.88,
    )
    result = ToolResult(
        success=True,
        tool_name="table_analysis_tool",
        answer="No rows.",
        table=[],
        data={},
        confidence=0.8,
    )

    validation = ValidatorAgent().validate(query_plan, execution_plan, [result], _source("table"))

    assert validation.is_valid is False
    assert "Table answer has no pandas-grounded data." in validation.issues
    assert validation.requires_retry is True


def test_validator_rejects_missing_document_citation() -> None:
    query_plan = QueryPlan(intent="rag_question", confidence=0.85)
    execution_plan = ExecutionPlan(
        query_plan=query_plan,
        tool_calls=[ToolCall(tool_name="rag_qa_tool")],
        confidence=0.84,
    )
    result = ToolResult(
        success=True,
        tool_name="rag_qa_tool",
        answer="Revenue increased according to the report.",
        data={"retrieved_chunks": [{"chunk_id": "c1"}], "answer_found": True},
        citations=[],
        confidence=0.82,
    )

    validation = ValidatorAgent().validate(query_plan, execution_plan, [result], _source("document"))

    assert validation.is_valid is False
    assert "Document answer is missing citations." in validation.issues


def test_validator_low_confidence_requests_clarification() -> None:
    query_plan = QueryPlan(intent="general_finance", confidence=0.4)
    execution_plan = ExecutionPlan(query_plan=query_plan, confidence=0.4)
    result = ToolResult(
        success=True,
        tool_name="general_finance_tool",
        answer="A general answer.",
        confidence=0.4,
    )

    validation = ValidatorAgent().validate(query_plan, execution_plan, [result], None)

    assert validation.clarification_needed is True
    assert validation.clarification_question
    assert validation.warnings


def test_validator_rejects_failed_tool_result() -> None:
    query_plan = QueryPlan(intent="table_analysis", confidence=0.8)
    execution_plan = ExecutionPlan(
        query_plan=query_plan,
        tool_calls=[ToolCall(tool_name="table_analysis_tool")],
        confidence=0.8,
    )
    result = ToolResult(
        success=False,
        tool_name="table_analysis_tool",
        error_msg="Aggregation column does not exist.",
        confidence=0.0,
    )

    validation = ValidatorAgent().validate(query_plan, execution_plan, [result], _source("table"))

    assert validation.is_valid is False
    assert validation.requires_retry is True
    assert any("failed" in issue for issue in validation.issues)


def test_validator_allows_partial_chart_answer_with_warning() -> None:
    query_plan = QueryPlan(intent="chart_request", chart_requested=True, confidence=0.88)
    execution_plan = ExecutionPlan(
        query_plan=query_plan,
        tool_calls=[
            ToolCall(tool_name="table_analysis_tool"),
            ToolCall(tool_name="chart_tool", depends_on=["table_analysis_tool"]),
        ],
        confidence=0.85,
    )
    table_result = ToolResult(
        success=True,
        tool_name="table_analysis_tool",
        answer="Grouped result contains 2 rows.",
        table=[{"transmission": "Manual", "count_rows": 3}],
        data={"rows": [{"transmission": "Manual", "count_rows": 3}]},
        confidence=0.9,
    )
    chart_result = ToolResult(
        success=False,
        tool_name="chart_tool",
        error_msg="Chart rendering failed.",
        confidence=0.0,
    )

    validation = ValidatorAgent().validate(query_plan, execution_plan, [table_result, chart_result], _source("table"))

    assert validation.is_valid is True
    assert validation.requires_retry is False
    assert any("Chart tool failed" in warning for warning in validation.warnings)


def test_validator_preserves_clarification_required_from_query_plan() -> None:
    query_plan = QueryPlan(
        intent="table_analysis",
        confidence=0.7,
        clarification_needed=True,
        clarification_question="Which file should I use?",
    )
    execution_plan = ExecutionPlan(
        query_plan=query_plan,
        tool_calls=[ToolCall(tool_name="table_analysis_tool")],
        confidence=0.7,
    )
    result = ToolResult(
        success=True,
        tool_name="table_analysis_tool",
        table=[{"profit": 100}],
        data={"value": 100},
        confidence=0.8,
    )

    validation = ValidatorAgent().validate(query_plan, execution_plan, [result], _source("table"))

    assert validation.clarification_needed is True
    assert validation.clarification_question == "Which file should I use?"
    assert any("requested clarification" in warning for warning in validation.warnings)


def test_validator_accepts_document_result_with_citation() -> None:
    query_plan = QueryPlan(intent="summarize_document", confidence=0.86)
    execution_plan = ExecutionPlan(
        query_plan=query_plan,
        tool_calls=[ToolCall(tool_name="summarize_tool")],
        confidence=0.86,
    )
    result = ToolResult(
        success=True,
        tool_name="summarize_tool",
        answer="1. Revenue improved.",
        citations=[Citation(source_id="s1", filename="report.pdf", chunk_id="c1", text_snippet="Revenue improved.")],
        confidence=0.86,
    )

    validation = ValidatorAgent().validate(query_plan, execution_plan, [result], _source("document"))

    assert validation.is_valid is True
    assert validation.issues == []
