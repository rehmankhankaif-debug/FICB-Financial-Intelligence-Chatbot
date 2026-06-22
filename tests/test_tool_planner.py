from __future__ import annotations

from src.agents.tool_planner import ToolPlannerAgent
from src.models.execution import ExecutionPlan
from src.models.query import QueryPlan
from src.models.source import SourceSelection
from src.tools.manager import ToolManager


def _planner() -> ToolPlannerAgent:
    return ToolPlannerAgent(registry=ToolManager().get_registry())


def _source(source_type: str = "table") -> SourceSelection:
    return SourceSelection(
        selected_source_id="source_1",
        source_type=source_type,
        confidence=0.88,
        reason="test source",
    )


def test_table_analysis_query_creates_one_tool_call() -> None:
    plan = QueryPlan(intent="table_analysis", required_source_type="table", confidence=0.86)

    execution_plan = _planner().create_execution_plan(plan, _source("table"))

    assert isinstance(execution_plan, ExecutionPlan)
    assert [call.tool_name for call in execution_plan.tool_calls] == ["table_analysis_tool"]
    assert execution_plan.requires_tool_chain is False


def test_chart_query_creates_table_analysis_and_chart_chain() -> None:
    plan = QueryPlan(intent="chart_request", required_source_type="table", chart_requested=True, chart_type="bar", confidence=0.87)

    execution_plan = _planner().create_execution_plan(plan, _source("table"))

    assert [call.tool_name for call in execution_plan.tool_calls] == ["table_analysis_tool", "chart_tool"]
    assert execution_plan.tool_calls[1].depends_on == ["table_analysis_tool"]
    assert execution_plan.requires_tool_chain is True


def test_summarize_query_creates_summarize_tool() -> None:
    plan = QueryPlan(intent="summarize_document", required_source_type="document", confidence=0.86)

    execution_plan = _planner().create_execution_plan(plan, _source("document"))

    assert [call.tool_name for call in execution_plan.tool_calls] == ["summarize_tool"]


def test_compare_query_creates_table_rag_compare_chain() -> None:
    plan = QueryPlan(intent="compare_documents", required_source_type="mixed", confidence=0.84)

    execution_plan = _planner().create_execution_plan(plan, _source("mixed"))

    assert [call.tool_name for call in execution_plan.tool_calls] == [
        "table_analysis_tool",
        "rag_qa_tool",
        "compare_tool",
    ]
    assert execution_plan.tool_calls[2].depends_on == ["table_analysis_tool", "rag_qa_tool"]


def test_document_only_comparison_does_not_require_a_table_tool() -> None:
    plan = QueryPlan(intent="compare_documents", required_source_type="mixed", confidence=0.9)
    selection = SourceSelection(
        selected_source_id="report-2023",
        selected_source_ids=["report-2023", "report-2022"],
        selected_source_types={"report-2023": "document", "report-2022": "document"},
        source_type="mixed",
        confidence=0.9,
    )

    execution_plan = _planner().create_execution_plan(plan, selection)

    assert [call.tool_name for call in execution_plan.tool_calls] == ["rag_qa_tool", "compare_tool"]
    assert execution_plan.tool_calls[1].depends_on == ["rag_qa_tool"]


def test_general_finance_query_creates_general_finance_tool() -> None:
    plan = QueryPlan(intent="general_finance", confidence=0.81)

    execution_plan = _planner().create_execution_plan(plan, None)

    assert [call.tool_name for call in execution_plan.tool_calls] == ["general_finance_tool"]
    assert execution_plan.selected_sources == []


def test_low_confidence_plan_creates_warning() -> None:
    plan = QueryPlan(intent="general_finance", confidence=0.3)

    execution_plan = _planner().create_execution_plan(plan, None)

    assert execution_plan.warnings
    assert "confidence is low" in execution_plan.warnings[0]


def test_missing_source_is_handled_safely() -> None:
    plan = QueryPlan(intent="table_analysis", required_source_type="table", confidence=0.8)

    execution_plan = _planner().create_execution_plan(plan, None)

    assert execution_plan.tool_calls == []
    assert execution_plan.warnings
    assert "Missing required uploaded source" in execution_plan.warnings[-1]


def test_unknown_intent_is_handled_safely() -> None:
    plan = QueryPlan(intent="unknown_intent", confidence=0.8)

    execution_plan = _planner().create_execution_plan(plan, None)

    assert execution_plan.tool_calls == []
    assert execution_plan.confidence == 0.0
    assert "Unknown intent" in execution_plan.warnings[-1]
