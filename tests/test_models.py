from __future__ import annotations

from src.models import QueryPlan, SourceSelection, TableProfile, ToolResult


def test_tool_result_can_be_created() -> None:
    result = ToolResult(success=True, tool_name="dummy", data={"value": 1})
    assert result.success is True
    assert result.tool_name == "dummy"


def test_query_plan_can_be_created() -> None:
    plan = QueryPlan(
        original_query="average monthly profit",
        rewritten_query="Calculate the average monthly profit.",
        language="en",
        intent="table_analysis",
    )
    assert plan.intent == "table_analysis"


def test_table_profile_can_be_created() -> None:
    profile = TableProfile(
        source_id="src_1",
        filename="sales.csv",
        shape=(10, 3),
        columns=["month", "profit", "region"],
    )
    assert profile.shape == (10, 3)
    assert profile.columns == ["month", "profit", "region"]


def test_source_selection_can_be_created() -> None:
    selection = SourceSelection(
        selected_source_id="src_1",
        source_type="csv",
        confidence=0.9,
        reason="Best schema match.",
    )
    assert selection.selected_source_id == "src_1"


def test_default_list_and_dict_fields_are_not_shared() -> None:
    first = ToolResult()
    second = ToolResult()
    first.warnings.append("warning")
    first.data["value"] = 1
    assert second.warnings == []
    assert second.data == {}
