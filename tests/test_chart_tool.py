from __future__ import annotations

from src.models.query import QueryPlan
from src.tools.chart_tool import ChartTool


def _dump(model):
    return model.model_dump() if hasattr(model, "model_dump") else model.dict()


def test_chart_tool_creates_bar_chart_from_table_result() -> None:
    plan = QueryPlan(intent="chart_request", chart_requested=True, chart_type="bar")
    result = ChartTool().safe_run(
        {
            "query_plan": _dump(plan),
            "dependency_results": {
                "table_analysis_tool": {
                    "success": True,
                    "tool_name": "table_analysis_tool",
                    "table": [
                        {"transmission": "Manual", "count_rows": 3},
                        {"transmission": "Automatic", "count_rows": 2},
                    ],
                }
            },
        }
    )

    assert result.success is True
    assert result.tool_name == "chart_tool"
    assert result.chart is not None
    assert result.data["chart_type"] == "bar"
    assert result.metadata["plotly_spec"]["data"]


def test_chart_tool_uses_second_category_as_grouped_bar_color() -> None:
    plan = QueryPlan(intent="chart_request", chart_requested=True, chart_type="bar")
    result = ChartTool().safe_run(
        {
            "query_plan": _dump(plan),
            "table": [
                {"transmission": "Manual", "fuel_type": "Diesel", "count_rows": 3},
                {"transmission": "Manual", "fuel_type": "Petrol", "count_rows": 2},
                {"transmission": "Automatic", "fuel_type": "Diesel", "count_rows": 1},
            ],
        }
    )

    assert result.success is True
    assert result.data["x"] == "transmission"
    assert result.data["y"] == "count_rows"
    assert result.data["color"] == "fuel_type"
    assert result.metadata["plotly_spec"]["layout"]["barmode"] == "group"
    assert {trace["name"] for trace in result.metadata["plotly_spec"]["data"]} == {"Diesel", "Petrol"}


def test_chart_tool_supports_line_pie_scatter_histogram() -> None:
    table = [{"month": "Jan", "sales": 10}, {"month": "Feb", "sales": 20}]
    for chart_type in ["line", "pie", "scatter", "histogram"]:
        plan = QueryPlan(intent="chart_request", chart_requested=True, chart_type=chart_type)
        result = ChartTool().safe_run({"query_plan": _dump(plan), "table": table})

        assert result.success is True
        assert result.data["chart_type"] == chart_type


def test_chart_tool_missing_table_fails_safely() -> None:
    result = ChartTool().safe_run({})

    assert result.success is False
    assert result.error_msg


def test_chart_tool_generates_bar_and_pie_from_gender_counts() -> None:
    plan = QueryPlan(
        intent="chart_request",
        chart_requested=True,
        chart_type="bar",
        chart_types=["bar", "pie"],
    )
    result = ChartTool().safe_run(
        {
            "query_plan": _dump(plan),
            "table": [{"Gender": "F", "count_rows": 934}, {"Gender": "M", "count_rows": 534}],
        }
    )

    assert result.success is True
    assert result.data["chart_types"] == ["bar", "pie"]
    assert isinstance(result.chart, list)
    assert len(result.chart) == 2
    assert len(result.metadata["plotly_specs"]) == 2
