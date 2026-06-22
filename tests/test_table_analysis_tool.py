from __future__ import annotations

import pandas as pd

from src.models.query import QueryPlan
from src.table_intelligence.profiler import TableProfiler
from src.tools.table_analysis_tool import TableAnalysisTool


def _dump(model):
    return model.model_dump() if hasattr(model, "model_dump") else model.dict()


def _payload(plan: QueryPlan, dataframe: pd.DataFrame):
    return {
        "query_plan": _dump(plan),
        "dataframe": dataframe,
        "table_profile": _dump(TableProfiler().profile(dataframe)),
    }


def test_table_analysis_average_monthly_profit() -> None:
    dataframe = pd.DataFrame(
        {
            "month": ["Jan", "Jan", "Feb"],
            "profit": [100, 300, 500],
        }
    )
    plan = QueryPlan(
        intent="table_analysis",
        metrics=[{"name": "profit"}],
        aggregations=[{"operation": "mean"}],
        grouping=["month"],
        confidence=0.9,
    )

    result = TableAnalysisTool().safe_run(_payload(plan, dataframe))

    assert result.success is True
    assert result.tool_name == "table_analysis_tool"
    rows = {row["month"]: row["mean_profit"] for row in result.table}
    assert rows == {"Jan": 200.0, "Feb": 500.0}


def test_table_analysis_top_5_products_by_sales() -> None:
    dataframe = pd.DataFrame(
        {
            "product": ["A", "B", "C", "D", "E", "F"],
            "sales": [10, 50, 20, 70, 5, 40],
        }
    )
    plan = QueryPlan(
        intent="table_analysis",
        metrics=[{"name": "sales"}],
        aggregations=[{"operation": "sum"}],
        grouping=["product"],
        limit=5,
        confidence=0.9,
    )

    result = TableAnalysisTool().safe_run(_payload(plan, dataframe))

    assert result.success is True
    assert len(result.table) == 5
    assert result.table[0]["product"] == "D"
    assert result.table[0]["sum_sales"] == 70


def test_table_analysis_count_manual_vs_automatic_cars() -> None:
    dataframe = pd.DataFrame({"transmission": ["Manual", "Automatic", "Manual", "Automatic", "Manual"]})
    plan = QueryPlan(
        intent="chart_request",
        metrics=[{"name": "count"}],
        aggregations=[{"operation": "count"}],
        grouping=["transmission"],
        chart_requested=True,
        chart_type="bar",
        confidence=0.9,
    )

    result = TableAnalysisTool().safe_run(_payload(plan, dataframe))

    assert result.success is True
    rows = {row["transmission"]: row["count_rows"] for row in result.table}
    assert rows == {"Manual": 3, "Automatic": 2}


def test_table_analysis_manual_automatic_entities_use_in_filter() -> None:
    dataframe = pd.DataFrame(
        {
            "transmission": ["Manual", "Automatic", "Manual", "Automatic", "Manual", "CVT"],
        }
    )
    plan = QueryPlan(
        original_query="transmission type number of automatic and manual bar graph",
        intent="chart_request",
        entities=[
            {"text": "manual", "normalized": "Manual", "type": "category_value"},
            {"text": "automatic", "normalized": "Automatic", "type": "category_value"},
        ],
        metrics=[{"name": "count"}],
        aggregations=[{"operation": "count"}],
        grouping=["transmission"],
        chart_requested=True,
        chart_type="bar",
        confidence=0.9,
    )

    result = TableAnalysisTool().safe_run(_payload(plan, dataframe))

    assert result.success is True
    rows = {row["transmission"]: row["count_rows"] for row in result.table}
    assert rows == {"Manual": 3, "Automatic": 2}
    assert result.metadata["operation"]["filters"] == [
        {"column": "transmission", "operator": "in", "value": ["Manual", "Automatic"]}
    ]


def test_table_analysis_fuel_type_quantity_group_count() -> None:
    dataframe = pd.DataFrame(
        {
            "fuel_type": ["Diesel", "Petrol", "Petrol", "CNG", "Diesel"],
        }
    )
    plan = QueryPlan(
        original_query="Give me fuel type quantity data and bar graph",
        intent="chart_request",
        metrics=[{"name": "count"}],
        aggregations=[{"operation": "count"}],
        grouping=["fuel_type"],
        chart_requested=True,
        chart_type="bar",
        confidence=0.9,
    )

    result = TableAnalysisTool().safe_run(_payload(plan, dataframe))

    assert result.success is True
    rows = {row["fuel_type"]: row["count_rows"] for row in result.table}
    assert rows == {"Diesel": 2, "Petrol": 2, "CNG": 1}


def test_table_analysis_transmission_fuel_cross_tab_keeps_transmission_and_expands_etc_fuel() -> None:
    dataframe = pd.DataFrame(
        {
            "transmission": ["Manual", "Manual", "Manual", "Automatic", "Automatic", "Automatic", "CVT"],
            "fuel_type": ["Diesel", "Petrol", "CNG", "Diesel", "Petrol", "Electric", "Petrol"],
            "model": ["A", "B", "C", "D", "E", "F", "G"],
            "title": ["A Diesel", "B Petrol", "C CNG", "D Diesel", "E Petrol", "F Electric", "G Petrol"],
        }
    )
    plan = QueryPlan(
        original_query="Manual transmission mei petrol diesel cng and automatic mei petrol disel cng etc kitni h, graph do",
        intent="chart_request",
        entities=[
            {"text": "manual", "normalized": "Manual", "type": "category_value", "field": "transmission"},
            {"text": "automatic", "normalized": "Automatic", "type": "category_value", "field": "transmission"},
            {"text": "petrol", "normalized": "Petrol", "type": "category_value", "field": "fuel_type"},
            {"text": "diesel", "normalized": "Diesel", "type": "category_value", "field": "fuel_type"},
            {"text": "cng", "normalized": "CNG", "type": "category_value", "field": "fuel_type"},
        ],
        metrics=[{"name": "count"}],
        aggregations=[{"operation": "count"}],
        grouping=["transmission", "fuel_type"],
        chart_requested=True,
        chart_type="bar",
        confidence=0.9,
    )

    result = TableAnalysisTool().safe_run(_payload(plan, dataframe))

    assert result.success is True
    rows = {(row["transmission"], row["fuel_type"]): row["count_rows"] for row in result.table}
    assert rows == {
        ("Manual", "Diesel"): 1,
        ("Manual", "Petrol"): 1,
        ("Manual", "CNG"): 1,
        ("Automatic", "Diesel"): 1,
        ("Automatic", "Petrol"): 1,
        ("Automatic", "Electric"): 1,
    }
    assert result.metadata["operation"]["filters"] == [
        {"column": "transmission", "operator": "in", "value": ["Manual", "Automatic"]}
    ]


def test_table_analysis_percentage_share_among_filtered_entities() -> None:
    dataframe = pd.DataFrame(
        {
            "fuel_type": ["Diesel", "Diesel", "Diesel", "Petrol"],
            "transmission": ["Manual", "Automatic", "Automatic", "Automatic"],
        }
    )
    plan = QueryPlan(
        original_query="Among diesel vehicles, what percentage are automatic?",
        intent="table_analysis",
        entities=[
            {"text": "automatic", "normalized": "Automatic", "type": "category_value", "field": "transmission"},
            {"text": "diesel", "normalized": "Diesel", "type": "category_value", "field": "fuel_type"},
        ],
        confidence=0.9,
    )

    result = TableAnalysisTool().safe_run(_payload(plan, dataframe))

    assert result.success is True
    assert result.table[0]["base_column"] == "fuel_type"
    assert result.table[0]["target_column"] == "transmission"
    assert result.table[0]["numerator_count"] == 2
    assert result.table[0]["denominator_count"] == 3
    assert round(result.table[0]["percentage"], 2) == 66.67


def test_table_analysis_entity_comparison_uses_semantic_mileage_column() -> None:
    dataframe = pd.DataFrame(
        {
            "fuel_type": ["Diesel", "Diesel", "Petrol", "Petrol"],
            "km_driven": [100, 120, 40, 60],
        }
    )
    plan = QueryPlan(
        original_query="Are diesel cars generally associated with higher mileage than petrol cars?",
        intent="table_analysis",
        entities=[
            {"text": "Diesel", "normalized": "Diesel", "type": "category_value", "field": "fuel_type"},
            {"text": "Petrol", "normalized": "Petrol", "type": "category_value", "field": "fuel_type"},
        ],
        metrics=[{"name": "mileage"}],
        confidence=0.9,
    )

    result = TableAnalysisTool().safe_run(_payload(plan, dataframe))

    assert result.success is True
    rows = {row["fuel_type"]: row["mean_km_driven"] for row in result.table}
    assert rows == {"Diesel": 110.0, "Petrol": 50.0}


def test_table_analysis_above_median_segment_distribution() -> None:
    dataframe = pd.DataFrame(
        {
            "fuel_type": ["Diesel", "Diesel", "Petrol", "CNG"],
            "price": [100, 300, 400, 50],
        }
    )
    plan = QueryPlan(
        original_query="If I only consider vehicles above the median price, does fuel distribution change?",
        intent="table_analysis",
        metrics=[{"name": "price"}],
        aggregations=[{"operation": "count"}],
        grouping=["fuel_type"],
        confidence=0.9,
    )

    result = TableAnalysisTool().safe_run(_payload(plan, dataframe))

    assert result.success is True
    rows = {row["fuel_type"]: row["count_rows"] for row in result.table}
    assert rows == {"Petrol": 1, "Diesel": 1}
    assert result.data["segment"] == "above_median"


def test_table_analysis_correlation_shortcut() -> None:
    dataframe = pd.DataFrame({"price": [10, 20, 30], "overall_cost": [1, 2, 3]})
    plan = QueryPlan(
        original_query="Are price and overall_cost related?",
        intent="table_analysis",
        metrics=[{"name": "price"}, {"name": "overall_cost"}],
        comparison={"type": "correlation"},
        confidence=0.9,
    )

    result = TableAnalysisTool().safe_run(_payload(plan, dataframe))

    assert result.success is True
    assert result.data["correlation"] == 1.0


def test_table_analysis_virat_max_runs_and_strike_rate() -> None:
    dataframe = pd.DataFrame(
        {
            "batter": ["Virat Kohli", "Virat Kohli", "MS Dhoni"],
            "batsman_runs": [10, 40, 50],
            "sr": [120.0, 150.0, 130.0],
        }
    )
    plan = QueryPlan(
        intent="table_analysis",
        entities=[{"text": "Virat", "normalized": "Virat Kohli"}],
        metrics=[{"name": "runs"}, {"name": "strike_rate"}],
        aggregations=[{"operation": "max"}],
        confidence=0.9,
    )

    result = TableAnalysisTool().safe_run(_payload(plan, dataframe))

    assert result.success is True
    assert result.table == [{"max_batsman_runs": 40.0, "max_sr": 150.0}]
    assert result.metadata["multi_metric"] is True


def test_table_analysis_key_insights_returns_profile_summary() -> None:
    dataframe = pd.DataFrame(
        {
            "gender": ["Female", "Male", "Female", "Male"],
            "age": [34, 30, 23, 28],
            "Avenue": ["Mutual Fund", "Equity", "Mutual Fund", "Fixed Deposits"],
            "Objective": ["Capital Appreciation", "Growth", "Capital Appreciation", "Income"],
        }
    )
    plan = QueryPlan(
        original_query="give me key insights",
        rewritten_query="give me key insights",
        intent="table_analysis",
        metrics=[{"name": "dataset_summary"}],
        confidence=0.85,
    )

    result = TableAnalysisTool().safe_run(_payload(plan, dataframe))

    assert result.success is True
    assert result.tool_name == "table_analysis_tool"
    assert result.metadata["operation"]["operation"] == "table_profile_summary"
    assert result.data["row_count"] == 4
    assert result.data["column_count"] == 4
    assert any(row["insight"] == "Dataset size" for row in result.table)
    assert "I found 4 rows and 4 columns" in result.answer


def test_table_analysis_missing_dataframe_returns_tool_result() -> None:
    plan = QueryPlan(intent="table_analysis", metrics=[{"name": "profit"}], confidence=0.9)

    result = TableAnalysisTool().safe_run({"query_plan": _dump(plan)})

    assert result.success is False
    assert result.tool_name == "table_analysis_tool"
    assert result.error_msg


def test_table_analysis_finance_average_age() -> None:
    dataframe = pd.DataFrame({"age": [34, 23, 30, 22]})
    plan = QueryPlan(
        original_query="What is the average age?",
        intent="table_analysis",
        metrics=[{"name": "age"}],
        aggregations=[{"operation": "mean"}],
        confidence=0.9,
    )

    result = TableAnalysisTool().safe_run(_payload(plan, dataframe))

    assert result.success is True
    assert result.table == [{"mean_age": 27.25}]
    assert "mean of age: 27.25" in result.answer


def test_table_analysis_finance_preferred_avenue_group_count() -> None:
    dataframe = pd.DataFrame(
        {
            "Avenue": ["Mutual Fund", "Equity", "Mutual Fund", "Fixed Deposits"],
        }
    )
    plan = QueryPlan(
        original_query="What is the most preferred investment avenue?",
        intent="table_analysis",
        grouping=["avenue"],
        aggregations=[{"operation": "count"}],
        sorting={"direction": "desc"},
        confidence=0.9,
    )

    result = TableAnalysisTool().safe_run(_payload(plan, dataframe))

    assert result.success is True
    assert result.table[0] == {"Avenue": "Mutual Fund", "count_rows": 2}


def test_table_analysis_finance_stock_market_yes_count() -> None:
    dataframe = pd.DataFrame({"Stock_Marktet": ["Yes", "No", "Yes", "Yes"]})
    plan = QueryPlan(
        original_query="How many people invest in the stock market?",
        intent="table_analysis",
        metrics=[{"name": "count"}],
        aggregations=[{"operation": "count"}],
        filters=[{"field": "stock market", "operator": "equals", "value": "Yes"}],
        confidence=0.9,
    )

    result = TableAnalysisTool().safe_run(_payload(plan, dataframe))

    assert result.success is True
    assert result.table == [{"count_rows": 3}]
