from __future__ import annotations

from typing import Optional

import pandas as pd

from src.agents.query_planner import QueryPlannerAgent
from src.llm.gemini_client import GeminiClient
from src.models.query import QueryPlan, RewrittenQuery
from src.table_intelligence.profiler import TableProfiler


class FakeGemini(GeminiClient):
    def __init__(self, payload):
        self.payload = payload

    def is_available(self) -> bool:
        return True

    def generate_json(self, prompt, fallback=None):
        return self.payload


def _rewritten(original: str, rewritten: Optional[str] = None) -> RewrittenQuery:
    return RewrittenQuery(
        original_query=original,
        rewritten_query=rewritten or original,
        language="hi-en",
        detected_language="hi-en",
        confidence=0.8,
    )


def test_average_monthly_profit_plan() -> None:
    plan = QueryPlannerAgent(gemini_client=GeminiClient(api_key="", client=None)).plan(
        "average monthly profit batao",
        _rewritten("average monthly profit batao"),
        available_sources=[{"source_id": "sales", "file_type": "csv"}],
    )

    assert plan.intent == "table_analysis"
    assert any(metric["name"] == "profit" for metric in plan.metrics)
    assert any(item["operation"] == "mean" for item in plan.aggregations)
    assert "month" in plan.grouping


def test_manual_automatic_chart_plan() -> None:
    plan = QueryPlannerAgent(gemini_client=GeminiClient(api_key="", client=None)).plan(
        "manual aur automatic cars kitni hain bar graph bnao",
        _rewritten("manual aur automatic cars kitni hain bar graph bnao"),
        available_sources=[{"source_id": "cars", "file_type": "csv"}],
    )

    assert plan.intent == "chart_request"
    assert plan.chart_requested is True
    assert plan.chart_type == "bar"
    assert "transmission" in plan.grouping
    assert any(metric["name"] == "count" for metric in plan.metrics)


def test_outline_report_plan() -> None:
    plan = QueryPlannerAgent(gemini_client=GeminiClient(api_key="", client=None)).plan(
        "outline this report",
        _rewritten("outline this report", "Create a structured outline of the uploaded report."),
        available_sources=[{"source_id": "report", "file_type": "pdf"}],
    )

    assert plan.intent == "summarize_document"
    assert plan.required_source_type == "document"


def test_short_summary_request_uses_uploaded_document_context() -> None:
    plan = QueryPlannerAgent(gemini_client=GeminiClient(api_key="", client=None)).plan(
        "Summarise please",
        _rewritten("Summarise please"),
        available_sources=[{"source_id": "loan", "filename": "loan.pdf", "file_type": "pdf", "source_category": "document"}],
    )

    assert plan.intent == "summarize_document"
    assert plan.required_source_type == "document"
    assert plan.clarification_needed is False
    assert plan.confidence >= 0.76


def test_short_summary_request_repairs_mocked_gemini_general_finance_when_document_exists() -> None:
    agent = QueryPlannerAgent(
        gemini_client=FakeGemini(
            {
                "intent": "general_finance",
                "confidence": 0.3,
                "clarification_needed": True,
                "clarification_question": "Ask about a finance term.",
            }
        )
    )

    plan = agent.plan(
        "Summarise please",
        _rewritten("Summarise please"),
        available_sources=[{"source_id": "loan", "filename": "loan.pdf", "file_type": "pdf", "source_category": "document"}],
    )

    assert plan.intent == "summarize_document"
    assert plan.required_source_type == "document"
    assert plan.clarification_needed is False
    assert plan.confidence >= 0.76


def test_short_summary_request_uses_table_summary_when_only_csv_exists() -> None:
    plan = QueryPlannerAgent(gemini_client=GeminiClient(api_key="", client=None)).plan(
        "summary please",
        _rewritten("summary please"),
        available_sources=[{"source_id": "finance", "filename": "Finance_data.csv", "file_type": "csv", "source_category": "table"}],
    )

    assert plan.intent == "table_analysis"
    assert plan.required_source_type == "table"
    assert any(metric["name"] == "dataset_summary" for metric in plan.metrics)


def test_virat_runs_strike_rate_plan() -> None:
    plan = QueryPlannerAgent(gemini_client=GeminiClient(api_key="", client=None)).plan(
        "Virat ke maximum runs aur strike rate batao",
        _rewritten("Virat ke maximum runs aur strike rate batao"),
        available_sources=[{"source_id": "ipl", "file_type": "csv"}],
    )

    assert plan.intent == "table_analysis"
    assert any(entity.get("normalized") == "Virat Kohli" for entity in plan.entities)
    assert any(metric["name"] == "runs" for metric in plan.metrics)
    assert any(metric["name"] == "strike_rate" for metric in plan.metrics)
    assert any(item["operation"] == "max" for item in plan.aggregations)


def test_compare_revenue_trends_plan() -> None:
    plan = QueryPlannerAgent(gemini_client=GeminiClient(api_key="", client=None)).plan(
        "compare revenue trends in csv with annual report",
        _rewritten("compare revenue trends in csv with annual report"),
        available_sources=[{"source_id": "sales", "file_type": "csv"}, {"source_id": "annual", "file_type": "pdf"}],
    )

    assert plan.intent == "compare_documents"
    assert plan.required_source_type == "mixed"
    assert plan.comparison["type"] == "revenue_trend"


def test_anomaly_trends_do_not_force_an_unrequested_chart() -> None:
    dataframe = pd.DataFrame(
        {
            "fuel_type": ["Diesel", "Petrol"],
            "transmission": ["Manual", "Automatic"],
            "price": [100, 200],
        }
    )
    profile = TableProfiler().profile(dataframe, source_id="cars", filename="cars.csv")
    query = "Identify unusual trends involving fuel_type, transmission, and price."

    plan = QueryPlannerAgent(gemini_client=GeminiClient(api_key="", client=None)).plan(
        query,
        _rewritten(query),
        available_sources=[{"source_id": "cars", "filename": "cars.csv", "file_type": "csv"}],
        table_profiles=[profile],
    )

    assert plan.intent == "table_analysis"
    assert plan.chart_requested is False


def test_offline_spend_column_does_not_match_line_chart_substring() -> None:
    dataframe = pd.DataFrame({"Date": ["2025-01", "2025-02"], "Offline_Spend": [100, 150]})
    profile = TableProfiler().profile(dataframe, source_id="marketing", filename="marketing.csv")
    query = "Overall Offline_Spend ka scene kaisa hai? Average batao."

    plan = QueryPlannerAgent(gemini_client=GeminiClient(api_key="", client=None)).plan(
        query,
        _rewritten(query),
        available_sources=[{"source_id": "marketing", "filename": "marketing.csv", "file_type": "csv"}],
        table_profiles=[profile],
    )

    assert plan.intent == "table_analysis"
    assert plan.chart_requested is False
    assert plan.chart_types == []


def test_key_insights_with_csv_source_routes_to_table_analysis() -> None:
    plan = QueryPlannerAgent(gemini_client=GeminiClient(api_key="", client=None)).plan(
        "give me key insights",
        _rewritten("give me key insights"),
        available_sources=[{"source_id": "finance", "file_type": "csv", "source_category": "table"}],
    )

    assert plan.intent == "table_analysis"
    assert plan.required_source_type == "table"
    assert plan.clarification_needed is False
    assert any(metric["name"] == "dataset_summary" for metric in plan.metrics)


def test_mocked_gemini_plan_is_used_and_normalized() -> None:
    agent = QueryPlannerAgent(
        gemini_client=FakeGemini(
            {
                "original_query": "manual cars chart",
                "rewritten_query": "Count manual cars and chart them.",
                "language": "en",
                "intent": "chart_request",
                "required_source_type": "table",
                "metrics": ["count"],
                "grouping": "transmission",
                "chart_requested": True,
                "chart_type": "bar",
                "confidence": 0.9,
            }
        )
    )

    plan = agent.plan("manual cars chart", _rewritten("manual cars chart"))

    assert isinstance(plan, QueryPlan)
    assert plan.intent == "chart_request"
    assert plan.metrics == [{"name": "count"}]
    assert plan.grouping == ["transmission"]


def test_gender_count_multi_chart_repairs_gemini_chart_type_list() -> None:
    profile = TableProfiler().profile(
        pd.DataFrame({"Gender": ["M", "F", "F"], "Tenure_Months": [12, 24, 36]}),
        source_id="customers",
        filename="CustomersData.xlsx",
    )
    agent = QueryPlannerAgent(
        gemini_client=FakeGemini(
            {
                "intent": "chart_request",
                "required_source_type": "table",
                "metrics": [{"name": "count"}],
                "aggregations": [{"operation": "count"}],
                "grouping": ["Gender"],
                "chart_requested": True,
                "chart_type": ["bar_chart", "pie_chart"],
                "confidence": 0.94,
            }
        )
    )

    plan = agent.plan(
        "Gender M and F give me Bar Graph and pie chart of quantity",
        _rewritten("Gender M and F give me Bar Graph and pie chart of quantity"),
        available_sources=[{"source_id": "customers", "file_type": "xlsx", "source_category": "table"}],
        table_profiles=[profile],
    )

    assert plan.intent == "chart_request"
    assert plan.required_source_type == "table"
    assert plan.chart_types == ["bar", "pie"]
    assert plan.chart_type == "bar"
    assert plan.grouping == ["Gender"]
    assert any(metric["name"] == "count" for metric in plan.metrics)
    assert plan.clarification_needed is False


def test_gender_quantity_multi_chart_routes_to_grouped_pandas_count_without_llm() -> None:
    profile = TableProfiler().profile(
        pd.DataFrame({"Gender": ["M", "F", "F"], "Tenure_Months": [12, 24, 36]}),
        source_id="customers",
        filename="CustomersData.xlsx",
    )
    query = "Gender M and F give me Bar Graph and pie chart of quantity"

    plan = QueryPlannerAgent(gemini_client=GeminiClient(api_key="", client=None)).plan(
        query,
        _rewritten(query),
        available_sources=[{"source_id": "customers", "file_type": "xlsx", "source_category": "table"}],
        table_profiles=[profile],
    )

    assert plan.intent == "chart_request"
    assert plan.chart_types == ["bar", "pie"]
    assert plan.grouping == ["Gender"]
    assert any(metric["name"] == "count" for metric in plan.metrics)
    assert plan.aggregations[0]["operation"] == "count"


def test_key_insights_overrides_mocked_gemini_general_finance_when_csv_exists() -> None:
    agent = QueryPlannerAgent(
        gemini_client=FakeGemini(
            {
                "intent": "general_finance",
                "confidence": 0.4,
                "clarification_needed": True,
                "clarification_question": "Ask about a finance term.",
            }
        )
    )

    plan = agent.plan(
        "key insights",
        _rewritten("key insights"),
        available_sources=[{"source_id": "finance", "file_type": "csv", "source_category": "table"}],
    )

    assert plan.intent == "table_analysis"
    assert plan.required_source_type == "table"
    assert plan.clarification_needed is False
    assert plan.confidence >= 0.72


def test_finance_csv_average_age_routes_to_table_analysis() -> None:
    plan = QueryPlannerAgent(gemini_client=GeminiClient(api_key="", client=None)).plan(
        "What is the average age of respondents?",
        _rewritten("What is the average age of respondents?"),
        available_sources=[{"source_id": "finance", "filename": "Finance_data.csv", "file_type": "csv"}],
    )

    assert plan.intent == "table_analysis"
    assert plan.required_source_type == "table"
    assert any(metric["name"] == "age" for metric in plan.metrics)
    assert any(item["operation"] == "mean" for item in plan.aggregations)


def test_finance_csv_preferred_investment_avenue_routes_to_grouped_count() -> None:
    plan = QueryPlannerAgent(gemini_client=GeminiClient(api_key="", client=None)).plan(
        "What is the most preferred investment avenue?",
        _rewritten("What is the most preferred investment avenue?"),
        available_sources=[{"source_id": "finance", "filename": "Finance_data.csv", "file_type": "csv"}],
    )

    assert plan.intent == "table_analysis"
    assert "avenue" in plan.grouping
    assert any(item["operation"] == "count" for item in plan.aggregations)
    assert plan.sorting["direction"] == "desc"


def test_finance_csv_stock_market_count_adds_yes_filter() -> None:
    plan = QueryPlannerAgent(gemini_client=GeminiClient(api_key="", client=None)).plan(
        "How many people invest in the stock market?",
        _rewritten("How many people invest in the stock market?"),
        available_sources=[{"source_id": "finance", "filename": "Finance_data.csv", "file_type": "csv"}],
    )

    assert plan.intent == "table_analysis"
    assert any(metric["name"] == "count" for metric in plan.metrics)
    assert plan.filters == [{"field": "stock market", "operator": "equals", "value": "Yes", "confidence": 0.82}]


def test_finance_csv_information_source_routes_to_grouped_count() -> None:
    plan = QueryPlannerAgent(gemini_client=GeminiClient(api_key="", client=None)).plan(
        "Which source of financial information is most common?",
        _rewritten("Which source of financial information is most common?"),
        available_sources=[{"source_id": "finance", "filename": "Finance_data.csv", "file_type": "csv"}],
    )

    assert plan.intent == "table_analysis"
    assert "source" in plan.grouping
    assert any(item["operation"] == "count" for item in plan.aggregations)


def test_finance_csv_average_age_repairs_empty_gemini_metrics() -> None:
    agent = QueryPlannerAgent(
        gemini_client=FakeGemini(
            {
                "intent": "general_finance",
                "metrics": [],
                "aggregations": [{"operation": "mean", "confidence": 0.78}],
                "confidence": 0.57,
            }
        )
    )

    plan = agent.plan(
        "What is the average age of respondents?",
        _rewritten("What is the average age of respondents?"),
        available_sources=[{"source_id": "finance", "filename": "Finance_data.csv", "file_type": "csv"}],
    )

    assert plan.intent == "table_analysis"
    assert plan.required_source_type == "table"
    assert any(metric["name"] == "age" for metric in plan.metrics)
    assert any(item["operation"] == "mean" for item in plan.aggregations)


def test_semantic_schema_extracts_fuel_type_grouping_for_chart() -> None:
    dataframe = pd.DataFrame(
        {
            "fuel_type": ["Diesel", "Petrol", "Petrol", "CNG"],
            "price": [100, 200, 300, 150],
        }
    )
    profile = TableProfiler().profile(dataframe, source_id="cars", filename="pre-owned cars.csv")

    plan = QueryPlannerAgent(gemini_client=GeminiClient(api_key="", client=None)).plan(
        "Give me fuel type quantity data and bar graph",
        _rewritten("Give me fuel type quantity data and bar graph"),
        available_sources=[{"source_id": "cars", "filename": "pre-owned cars.csv", "file_type": "csv"}],
        table_profiles=[profile],
    )

    assert plan.intent == "chart_request"
    assert plan.required_source_type == "table"
    assert plan.chart_requested is True
    assert plan.chart_type == "bar"
    assert "fuel_type" in plan.grouping
    assert any(metric["name"] == "count" for metric in plan.metrics)
    assert any(item["operation"] == "count" for item in plan.aggregations)


def test_semantic_schema_extracts_transmission_fuel_cross_tab_without_long_sample_entities() -> None:
    dataframe = pd.DataFrame(
        {
            "brand": ["Ford", "Tata", "Honda", "Hyundai"],
            "model": ["Ecosport TITANIUM 1.5L DIESEL", "NEXON XMA DIESEL", "City VX", "i20 Sportz"],
            "transmission": ["Manual", "Automatic", "Manual", "Automatic"],
            "fuel_type": ["Diesel", "Diesel", "Petrol", "CNG"],
            "title": [
                "Ford Ecosport TITANIUM 1.5L DIESEL",
                "Tata NEXON XMA DIESEL",
                "Honda City VX Petrol",
                "Hyundai i20 Sportz CNG",
            ],
        }
    )
    profile = TableProfiler().profile(dataframe, source_id="cars", filename="pre-owned cars.csv")
    query = "Manual transmission mei petrol diesel cng and automatic mei petrol disel cng etc kitni h, graph do"

    plan = QueryPlannerAgent(gemini_client=GeminiClient(api_key="", client=None)).plan(
        query,
        _rewritten(query),
        available_sources=[{"source_id": "cars", "filename": "pre-owned cars.csv", "file_type": "csv"}],
        table_profiles=[profile],
    )

    assert plan.intent == "chart_request"
    assert "transmission" in plan.grouping
    assert "fuel_type" in plan.grouping
    entity_fields = {entity.get("field") for entity in plan.entities if isinstance(entity, dict) and entity.get("field")}
    assert entity_fields.issubset({"transmission", "fuel_type"})
    entity_values = {str(entity.get("normalized") or entity.get("text")) for entity in plan.entities if isinstance(entity, dict)}
    assert "Ecosport TITANIUM 1.5L DIESEL" not in entity_values
    assert "NEXON XMA DIESEL" not in entity_values
