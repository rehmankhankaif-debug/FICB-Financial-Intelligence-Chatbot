from __future__ import annotations

from src.agents.query_planner import QueryPlannerAgent
from src.agents.query_rewriter import QueryRewriterAgent
from src.agents.source_selector import SourceSelector
from src.agents.tool_planner import ToolPlannerAgent
from src.llm.gemini_client import GeminiClient
from src.models.table import TableProfile
from src.tools.manager import ToolManager


def _offline_client() -> GeminiClient:
    return GeminiClient(api_key="", client=None)


def _plan(query: str, sources, profiles=None, language=None):
    rewritten = QueryRewriterAgent(gemini_client=_offline_client()).rewrite(query, language=language)
    return QueryPlannerAgent(gemini_client=_offline_client()).plan(
        query,
        rewritten,
        available_sources=sources,
        table_profiles=profiles or [],
    )


def test_assignment_excel_revenue_trend_routes_to_line_chart() -> None:
    source = {"source_id": "sales", "filename": "sales.xlsx", "file_type": "xlsx"}
    profile = TableProfile(
        source_id="sales", filename="sales.xlsx", columns=["Quarter", "Revenue"],
        normalized_columns={"quarter": "Quarter", "revenue": "Revenue"},
        categorical_columns=["Quarter"], numeric_columns=["Revenue"],
        metric_candidate_columns=["Revenue"],
    )

    plan = _plan("Show me the revenue trends for Q1 and Q2 from the attached Excel file.", [source], [profile])

    assert plan.intent == "chart_request"
    assert plan.required_source_type == "table"
    assert "line" in plan.chart_types
    assert any(item.get("operation") == "sum" for item in plan.aggregations)


def test_assignment_two_pdf_expense_comparison_uses_both_documents() -> None:
    sources = [
        {"source_id": "r2023", "filename": "financial_report_2023.pdf", "file_type": "pdf"},
        {"source_id": "r2022", "filename": "financial_report_2022.pdf", "file_type": "pdf"},
    ]
    plan = _plan("Compare the total expenses between the 2023 and 2022 financial reports in the PDF documents.", sources)
    selection = SourceSelector().select_source(plan, sources)
    execution = ToolPlannerAgent(ToolManager().get_registry()).create_execution_plan(plan, selection)

    assert plan.intent == "compare_documents"
    assert set(selection.selected_source_ids) == {"r2023", "r2022"}
    assert [call.tool_name for call in execution.tool_calls] == ["rag_qa_tool", "compare_tool"]


def test_assignment_docx_summary_routes_to_summary_tool() -> None:
    source = {"source_id": "outlook", "filename": "market_outlook.docx", "file_type": "docx"}
    plan = _plan("Summarize the key points from the DOCX document on market outlook.", [source])
    selection = SourceSelector().select_source(plan, [source])
    execution = ToolPlannerAgent(ToolManager().get_registry()).create_execution_plan(plan, selection)

    assert plan.intent == "summarize_document"
    assert [call.tool_name for call in execution.tool_calls] == ["summarize_tool"]


def test_assignment_csv_average_monthly_profit_plans_pandas_mean() -> None:
    source = {"source_id": "profit", "filename": "profit.csv", "file_type": "csv"}
    profile = TableProfile(
        source_id="profit", filename="profit.csv", columns=["Month", "Profit"],
        normalized_columns={"month": "Month", "profit": "Profit"},
        categorical_columns=["Month"], numeric_columns=["Profit"],
        metric_candidate_columns=["Profit"],
    )

    plan = _plan("What is the average monthly profit according to the CSV data provided?", [source], [profile])

    assert plan.intent == "table_analysis"
    assert any(item.get("operation") == "mean" for item in plan.aggregations)
    assert plan.required_source_type == "table"


def test_assignment_excel_top_five_products_routes_to_pandas_ranking() -> None:
    source = {"source_id": "products", "filename": "products.xlsx", "file_type": "xlsx"}
    profile = TableProfile(
        source_id="products", filename="products.xlsx", columns=["Product", "Sales"],
        normalized_columns={"product": "Product", "sales": "Sales"},
        categorical_columns=["Product"], numeric_columns=["Sales"],
        metric_candidate_columns=["Sales"], entity_candidate_columns=["Product"],
    )

    plan = _plan("Extract the top five products by sales from the embedded table in the Excel file.", [source], [profile])

    assert plan.intent == "table_analysis"
    assert plan.limit == 5


def test_assignment_spanish_report_query_keeps_spanish_and_uses_rag() -> None:
    source = {"source_id": "informe", "filename": "informe.pdf", "file_type": "pdf"}
    query = "¿Cuáles son las tendencias de ingresos del último trimestre según el informe adjunto?"
    plan = _plan(query, [source], language="es")

    assert plan.language == "es"
    assert plan.intent == "rag_question"
    assert plan.required_source_type == "document"


def test_assignment_linked_report_query_routes_to_url_lookup() -> None:
    source = {"source_id": "market", "filename": "market_report", "file_type": "url"}
    plan = _plan("What are the latest market trends mentioned in the linked online report?", [source])

    assert plan.intent == "url_lookup"
    assert plan.required_source_type == "document"
