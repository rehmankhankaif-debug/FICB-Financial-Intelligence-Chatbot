from __future__ import annotations

from src.agents.source_selector import SourceSelector
from src.models.query import QueryPlan
from src.models.source import SourceSelection
from src.models.table import TableProfile


def test_selects_table_source_for_table_analysis() -> None:
    plan = QueryPlan(
        intent="table_analysis",
        required_source_type="table",
        original_query="average monthly profit",
        rewritten_query="Calculate average monthly profit.",
        metrics=[{"name": "profit"}],
        grouping=["month"],
        confidence=0.85,
    )
    profile = TableProfile(
        source_id="sales",
        filename="sales.csv",
        columns=["month", "profit"],
        normalized_columns={"month": "month", "profit": "profit"},
        metric_candidate_columns=["profit"],
        semantic_summary="Monthly sales profit table.",
    )

    selection = SourceSelector().select_source(
        plan,
        available_sources=[{"source_id": "sales", "filename": "sales.csv", "file_type": "csv"}],
        table_profiles=[profile],
    )

    assert isinstance(selection, SourceSelection)
    assert selection.selected_source_id == "sales"
    assert selection.source_type == "table"
    assert selection.confidence >= 0.8


def test_selects_document_source_for_summary() -> None:
    plan = QueryPlan(
        intent="summarize_document",
        required_source_type="document",
        original_query="outline this report",
        rewritten_query="Create a structured outline of the uploaded report.",
        confidence=0.86,
    )

    selection = SourceSelector().select_source(
        plan,
        available_sources=[{"source_id": "report", "filename": "annual_report.pdf", "file_type": "pdf"}],
        document_metadata=[{"source_id": "report", "title": "Annual report", "summary": "Financial report"}],
    )

    assert selection.selected_source_id == "report"
    assert selection.source_type == "document"
    assert selection.confidence >= 0.55


def test_no_source_returns_low_confidence() -> None:
    plan = QueryPlan(intent="table_analysis", required_source_type="table", confidence=0.8)

    selection = SourceSelector().select_source(plan, available_sources=[])

    assert selection.selected_source_id is None
    assert selection.confidence == 0.0
    assert "No uploaded sources" in selection.reason


def test_irrelevant_source_is_not_selected() -> None:
    plan = QueryPlan(
        intent="table_analysis",
        required_source_type="table",
        original_query="average monthly profit",
        metrics=[{"name": "profit"}],
        grouping=["month"],
        confidence=0.8,
    )

    selection = SourceSelector().select_source(
        plan,
        available_sources=[{"source_id": "report", "filename": "annual_report.pdf", "file_type": "pdf"}],
    )

    assert selection.selected_source_id is None
    assert selection.confidence < 0.55


def test_selects_only_table_source_from_nested_metadata_category() -> None:
    plan = QueryPlan(
        intent="table_analysis",
        required_source_type="table",
        original_query="give me key insights from this dataset",
        metrics=[{"name": "dataset_summary"}],
        confidence=0.72,
    )

    selection = SourceSelector().select_source(
        plan,
        available_sources=[
            {
                "source_id": "finance",
                "filename": "Finance_data",
                "metadata": {"source_category": "table"},
            }
        ],
    )

    assert selection.selected_source_id == "finance"
    assert selection.source_type == "table"
    assert selection.confidence >= 0.58


def test_comparison_selects_every_relevant_uploaded_document() -> None:
    plan = QueryPlan(
        intent="compare_documents",
        required_source_type="mixed",
        original_query="Compare total expenses in the 2023 and 2022 PDF reports",
        confidence=0.9,
    )

    selection = SourceSelector().select_source(
        plan,
        available_sources=[
            {"source_id": "report-2023", "filename": "report_2023.pdf", "file_type": "pdf"},
            {"source_id": "report-2022", "filename": "report_2022.pdf", "file_type": "pdf"},
        ],
    )

    assert selection.source_type == "mixed"
    assert set(selection.selected_source_ids) == {"report-2023", "report-2022"}
    assert set(selection.selected_source_types.values()) == {"document"}
