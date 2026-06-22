from __future__ import annotations

from src.tools.compare_tool import CompareTool


def test_compare_tool_compares_table_and_document_results() -> None:
    result = CompareTool().safe_run(
        {
            "dependency_results": {
                "table_analysis_tool": {
                    "success": True,
                    "tool_name": "table_analysis_tool",
                    "answer": "sum revenue: 100",
                    "table": [{"sum_revenue": 100}],
                },
                "rag_qa_tool": {
                    "success": True,
                    "tool_name": "rag_qa_tool",
                    "answer": "Report mentions revenue growth.",
                    "citations": [{"source_id": "s1", "filename": "report.pdf", "text_snippet": "revenue growth"}],
                },
            }
        }
    )

    assert result.success is True
    assert "Structured data result" in result.answer
    assert "Document evidence" in result.answer
    assert len(result.table) == 2


def test_compare_tool_warns_when_only_one_input_available() -> None:
    result = CompareTool().safe_run(
        {"dependency_results": {"table_analysis_tool": {"success": True, "tool_name": "table_analysis_tool", "table": [{"x": 1}]}}}
    )

    assert result.success is True
    assert result.warnings
    assert result.confidence < 0.55


def test_compare_tool_recognizes_two_documents_inside_one_rag_result() -> None:
    result = CompareTool().safe_run(
        {
            "dependency_results": {
                "rag_qa_tool": {
                    "success": True,
                    "tool_name": "rag_qa_tool",
                    "data": {
                        "retrieved_chunks": [
                            {"source_id": "r2023", "filename": "2023.pdf", "content": "Expenses were 120."},
                            {"source_id": "r2022", "filename": "2022.pdf", "content": "Expenses were 100."},
                        ]
                    },
                    "citations": [{"source_id": "r2023"}, {"source_id": "r2022"}],
                }
            }
        }
    )

    assert result.success is True
    assert result.data["source_count"] == 2
    assert {row["source_id"] for row in result.table} == {"r2023", "r2022"}
    assert result.confidence >= 0.8


def test_compare_tool_calculates_cited_pdf_expense_difference() -> None:
    result = CompareTool().safe_run(
        {
            "query_plan": {
                "original_query": "Compare total expenses between the 2023 and 2022 PDF reports.",
                "rewritten_query": "Compare total expenses for 2023 and 2022.",
                "metrics": [{"name": "expenses"}],
            },
            "dependency_results": {
                "rag_qa_tool": {
                    "success": True,
                    "tool_name": "rag_qa_tool",
                    "data": {
                        "retrieved_chunks": [
                            {
                                "source_id": "r2023",
                                "filename": "financial_report_2023.pdf",
                                "page": 12,
                                "chunk_id": "r2023-p12",
                                "content": "Total expenses were USD 120 million in 2023.",
                            },
                            {
                                "source_id": "r2022",
                                "filename": "financial_report_2022.pdf",
                                "page": 10,
                                "chunk_id": "r2022-p10",
                                "content": "Total expenses were USD 100 million in 2022.",
                            },
                        ]
                    },
                    "citations": [{"source_id": "r2023"}, {"source_id": "r2022"}],
                }
            },
        }
    )

    assert result.success is True
    assert result.metadata["comparison_type"] == "numeric_financial"
    assert result.data["absolute_difference"] == 20_000_000.0
    assert result.data["percentage_change"] == 20.0
    assert result.data["direction"] == "increase"
    assert "USD 20 million" in result.answer
    assert {citation.page for citation in result.citations} == {10, 12}
    assert {row["period"] for row in result.table} == {"2022", "2023"}


def test_compare_tool_does_not_calculate_across_different_currencies() -> None:
    result = CompareTool().safe_run(
        {
            "query_plan": {"original_query": "Compare revenue in 2024 and 2023."},
            "dependency_results": {
                "rag_qa_tool": {
                    "success": True,
                    "data": {
                        "retrieved_chunks": [
                            {"source_id": "a", "filename": "a.pdf", "content": "Revenue was USD 10 million in 2024."},
                            {"source_id": "b", "filename": "b.pdf", "content": "Revenue was EUR 8 million in 2023."},
                        ]
                    },
                }
            },
        }
    )

    assert result.success is True
    assert result.data["absolute_difference"] is None
    assert result.data["unit_compatible"] is False
    assert result.warnings
