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
