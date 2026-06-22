from __future__ import annotations

from typing import Any, Dict, List

from src.models.tool import ToolResult
from src.tools.base import BaseTool
from src.tools.financial_comparison import compare_document_financial_metrics


class CompareTool(BaseTool):
    name = "compare_tool"
    description = "Compare structured table outputs and document evidence."
    supported_intents = ["compare_documents"]
    supported_source_types = ["mixed", "table", "document"]
    input_types = ["table_analysis_tool result", "rag_qa_tool result", "summarize_tool result"]
    output_types = ["comparison", "warnings", "metadata"]
    input_requirements = ["at least two dependency results"]
    capabilities = ["compare_metrics", "compare_trends", "cross_source_comparison"]
    positive_examples = ["compare revenue trends in csv with annual report"]
    negative_examples = ["manual automatic cars bar graph"]
    can_chain_after = ["table_analysis_tool", "rag_qa_tool"]
    confidence = 0.84

    def run(self, input_payload: Dict[str, Any]) -> ToolResult:
        try:
            dependency_results = input_payload.get("dependency_results") or {}
            previous_results = input_payload.get("previous_results") or []
            items = list(dependency_results.values()) or previous_results
            successful = [item for item in items if isinstance(item, dict) and item.get("success")]
            source_rows = self._source_rows(successful)
            source_count = len({row.get("source_id") for row in source_rows if row.get("source_id")})
            financial_comparison = compare_document_financial_metrics(
                self._retrieved_chunks(successful),
                input_payload.get("query_plan") or {},
            )
            if financial_comparison:
                comparison_data = dict(financial_comparison["data"])
                comparison_data.update(
                    {
                        "input_count": len(items),
                        "successful_input_count": len(successful),
                    }
                )
                return ToolResult(
                    success=True,
                    tool_name=self.name,
                    data=comparison_data,
                    answer=financial_comparison["answer"],
                    table=financial_comparison["table"],
                    citations=financial_comparison["citations"],
                    confidence=0.92,
                    warnings=financial_comparison["warnings"],
                    metadata={
                        "compared_tools": [item.get("tool_name") for item in successful],
                        "comparison_type": "numeric_financial",
                    },
                )
            warnings = []
            if len(successful) < 2 and source_count < 2:
                warnings.append("Comparison has fewer than two successful inputs.")

            table_items = [item for item in successful if item.get("table")]
            citation_items = [item for item in successful if item.get("citations")]
            answer_parts = []
            if table_items:
                answer_parts.append("Structured data result available with {0} table result(s).".format(len(table_items)))
            if citation_items:
                answer_parts.append("Document evidence available with {0} cited result(s).".format(len(citation_items)))
            if not answer_parts:
                answer_parts.append("No comparable evidence was available.")

            return ToolResult(
                success=bool(successful),
                tool_name=self.name,
                data={"input_count": len(items), "successful_input_count": len(successful), "source_count": source_count},
                answer=" ".join(answer_parts),
                table=source_rows or self._comparison_table(successful),
                citations=[],
                confidence=self.confidence if len(successful) >= 2 or source_count >= 2 else 0.45,
                warnings=warnings,
                error_msg=None if successful else "No successful inputs available for comparison.",
                metadata={"compared_tools": [item.get("tool_name") for item in successful]},
            )
        except Exception as exc:
            return ToolResult(success=False, tool_name=self.name, confidence=0.0, error_msg="Comparison failed safely: {0}".format(str(exc)), metadata={})

    def _comparison_table(self, successful: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        rows = []
        for item in successful:
            rows.append(
                {
                    "tool_name": item.get("tool_name"),
                    "answer": item.get("answer"),
                    "has_table": bool(item.get("table")),
                    "citation_count": len(item.get("citations") or []),
                }
            )
        return rows

    def _source_rows(self, successful: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for item in successful:
            for source_result in (item.get("data") or {}).get("source_results", []):
                rows.append(
                    {
                        "source_id": source_result.get("source_id"),
                        "filename": source_result.get("filename"),
                        "answer": source_result.get("answer"),
                        "evidence_type": "table",
                    }
                )
            chunks = (item.get("data") or {}).get("retrieved_chunks", [])
            grouped: Dict[str, Dict[str, Any]] = {}
            for chunk in chunks:
                source_id = str(chunk.get("source_id") or "")
                if not source_id:
                    continue
                row = grouped.setdefault(
                    source_id,
                    {
                        "source_id": source_id,
                        "filename": chunk.get("filename") or source_id,
                        "answer": "",
                        "evidence_type": "document",
                    },
                )
                if chunk.get("content") and len(row["answer"]) < 1200:
                    row["answer"] = (row["answer"] + " " + str(chunk.get("content"))).strip()
            rows.extend(grouped.values())
        return rows

    def _retrieved_chunks(self, successful: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        chunks: List[Dict[str, Any]] = []
        for item in successful:
            for chunk in (item.get("data") or {}).get("retrieved_chunks", []):
                if isinstance(chunk, dict):
                    chunks.append(dict(chunk))
        return chunks
