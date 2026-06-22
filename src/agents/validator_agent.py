from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from src.agents.confidence import combine_confidence, is_low_confidence, normalize_confidence
from src.models.execution import ExecutionPlan
from src.models.query import QueryPlan
from src.models.source import SourceSelection
from src.models.tool import ToolResult
from src.models.validation import ValidationResult


TABLE_TOOLS = {"table_analysis_tool"}
DOCUMENT_TOOLS = {"rag_qa_tool", "summarize_tool", "url_lookup_tool"}
CHART_TOOLS = {"chart_tool"}
SOURCE_REQUIRED_INTENTS = {
    "table_analysis",
    "chart_request",
    "summarize_document",
    "compare_documents",
    "rag_question",
    "url_lookup",
}


def _dump_model(model: Any) -> Dict[str, Any]:
    if model is None:
        return {}
    if isinstance(model, dict):
        return dict(model)
    if hasattr(model, "model_dump"):
        return model.model_dump()
    if hasattr(model, "dict"):
        return model.dict()
    return {"value": str(model)}


def _has_number(text: str) -> bool:
    return bool(re.search(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?%?", text or ""))


class ValidatorAgent:
    def validate(
        self,
        query_plan: QueryPlan,
        execution_plan: ExecutionPlan,
        tool_results: List[ToolResult],
        source_selection: Optional[SourceSelection] = None,
    ) -> ValidationResult:
        try:
            issues: List[str] = []
            warnings: List[str] = []
            clarification_needed = bool(getattr(query_plan, "clarification_needed", False))
            clarification_question = query_plan.clarification_question

            results = [result for result in (tool_results or []) if isinstance(result, ToolResult)]
            result_by_tool = {result.tool_name: result for result in results if result.tool_name}

            if not results:
                issues.append("No tool results were produced.")

            self._check_source(query_plan, source_selection, warnings)
            self._check_planned_tools(execution_plan, result_by_tool, issues, warnings)
            self._check_tool_success(results, issues, warnings)
            self._check_table_grounding(query_plan, results, issues)
            self._check_document_grounding(query_plan, results, issues, warnings)
            self._check_chart(query_plan, result_by_tool, warnings)

            confidence = self._combined_confidence(query_plan, execution_plan, results, source_selection)
            if is_low_confidence(confidence):
                warnings.append("Overall validation confidence is low.")
                clarification_needed = True
                clarification_question = clarification_question or "Could you clarify the question or upload a more relevant source?"

            if query_plan.clarification_needed:
                warnings.append("Query plan requested clarification before final response.")
                clarification_needed = True
                clarification_question = clarification_question or query_plan.clarification_question

            has_successful_required_result = any(result.success for result in results)
            is_valid = not issues and has_successful_required_result and not (clarification_needed and confidence < 0.35)
            requires_retry = bool(issues) or any(self._is_failed_required_result(query_plan, result) for result in results)

            return ValidationResult(
                is_valid=is_valid,
                confidence=confidence,
                issues=issues,
                warnings=self._dedupe(warnings),
                requires_retry=requires_retry,
                clarification_needed=clarification_needed,
                clarification_question=clarification_question,
            )
        except Exception as exc:
            return ValidationResult(
                is_valid=False,
                confidence=0.0,
                issues=["Validator agent failed safely: {0}".format(str(exc))],
                warnings=[],
                requires_retry=True,
                clarification_needed=True,
                clarification_question="Could you rephrase your request or try again?",
            )

    def _check_source(
        self,
        query_plan: QueryPlan,
        source_selection: Optional[SourceSelection],
        warnings: List[str],
    ) -> None:
        if query_plan.intent in SOURCE_REQUIRED_INTENTS and (
            source_selection is None or not source_selection.selected_source_id
        ):
            warnings.append("No selected source is available for a source-dependent query.")

    def _check_planned_tools(
        self,
        execution_plan: ExecutionPlan,
        result_by_tool: Dict[str, ToolResult],
        issues: List[str],
        warnings: List[str],
    ) -> None:
        for tool_call in execution_plan.tool_calls:
            if tool_call.tool_name not in result_by_tool:
                if tool_call.tool_name in CHART_TOOLS:
                    warnings.append("Planned chart tool did not produce a result.")
                else:
                    issues.append("Planned tool did not produce a result: {0}".format(tool_call.tool_name))

    def _check_tool_success(self, results: List[ToolResult], issues: List[str], warnings: List[str]) -> None:
        for result in results:
            warnings.extend(result.warnings or [])
            if result.success:
                continue
            if result.tool_name in CHART_TOOLS:
                warnings.append("Chart tool failed or was skipped.")
            else:
                issues.append(
                    "{0} failed: {1}".format(result.tool_name or "tool", result.error_msg or "unknown error")
                )

    def _check_table_grounding(self, query_plan: QueryPlan, results: List[ToolResult], issues: List[str]) -> None:
        table_results = [result for result in results if result.tool_name in TABLE_TOOLS]
        if query_plan.intent in {"table_analysis", "chart_request", "compare_documents"} and table_results:
            for result in table_results:
                if not result.success:
                    continue
                if not self._has_table_data(result):
                    issues.append("Table answer has no pandas-grounded data.")
                if _has_number(result.answer or "") and not self._has_numeric_grounding(result):
                    issues.append("Numeric answer is not grounded in pandas output.")

        if query_plan.intent in {"table_analysis", "chart_request"} and not table_results:
            issues.append("Table query did not produce a table analysis result.")

    def _check_document_grounding(
        self,
        query_plan: QueryPlan,
        results: List[ToolResult],
        issues: List[str],
        warnings: List[str],
    ) -> None:
        document_results = [result for result in results if result.tool_name in DOCUMENT_TOOLS]
        if query_plan.intent in {"summarize_document", "rag_question", "url_lookup", "compare_documents"} and document_results:
            for result in document_results:
                if not result.success:
                    continue
                if result.metadata.get("retrieval_empty") or result.data.get("answer_found") is False:
                    issues.append("Document retrieval did not find relevant chunks.")
                if result.answer and not result.citations:
                    issues.append("Document answer is missing citations.")
                retrieved_chunks = result.data.get("retrieved_chunks")
                if retrieved_chunks == []:
                    issues.append("Document answer has no retrieved chunks.")

        if query_plan.intent in {"summarize_document", "rag_question", "url_lookup"} and not document_results:
            issues.append("Document query did not produce a document-grounded result.")

        if query_plan.intent == "compare_documents" and not document_results:
            warnings.append("Comparison is missing document evidence.")

    def _check_chart(self, query_plan: QueryPlan, result_by_tool: Dict[str, ToolResult], warnings: List[str]) -> None:
        if not query_plan.chart_requested:
            return
        chart_result = result_by_tool.get("chart_tool")
        if chart_result is None:
            warnings.append("Chart was requested but no chart result was produced.")
        elif chart_result.success and chart_result.chart is None:
            warnings.append("Chart was requested but chart artifact is missing.")

    def _combined_confidence(
        self,
        query_plan: QueryPlan,
        execution_plan: ExecutionPlan,
        results: List[ToolResult],
        source_selection: Optional[SourceSelection],
    ) -> float:
        scores = [query_plan.confidence, execution_plan.confidence]
        if source_selection is not None:
            scores.append(source_selection.confidence)
        scores.extend(result.confidence for result in results)
        return normalize_confidence(combine_confidence(scores))

    def _has_table_data(self, result: ToolResult) -> bool:
        if isinstance(result.table, list) and len(result.table) > 0:
            return True
        data = result.data or {}
        if data.get("value") is not None:
            return True
        if data.get("metrics"):
            return True
        if data.get("rows"):
            return True
        return False

    def _has_numeric_grounding(self, result: ToolResult) -> bool:
        data_payload = _dump_model(result.data)
        if any(_has_number(str(value)) for value in data_payload.values()):
            return True
        if isinstance(result.table, list) and result.table:
            return any(_has_number(str(row)) for row in result.table)
        return False

    def _is_failed_required_result(self, query_plan: QueryPlan, result: ToolResult) -> bool:
        if result.success:
            return False
        if result.tool_name in CHART_TOOLS:
            return False
        if query_plan.intent == "general_finance" and result.tool_name == "general_finance_tool":
            return True
        return result.tool_name in TABLE_TOOLS or result.tool_name in DOCUMENT_TOOLS

    def _dedupe(self, values: List[str]) -> List[str]:
        seen = set()
        deduped = []
        for value in values:
            if value in seen:
                continue
            seen.add(value)
            deduped.append(value)
        return deduped
