from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from src.agents.confidence import combine_confidence, normalize_confidence
from src.llm.gemini_client import GeminiClient
from src.models.citation import Citation
from src.models.execution import ExecutionPlan
from src.models.query import QueryPlan
from src.models.response import FinalResponse
from src.models.source import SourceSelection
from src.models.tool import ToolResult
from src.models.validation import ValidationResult
from src.utils.text_summary import clean_document_text
from src.utils.language import detect_language, normalize_language_code


MAX_PROMPT_TABLE_ROWS = 12
MAX_PROMPT_CITATIONS = 8
DOCUMENT_TOOL_NAMES = {"rag_qa_tool", "summarize_tool", "url_lookup_tool"}


def _as_tool_result(value: Any) -> Optional[ToolResult]:
    if isinstance(value, ToolResult):
        return value
    if isinstance(value, dict):
        try:
            return ToolResult(**value)
        except Exception:
            return None
    return None


def _as_citation(value: Any) -> Optional[Citation]:
    if isinstance(value, Citation):
        return value
    if isinstance(value, dict):
        try:
            return Citation(**value)
        except Exception:
            return None
    return None


def _number_tokens(text: str) -> List[str]:
    tokens = re.findall(r"(?<![A-Za-z])[-+]?\d+(?:,\d{3})*(?:\.\d+)?%?", text or "")
    return [token.replace(",", "") for token in tokens]


class ResponseNarrator:
    def __init__(self, gemini_client: Optional[GeminiClient] = None) -> None:
        self.gemini_client = gemini_client or GeminiClient()

    def narrate(
        self,
        original_query: str,
        rewritten_query: str,
        query_plan: QueryPlan,
        source_selection: Optional[SourceSelection],
        execution_plan: Optional[ExecutionPlan],
        tool_results: List[ToolResult],
        validation_result: ValidationResult,
        language_preference: Optional[str] = None,
    ) -> FinalResponse:
        try:
            results = self._tool_results(tool_results)
            language = self._resolve_language(language_preference, query_plan, original_query)
            warnings = self._collect_warnings(results, execution_plan, validation_result)
            citations = self._collect_citations(results)
            table = self._first_table(results)
            chart = self._first_chart(results)
            confidence = self._final_confidence(query_plan, source_selection, execution_plan, results, validation_result)

            if validation_result.clarification_needed and validation_result.clarification_question:
                warnings.append("Clarification needed: {0}".format(validation_result.clarification_question))
                warnings = self._dedupe(warnings)

            if not validation_result.is_valid:
                answer = self._failure_answer(validation_result, language)
                return FinalResponse(
                    answer=answer,
                    table=table,
                    chart=chart,
                    citations=citations,
                    warnings=warnings,
                    confidence=confidence,
                    metadata=self._metadata(language, query_plan, source_selection, execution_plan, results, validation_result, False),
                )

            fallback_answer = self._fallback_answer(results, citations, warnings, chart is not None, language)
            gemini_answer = self._gemini_narration(
                original_query=original_query,
                rewritten_query=rewritten_query,
                query_plan=query_plan,
                tool_results=results,
                validation_result=validation_result,
                citations=citations,
                warnings=warnings,
                language=language,
                fallback_answer=fallback_answer,
            )
            used_gemini = bool(gemini_answer)
            answer = gemini_answer or fallback_answer

            metadata = self._metadata(language, query_plan, source_selection, execution_plan, results, validation_result, used_gemini)
            metadata["narration_mode"] = "gemini" if used_gemini else "deterministic_fallback"
            if not used_gemini and self._gemini_available():
                metadata["gemini_fallback_used"] = True
                metadata["gemini_error_type"] = getattr(self.gemini_client, "last_error_type", None)
                if metadata["gemini_error_type"] == "QuotaExceededError":
                    warnings.append("Gemini quota is exhausted. The verified local summary is shown; retry after the provider quota resets.")
                else:
                    warnings.append("AI narration was unavailable; showing the verified deterministic fallback.")
                warnings = self._dedupe(warnings)
            elif not used_gemini:
                metadata["gemini_fallback_used"] = True
                warnings.append("Gemini narration is not configured in this running app process; showing the deterministic fallback.")
                warnings = self._dedupe(warnings)

            return FinalResponse(
                answer=answer,
                table=table,
                chart=chart,
                citations=citations,
                warnings=warnings,
                confidence=confidence,
                metadata=metadata,
            )
        except Exception as exc:
            return FinalResponse(
                answer="I could not prepare a final response safely.",
                table=None,
                chart=None,
                citations=[],
                warnings=["Response narration failed safely: {0}".format(str(exc))],
                confidence=0.0,
                metadata={"error_type": exc.__class__.__name__},
            )

    def _tool_results(self, tool_results: List[ToolResult]) -> List[ToolResult]:
        results = []
        for item in tool_results or []:
            result = _as_tool_result(item)
            if result is not None:
                results.append(result)
        return results

    def _resolve_language(self, language_preference: Optional[str], query_plan: QueryPlan, original_query: str) -> str:
        if language_preference:
            return normalize_language_code(language_preference)
        plan_language = normalize_language_code(getattr(query_plan, "language", "") or "")
        detected_language = normalize_language_code(detect_language(original_query or ""))
        if detected_language == "hi-en":
            return detected_language
        if plan_language and plan_language != "en":
            return plan_language
        return plan_language or detected_language

    def _collect_warnings(
        self,
        results: List[ToolResult],
        execution_plan: Optional[ExecutionPlan],
        validation_result: ValidationResult,
    ) -> List[str]:
        warnings: List[str] = []
        if execution_plan is not None:
            warnings.extend(execution_plan.warnings or [])
        warnings.extend(validation_result.warnings or [])
        for issue in validation_result.issues or []:
            warnings.append("Issue: {0}".format(issue))
        for result in results:
            warnings.extend(result.warnings or [])
            if not result.success and result.error_msg:
                warnings.append("{0} failed: {1}".format(result.tool_name or "tool", result.error_msg))
        return self._dedupe([warning for warning in warnings if warning])

    def _collect_citations(self, results: List[ToolResult]) -> List[Citation]:
        citations: List[Citation] = []
        seen = set()
        for result in results:
            for item in result.citations or []:
                citation = _as_citation(item)
                if citation is None:
                    continue
                key = (citation.source_id, citation.filename, citation.page, citation.chunk_id, citation.text_snippet)
                if key in seen:
                    continue
                seen.add(key)
                citations.append(citation)
        return citations

    def _first_table(self, results: List[ToolResult]) -> Any:
        for result in results:
            table = result.table
            if table is None:
                continue
            if isinstance(table, list) and not table:
                continue
            return table
        return None

    def _first_chart(self, results: List[ToolResult]) -> Any:
        for result in results:
            if result.chart is not None:
                return result.chart
        return None

    def _final_confidence(
        self,
        query_plan: QueryPlan,
        source_selection: Optional[SourceSelection],
        execution_plan: Optional[ExecutionPlan],
        results: List[ToolResult],
        validation_result: ValidationResult,
    ) -> float:
        if validation_result and not validation_result.is_valid:
            return normalize_confidence(validation_result.confidence)

        scores = [getattr(query_plan, "confidence", 0.0), getattr(validation_result, "confidence", 0.0)]
        if execution_plan is not None:
            scores.append(execution_plan.confidence)
        if source_selection is not None:
            scores.append(source_selection.confidence)
        scores.extend(result.confidence for result in results)
        return normalize_confidence(combine_confidence(scores))

    def _failure_answer(self, validation_result: ValidationResult, language: str) -> str:
        issues = validation_result.issues or ["The answer could not be validated."]
        if language == "hi-en":
            parts = ["Main is query ka fully validated answer nahi de paaya."]
            parts.append("Reason: {0}".format("; ".join(issues)))
            if validation_result.clarification_needed and validation_result.clarification_question:
                parts.append("Clarification: {0}".format(validation_result.clarification_question))
            return " ".join(parts)

        if language == "es":
            parts = ["No pude generar una respuesta completamente validada para esta consulta."]
            parts.append("Motivo: {0}".format("; ".join(issues)))
            if validation_result.clarification_needed and validation_result.clarification_question:
                parts.append("Se necesita una aclaración: {0}".format(validation_result.clarification_question))
            return " ".join(parts)

        parts = ["I could not produce a fully validated answer for this query."]
        parts.append("Reason: {0}".format("; ".join(issues)))
        if validation_result.clarification_needed and validation_result.clarification_question:
            parts.append("Clarification needed: {0}".format(validation_result.clarification_question))
        return " ".join(parts)

    def _fallback_answer(
        self,
        results: List[ToolResult],
        citations: List[Citation],
        warnings: List[str],
        has_chart: bool,
        language: str,
    ) -> str:
        successful_answers = [self._clean_text(result.answer) for result in results if result.success and result.answer]
        successful_answers = [answer for answer in successful_answers if answer]

        if language == "hi-en":
            is_document_answer = self._has_document_answer(results)
            if successful_answers:
                if is_document_answer:
                    parts = [" ".join(successful_answers)]
                else:
                    parts = ["Yeh verified result hai: {0}".format(" ".join(successful_answers))]
            else:
                parts = ["Yeh response verified tool output se banaya gaya hai."]
            if has_chart:
                parts.append("Chart UI ke liye attached hai.")
            if citations:
                parts.append("Sources neeche attached hain.")
            if warnings:
                parts.append("Warnings: {0}".format("; ".join(warnings)))
            return self._join_response_parts(parts)

        if language == "es":
            parts = successful_answers or ["Aquí está el resultado verificado de las herramientas disponibles."]
            if has_chart:
                parts.append("El gráfico está adjunto.")
            if citations:
                parts.append("Las fuentes están adjuntas abajo.")
            if warnings:
                parts.append("Advertencias: {0}".format("; ".join(warnings)))
            return self._join_response_parts(parts)

        if successful_answers:
            parts = [" ".join(successful_answers)]
        else:
            parts = ["Here is the verified result from the available tool output."]
        if has_chart:
            parts.append("The chart is attached for the UI.")
        if citations:
            parts.append("Sources are attached below.")
        if warnings:
            parts.append("Warnings: {0}".format("; ".join(warnings)))
        return self._join_response_parts(parts)

    def _gemini_narration(
        self,
        original_query: str,
        rewritten_query: str,
        query_plan: QueryPlan,
        tool_results: List[ToolResult],
        validation_result: ValidationResult,
        citations: List[Citation],
        warnings: List[str],
        language: str,
        fallback_answer: str,
    ) -> str:
        if not self._gemini_available():
            return ""
        try:
            payload = self._verified_payload(
                original_query,
                rewritten_query,
                query_plan,
                tool_results,
                validation_result,
                citations,
                warnings,
                language,
            )
            prompt = self._narration_prompt(payload)
            answer = self._clean_text(self.gemini_client.generate(prompt))
            if not answer:
                return ""
            if self._contains_unverified_numbers(answer, payload, fallback_answer):
                return ""
            return answer
        except Exception:
            return ""

    def _verified_payload(
        self,
        original_query: str,
        rewritten_query: str,
        query_plan: QueryPlan,
        tool_results: List[ToolResult],
        validation_result: ValidationResult,
        citations: List[Citation],
        warnings: List[str],
        language: str,
    ) -> Dict[str, Any]:
        tool_payloads = []
        for result in tool_results:
            tool_payloads.append(
                {
                    "tool_name": result.tool_name,
                    "success": result.success,
                    "answer": result.answer,
                    "data": self._safe_jsonable(result.data),
                    "table": self._table_preview(result.table),
                    "chart_available": result.chart is not None,
                    "confidence": result.confidence,
                    "warnings": list(result.warnings or []),
                    "error_msg": result.error_msg,
                }
            )

        return {
            "original_query": original_query,
            "rewritten_query": rewritten_query,
            "intent": query_plan.intent,
            "language": language,
            "validation": {
                "is_valid": validation_result.is_valid,
                "confidence": validation_result.confidence,
                "warnings": list(validation_result.warnings or []),
                "issues": list(validation_result.issues or []),
            },
            "tool_results": tool_payloads,
            "citations": [self._citation_payload(citation) for citation in citations[:MAX_PROMPT_CITATIONS]],
            "warnings": warnings,
        }

    def _narration_prompt(self, payload: Dict[str, Any]) -> str:
        return (
            "You are the response narrator for a financial intelligence system.\n"
            "Use only the verified payload below. Do not add numbers, table values, citations, facts, or assumptions.\n"
            "Do not calculate anything. Keep the answer concise and user-facing.\n"
            "For document/RAG answers, rewrite the retrieved evidence into polished prose instead of copying raw chunks.\n"
            "When intent is summarize_document, synthesize a complete summary from summary_context and cover the full document, not only its opening.\n"
            "Remove OCR/PDF formatting artifacts such as divider lines, repeated punctuation, resume separators, page noise, and broken layout text.\n"
            "If language is hi-en, write in natural Hinglish. If language is en, write in English. If language is es, write in professional Spanish and preserve financial terminology accurately.\n"
            "Do not invent citation labels; citations are attached separately by the app.\n\n"
            "VERIFIED_PAYLOAD:\n{0}".format(json.dumps(payload, ensure_ascii=True, default=str))
        )

    def _contains_unverified_numbers(self, answer: str, payload: Dict[str, Any], fallback_answer: str) -> bool:
        answer_numbers = set(_number_tokens(answer))
        if not answer_numbers:
            return False
        verified_text = "{0} {1}".format(json.dumps(payload, ensure_ascii=True, default=str), fallback_answer)
        verified_numbers = set(_number_tokens(verified_text))
        return bool(answer_numbers.difference(verified_numbers))

    def _gemini_available(self) -> bool:
        try:
            return bool(self.gemini_client and self.gemini_client.is_available())
        except Exception:
            return False

    def _metadata(
        self,
        language: str,
        query_plan: QueryPlan,
        source_selection: Optional[SourceSelection],
        execution_plan: Optional[ExecutionPlan],
        results: List[ToolResult],
        validation_result: ValidationResult,
        used_gemini: bool,
    ) -> Dict[str, Any]:
        return {
            "language": language,
            "intent": query_plan.intent,
            "selected_source_id": getattr(source_selection, "selected_source_id", None),
            "source_type": getattr(source_selection, "source_type", None),
            "tool_names": [result.tool_name for result in results],
            "successful_tool_names": [result.tool_name for result in results if result.success],
            "planned_tool_names": [call.tool_name for call in execution_plan.tool_calls] if execution_plan else [],
            "validation_is_valid": validation_result.is_valid,
            "validation_issues": list(validation_result.issues or []),
            "used_gemini": used_gemini,
        }

    def _table_preview(self, table: Any) -> Any:
        if table is None:
            return None
        if isinstance(table, list):
            return table[:MAX_PROMPT_TABLE_ROWS]
        if hasattr(table, "head"):
            try:
                return table.head(MAX_PROMPT_TABLE_ROWS).to_dict(orient="records")
            except Exception:
                return str(table)
        return table

    def _safe_jsonable(self, value: Any) -> Any:
        try:
            json.dumps(value, default=str)
            return value
        except Exception:
            return str(value)

    def _citation_payload(self, citation: Citation) -> Dict[str, Any]:
        return {
            "source_id": citation.source_id,
            "filename": citation.filename,
            "page": citation.page,
            "chunk_id": citation.chunk_id,
            "text_snippet": citation.text_snippet,
        }

    def _citation_summary(self, citations: List[Citation]) -> str:
        labels = []
        for citation in citations[:MAX_PROMPT_CITATIONS]:
            label = citation.filename or citation.source_id or "source"
            if citation.page is not None:
                label = "{0} page {1}".format(label, citation.page)
            if citation.chunk_id:
                label = "{0} chunk {1}".format(label, citation.chunk_id)
            labels.append(label)
        return "; ".join(labels)

    def _has_document_answer(self, results: List[ToolResult]) -> bool:
        return any(result.success and result.tool_name in DOCUMENT_TOOL_NAMES for result in results)

    def _join_response_parts(self, parts: List[str]) -> str:
        answer = ""
        for part in [item for item in parts if item]:
            if not answer:
                answer = part
            elif "\n" in answer:
                answer = "{0}\n\n{1}".format(answer.rstrip(), part)
            else:
                answer = "{0} {1}".format(answer, part)
        return answer

    def _clean_text(self, text: Optional[str]) -> str:
        if not text:
            return ""
        cleaned = str(text).strip()
        cleaned = re.sub(r"^```(?:text|markdown)?", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
        return clean_document_text(cleaned)

    def _dedupe(self, values: List[str]) -> List[str]:
        seen = set()
        deduped = []
        for value in values:
            if value in seen:
                continue
            seen.add(value)
            deduped.append(value)
        return deduped
