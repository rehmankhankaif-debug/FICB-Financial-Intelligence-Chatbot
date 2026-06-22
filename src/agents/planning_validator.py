from __future__ import annotations

from typing import List, Optional

from src.agents.confidence import is_low_confidence, is_medium_confidence, normalize_confidence
from src.agents.query_planner import ALLOWED_INTENTS
from src.models.query import QueryPlan
from src.models.source import SourceSelection
from src.models.validation import ValidationResult


TABLE_INTENTS = {"table_analysis", "chart_request"}
DOCUMENT_INTENTS = {"summarize_document", "rag_question", "url_lookup"}
TABLE_SOURCE_TYPES = {"table", "csv", "xlsx", "xls"}
DOCUMENT_SOURCE_TYPES = {"document", "pdf", "docx", "txt", "html", "url"}


class PlanningValidator:
    def validate_plan(
        self,
        query_plan: QueryPlan,
        source_selection: Optional[SourceSelection] = None,
    ) -> ValidationResult:
        try:
            issues: List[str] = []
            warnings: List[str] = []
            clarification_needed = bool(getattr(query_plan, "clarification_needed", False))
            clarification_question = query_plan.clarification_question

            if query_plan.intent not in ALLOWED_INTENTS:
                issues.append("Invalid intent: {0}".format(query_plan.intent))

            confidence = normalize_confidence(query_plan.confidence)
            if is_low_confidence(confidence):
                warnings.append("Query plan confidence is low.")
                clarification_needed = True
                clarification_question = clarification_question or "Could you clarify what you want to analyze?"
            elif is_medium_confidence(confidence):
                warnings.append("Query plan confidence is medium; downstream validation should be cautious.")

            self._validate_source(query_plan, source_selection, issues, warnings)
            self._validate_chart(query_plan, warnings)
            self._validate_table_query(query_plan, issues)
            self._validate_document_query(query_plan, source_selection, issues)
            self._validate_entities_and_metrics(query_plan, warnings)

            is_valid = not issues and not (clarification_needed and confidence < 0.35)
            return ValidationResult(
                is_valid=is_valid,
                confidence=confidence,
                issues=issues,
                warnings=warnings,
                requires_retry=bool(issues),
                clarification_needed=clarification_needed,
                clarification_question=clarification_question,
            )
        except Exception as exc:
            return ValidationResult(
                is_valid=False,
                confidence=0.0,
                issues=["Planning validation failed safely: {0}".format(str(exc))],
                warnings=[],
                requires_retry=True,
                clarification_needed=True,
                clarification_question="Could you rephrase your request?",
            )

    def _validate_source(
        self,
        query_plan: QueryPlan,
        source_selection: Optional[SourceSelection],
        issues: List[str],
        warnings: List[str],
    ) -> None:
        if query_plan.intent == "general_finance":
            return

        if source_selection is None or not source_selection.selected_source_id:
            warnings.append("No source has been selected yet.")
            return

        source_type = (source_selection.source_type or "").lower()
        if query_plan.intent in TABLE_INTENTS and source_type not in TABLE_SOURCE_TYPES:
            issues.append("Table intent requires a table source.")
        if query_plan.intent in DOCUMENT_INTENTS and source_type not in DOCUMENT_SOURCE_TYPES:
            issues.append("Document intent requires a document source.")

    def _validate_chart(self, query_plan: QueryPlan, warnings: List[str]) -> None:
        if query_plan.chart_requested and not query_plan.chart_type:
            warnings.append("Chart was requested but chart_type is missing.")
        if query_plan.intent == "chart_request" and not query_plan.chart_requested:
            warnings.append("Intent is chart_request but chart_requested is false.")

    def _validate_table_query(self, query_plan: QueryPlan, issues: List[str]) -> None:
        if query_plan.intent not in TABLE_INTENTS:
            return
        if not any([query_plan.metrics, query_plan.entities, query_plan.filters, query_plan.grouping, query_plan.aggregations]):
            issues.append("Table query needs at least one metric, entity, filter, grouping, or aggregation.")

    def _validate_document_query(
        self,
        query_plan: QueryPlan,
        source_selection: Optional[SourceSelection],
        issues: List[str],
    ) -> None:
        if query_plan.intent not in DOCUMENT_INTENTS:
            return
        if source_selection is not None and source_selection.selected_source_id and source_selection.source_type not in DOCUMENT_SOURCE_TYPES:
            issues.append("Document query selected a non-document source.")

    def _validate_entities_and_metrics(self, query_plan: QueryPlan, warnings: List[str]) -> None:
        for entity in query_plan.entities:
            if isinstance(entity, dict) and not (entity.get("text") or entity.get("name") or entity.get("normalized")):
                warnings.append("A query entity is missing text or normalized value.")
        for metric in query_plan.metrics:
            if isinstance(metric, dict) and not (metric.get("name") or metric.get("text")):
                warnings.append("A query metric is missing a name.")
