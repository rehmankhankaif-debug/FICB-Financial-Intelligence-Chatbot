from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from src.agents.confidence import normalize_confidence
from src.agents.semantic_meaning import SemanticMeaningExtractor
from src.llm.gemini_client import GeminiClient
from src.llm.prompts import build_query_plan_prompt
from src.models.query import QueryPlan, RewrittenQuery


ALLOWED_INTENTS = {
    "table_analysis",
    "chart_request",
    "summarize_document",
    "compare_documents",
    "rag_question",
    "url_lookup",
    "general_finance",
}

METRIC_GROUPS = {
    "age": {"age"},
    "price": {"price", "premium", "priced"},
    "mileage": {"mileage", "km", "km_driven", "kilometers", "odometer"},
    "profit": {"profit", "margin", "gain"},
    "revenue": {"revenue", "sales", "income", "earnings"},
    "runs": {"runs", "run", "score", "batsman_runs"},
    "strike_rate": {"strike", "strike_rate", "rate", "sr"},
    "count": {"count", "dominant", "dominates", "frequency", "frequent", "how many", "kitni", "kitne", "number", "quantity", "qty", "volume"},
}

AGGREGATION_GROUPS = {
    "mean": {"average", "avg", "mean", "monthly"},
    "max": {"maximum", "max", "highest"},
    "min": {"minimum", "min", "lowest", "bottom", "weakest"},
    "count": {"common", "count", "dominant", "dominates", "frequency", "frequent", "kitni", "kitne", "how many", "most", "number", "preferred", "quantity", "qty", "volume"},
    "sum": {"contribute", "contributes", "contribution", "sum", "total"},
}

DOCUMENT_SIGNALS = {"report", "document", "pdf", "docx", "annual"}
SUMMARY_SIGNALS = {"outline", "summarise", "summarize", "summary", "key points"}
CHART_SIGNALS = {"chart", "graph", "plot", "bar graph", "bar"}
COMPARISON_SIGNALS = {"compare", "comparison", "versus", "vs"}
URL_SIGNALS = {"http://", "https://", "url", "link", "online"}
DATA_REFERENCE_SIGNALS = {"csv", "data", "dataset", "file", "table", "uploaded"}
TABLE_ANALYSIS_SIGNALS = {
    "average",
    "avg",
    "break down",
    "break that down",
    "breakdown",
    "common",
    "compare",
    "count",
    "distribution",
    "dominant",
    "dominates",
    "executive summary",
    "frequency",
    "frequent",
    "highest",
    "how many",
    "insight",
    "insights",
    "kitne",
    "kitni",
    "lowest",
    "mean",
    "most",
    "number",
    "pattern",
    "patterns",
    "preferred",
    "preference",
    "relationship",
    "related",
    "segment",
    "segmentation",
    "top",
    "trend",
    "trends",
    "unusual",
    "volume",
}
FINANCE_SURVEY_SIGNALS = {
    "age",
    "avenue",
    "bond",
    "debenture",
    "duration",
    "equity",
    "expected return",
    "fixed deposit",
    "gender",
    "gold",
    "investment",
    "monitor",
    "mutual fund",
    "objective",
    "ppf",
    "purpose",
    "saving",
    "savings",
    "source",
    "stock market",
}
TABLE_INSIGHT_SIGNALS = {
    "analyze data",
    "analyze dataset",
    "business insights",
    "data insight",
    "data insights",
    "data summary",
    "dataset insight",
    "dataset insights",
    "dataset summary",
    "executive summary",
    "hidden patterns",
    "interesting findings",
    "key finding",
    "key findings",
    "key insight",
    "key insights",
    "main findings",
    "overview",
    "risks and opportunities",
    "summary of this dataset",
    "summarize data",
    "summarize dataset",
}
TABLE_FILE_TYPES = {"csv", "xlsx", "xls"}
DOCUMENT_FILE_TYPES = {"pdf", "docx", "txt", "html", "url"}


def _model_payload(model: QueryPlan) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _tokens(text: str) -> List[str]:
    return [token for token in re.split(r"[^a-zA-Z0-9_]+", text.lower()) if token]


def _term_present(term: str, text: str, token_set: set) -> bool:
    normalized = term.lower().strip()
    if not normalized:
        return False
    if " " in normalized:
        return re.search(r"\b{0}\b".format(re.escape(normalized)), text) is not None
    if "_" in normalized:
        return normalized in token_set or normalized.replace("_", " ") in text
    return normalized in token_set


def _metric_name(metric: Any) -> str:
    if isinstance(metric, dict):
        return str(metric.get("name") or metric.get("text") or metric.get("metric") or "")
    return str(metric or "")


class QueryPlannerAgent:
    def __init__(self, gemini_client: Optional[GeminiClient] = None) -> None:
        self.gemini_client = gemini_client or GeminiClient()
        self.semantic_extractor = SemanticMeaningExtractor()

    def plan(
        self,
        query: str,
        rewritten_query: RewrittenQuery,
        available_sources: Optional[List[Any]] = None,
        table_profiles: Optional[List[Any]] = None,
    ) -> QueryPlan:
        try:
            sources = available_sources or []
            profiles = table_profiles or []
            semantic = self.semantic_extractor.extract(query, rewritten_query.rewritten_query, profiles)
            fallback_model = self._fallback_plan(query, rewritten_query, sources, semantic)
            fallback = _model_payload(fallback_model)

            if self.gemini_client.is_available():
                prompt = build_query_plan_prompt(
                    query,
                    self._rewritten_payload(rewritten_query),
                    self._source_payloads(sources),
                    self._table_profile_payloads(profiles),
                )
                payload = self.gemini_client.generate_json(prompt, fallback=fallback)
                plan = self._from_payload(payload, fallback)
                return self._stabilize_plan(plan, query, rewritten_query, sources, semantic)
            return self._stabilize_plan(fallback_model, query, rewritten_query, sources, semantic)
        except Exception as exc:
            return QueryPlan(
                original_query=query or "",
                rewritten_query=getattr(rewritten_query, "rewritten_query", "") or query or "",
                language=getattr(rewritten_query, "language", "en"),
                intent="general_finance",
                confidence=0.0,
                clarification_needed=True,
                clarification_question="Could you rephrase your question?",
                reasoning_short="Query planning failed safely: {0}".format(str(exc)),
            )

    def _from_payload(self, payload: Dict[str, Any], fallback: Dict[str, Any]) -> QueryPlan:
        merged = dict(fallback)
        if isinstance(payload, dict):
            merged.update({key: value for key, value in payload.items() if value is not None})
        merged["intent"] = merged.get("intent") if merged.get("intent") in ALLOWED_INTENTS else fallback.get("intent", "general_finance")
        merged["confidence"] = normalize_confidence(merged.get("confidence", 0.0))
        merged["entities"] = self._normalize_items(merged.get("entities"), "entity")
        merged["metrics"] = self._normalize_items(merged.get("metrics"), "metric")
        merged["filters"] = self._normalize_items(merged.get("filters"), "filter")
        merged["aggregations"] = self._normalize_items(merged.get("aggregations"), "aggregation")
        merged["grouping"] = self._normalize_string_list(merged.get("grouping"))
        merged["sorting"] = merged.get("sorting") if isinstance(merged.get("sorting"), dict) else {}
        merged["comparison"] = merged.get("comparison") if isinstance(merged.get("comparison"), dict) else {}
        raw_chart_types = merged.get("chart_types") or merged.get("chart_type")
        merged["chart_types"] = self._normalize_chart_types(raw_chart_types)
        merged["chart_type"] = merged["chart_types"][0] if merged["chart_types"] else None
        merged["chart_requested"] = bool(merged.get("chart_requested") or merged["chart_types"])
        return QueryPlan(**merged)

    def _fallback_plan(
        self,
        query: str,
        rewritten_query: RewrittenQuery,
        available_sources: List[Any],
        semantic: Optional[Dict[str, Any]] = None,
    ) -> QueryPlan:
        original = query or ""
        rewritten = rewritten_query.rewritten_query or original
        text = "{0} {1}".format(original, rewritten).lower()
        token_set = set(_tokens(text))
        semantic = semantic or {}

        table_insight_requested = self._is_table_insight_query(text, available_sources)
        metrics = self._extract_metrics(text, token_set)
        if (table_insight_requested or self._is_contextual_table_summary_query(text, available_sources)) and not metrics:
            metrics.append({"name": "dataset_summary", "confidence": 0.82})
        aggregations = self._extract_aggregations(text)
        grouping = self._extract_grouping(text, token_set)
        entities = self._extract_entities(text)
        filters = self._extract_filters(text, token_set)
        trend_visual_requested = (
            any(_term_present(term, text, token_set) for term in ["trend", "trends"])
            and not any(_term_present(term, text, token_set) for term in ["anomaly", "anomalies", "unusual"])
            and self._has_source_type(available_sources, TABLE_FILE_TYPES)
        )
        chart_requested = any(signal in text for signal in CHART_SIGNALS) or "pie" in text or trend_visual_requested
        chart_types = self._extract_chart_types(text)
        if trend_visual_requested and not chart_types:
            chart_types = ["line"]
        chart_type = chart_types[0] if chart_types else None
        metrics = self._merge_items(metrics, semantic.get("metrics"))
        aggregations = self._merge_items(aggregations, semantic.get("aggregations"))
        grouping = self._merge_strings(grouping, semantic.get("grouping"))
        entities = self._merge_items(entities, semantic.get("entities"))
        filters = self._merge_items(filters, semantic.get("filters"))
        chart_requested = chart_requested or bool(semantic.get("chart_requested"))
        chart_types = self._merge_strings(
            chart_types,
            self._normalize_chart_types(semantic.get("chart_types") or semantic.get("chart_type")),
        )
        chart_type = chart_types[0] if chart_types else (chart_type or semantic.get("chart_type"))

        if self._is_mixed_compare_query(text, available_sources):
            intent = "compare_documents"
        else:
            intent = semantic.get("intent") or self._infer_intent(
                text,
                metrics,
                aggregations,
                grouping,
                entities,
                chart_requested,
                table_insight_requested,
                available_sources,
            )
        required_source_type = self._required_source_type(intent)
        comparison = semantic.get("comparison") or self._extract_comparison(text, intent)
        confidence = self._fallback_confidence(intent, metrics, aggregations, grouping, entities, available_sources)
        if semantic.get("confidence"):
            confidence = max(confidence, normalize_confidence(semantic.get("confidence")))
        if self._is_contextual_document_summary_query(text, available_sources):
            confidence = max(confidence, 0.76)
        if self._is_contextual_table_summary_query(text, available_sources):
            confidence = max(confidence, 0.72)

        clarification_needed = False
        clarification_question = None
        if not original.strip():
            clarification_needed = True
            clarification_question = "What would you like to analyze?"
            confidence = 0.0
        elif confidence < 0.35:
            clarification_needed = True
            clarification_question = "Could you clarify which data or document you want me to use?"

        return QueryPlan(
            original_query=original,
            rewritten_query=rewritten,
            language=rewritten_query.language or "en",
            intent=intent,
            required_source_type=required_source_type,
            entities=entities,
            metrics=metrics,
            filters=filters,
            aggregations=aggregations,
            grouping=grouping,
            sorting=semantic.get("sorting") or self._extract_sorting(text),
            comparison=comparison,
            chart_requested=chart_requested,
            chart_type=chart_type,
            chart_types=chart_types,
            limit=self._extract_limit(text),
            confidence=confidence,
            clarification_needed=clarification_needed,
            clarification_question=clarification_question,
            reasoning_short="Deterministic planning fallback based on semantic signal groups.",
        )

    def _infer_intent(
        self,
        text: str,
        metrics: List[Dict[str, Any]],
        aggregations: List[Dict[str, Any]],
        grouping: List[str],
        entities: List[Dict[str, Any]],
        chart_requested: bool,
        table_insight_requested: bool = False,
        available_sources: Optional[List[Any]] = None,
    ) -> str:
        if any(signal in text for signal in URL_SIGNALS):
            return "url_lookup"
        if any(signal in text for signal in COMPARISON_SIGNALS) and ("csv" in text or "report" in text or "document" in text):
            return "compare_documents"
        if chart_requested:
            return "chart_request"
        if table_insight_requested:
            return "table_analysis"
        if self._is_contextual_document_summary_query(text, available_sources or []):
            return "summarize_document"
        if any(signal in text for signal in SUMMARY_SIGNALS) and any(signal in text for signal in DOCUMENT_SIGNALS):
            return "summarize_document"
        if any(signal in text for signal in DOCUMENT_SIGNALS) and self._has_source_type(available_sources or [], DOCUMENT_FILE_TYPES) and not self._has_source_type(available_sources or [], TABLE_FILE_TYPES):
            return "rag_question"
        if any(signal in text for signal in DOCUMENT_SIGNALS) and not metrics:
            return "rag_question"
        if self._is_contextual_table_summary_query(text, available_sources or []):
            return "table_analysis"
        if self._looks_like_table_question(text, metrics, aggregations, grouping, entities, available_sources or []):
            return "table_analysis"
        if metrics or "virat" in text or "manual" in text or "automatic" in text:
            return "table_analysis"
        return "general_finance"

    def _required_source_type(self, intent: str) -> Optional[str]:
        if intent in {"table_analysis", "chart_request"}:
            return "table"
        if intent in {"summarize_document", "rag_question", "url_lookup"}:
            return "document"
        if intent == "compare_documents":
            return "mixed"
        return None

    def _extract_metrics(self, text: str, token_set: set) -> List[Dict[str, Any]]:
        metrics = []
        for metric, terms in METRIC_GROUPS.items():
            if any(_term_present(term, text, token_set) for term in terms):
                metrics.append({"name": metric, "confidence": 0.78})
        if "manual" in token_set and "automatic" in token_set and not any(item["name"] == "count" for item in metrics):
            metrics.append({"name": "count", "confidence": 0.8})
        return metrics

    def _stabilize_plan(
        self,
        plan: QueryPlan,
        query: str,
        rewritten_query: RewrittenQuery,
        available_sources: List[Any],
        semantic: Optional[Dict[str, Any]] = None,
    ) -> QueryPlan:
        text = "{0} {1}".format(query or "", rewritten_query.rewritten_query or "").lower()
        token_set = set(_tokens(text))
        semantic = semantic or {}
        deterministic_metrics = self._extract_metrics(text, token_set)
        deterministic_aggregations = self._extract_aggregations(text)
        deterministic_grouping = self._extract_grouping(text, token_set)
        deterministic_filters = self._extract_filters(text, token_set)
        deterministic_sorting = self._extract_sorting(text)
        deterministic_chart_types = self._extract_chart_types(text)
        plan.metrics = self._merge_items(plan.metrics, semantic.get("metrics"))
        plan.aggregations = self._merge_items(plan.aggregations, semantic.get("aggregations"))
        plan.grouping = self._merge_strings(plan.grouping, semantic.get("grouping"))
        plan.entities = self._merge_items(plan.entities, semantic.get("entities"))
        plan.filters = self._merge_items(plan.filters, semantic.get("filters"))
        plan.chart_types = self._merge_strings(
            self._normalize_chart_types(plan.chart_types or plan.chart_type),
            self._normalize_chart_types(semantic.get("chart_types") or semantic.get("chart_type")),
        )
        plan.chart_types = self._merge_strings(plan.chart_types, deterministic_chart_types)
        plan.chart_type = plan.chart_types[0] if plan.chart_types else plan.chart_type
        plan.chart_requested = bool(plan.chart_requested or semantic.get("chart_requested") or plan.chart_types)
        if semantic.get("comparison") and not plan.comparison:
            plan.comparison = semantic["comparison"]
        if not plan.sorting and semantic.get("sorting"):
            plan.sorting = semantic["sorting"]
        if semantic.get("chart_requested"):
            plan.chart_requested = True
            plan.chart_type = plan.chart_type or semantic.get("chart_type")
        if plan.chart_requested and self._has_source_type(available_sources, TABLE_FILE_TYPES):
            plan.intent = "chart_request"
            plan.required_source_type = "table"
            if not plan.metrics:
                plan.metrics = deterministic_metrics or [{"name": "count", "confidence": 0.8}]
            if not plan.aggregations:
                plan.aggregations = deterministic_aggregations or [{"operation": "count", "confidence": 0.8}]
            if any(term in text for term in ("trend", "trends")):
                trend_aggregation = self._trend_aggregation(plan.metrics)
                aggregation_names = {
                    str(item.get("operation") or item.get("agg") or "").lower()
                    for item in plan.aggregations
                    if isinstance(item, dict)
                }
                if trend_aggregation and (not aggregation_names or aggregation_names == {"count"}):
                    plan.aggregations = [{"operation": trend_aggregation, "confidence": 0.9}]
            if not plan.grouping:
                plan.grouping = deterministic_grouping
            if not plan.grouping and any(term in text for term in ("trend", "trends")):
                entity_fields = [
                    str(item.get("field") or "")
                    for item in plan.entities
                    if isinstance(item, dict) and item.get("field")
                ]
                if entity_fields and len(set(entity_fields)) == 1:
                    plan.grouping = [entity_fields[0]]
            if not plan.filters:
                plan.filters = deterministic_filters
            if not plan.sorting:
                plan.sorting = deterministic_sorting
            plan.clarification_needed = False
            plan.clarification_question = None
            plan.confidence = max(normalize_confidence(plan.confidence), 0.84)
            plan.reasoning_short = "Chart query stabilized to pandas table analysis followed by deterministic chart rendering."
        if self._is_mixed_compare_query(text, available_sources):
            plan.intent = "compare_documents"
            plan.required_source_type = "mixed"
            plan.comparison = self._extract_comparison(text, "compare_documents")
            plan.clarification_needed = False
            plan.clarification_question = None
            plan.confidence = max(normalize_confidence(plan.confidence), 0.78)
        if plan.intent in {"", "general_finance"} and semantic.get("intent"):
            plan.intent = semantic["intent"]
            plan.required_source_type = semantic.get("required_source_type") or self._required_source_type(plan.intent)
            plan.confidence = max(normalize_confidence(plan.confidence), normalize_confidence(semantic.get("confidence", 0.0)))
            plan.reasoning_short = semantic.get("reasoning_short") or plan.reasoning_short
        should_force_table = plan.intent in {"", "general_finance"} or (
            plan.clarification_needed and not plan.required_source_type
        )
        if should_force_table and (
            self._is_contextual_table_summary_query(text, available_sources) or self._is_table_insight_query(text, available_sources) or self._looks_like_table_question(
                text,
                plan.metrics or deterministic_metrics,
                plan.aggregations or deterministic_aggregations,
                plan.grouping or deterministic_grouping,
                plan.entities,
                available_sources,
            )
        ):
            plan.intent = "table_analysis"
            plan.required_source_type = "table"
            if not plan.metrics:
                plan.metrics = deterministic_metrics
            if not plan.aggregations:
                plan.aggregations = deterministic_aggregations
            if not plan.grouping:
                plan.grouping = deterministic_grouping
            if not plan.filters:
                plan.filters = deterministic_filters
            if not plan.sorting:
                plan.sorting = deterministic_sorting
            if self._is_table_insight_query(text, available_sources) and not any(_metric_name(item) == "dataset_summary" for item in plan.metrics):
                plan.metrics.append({"name": "dataset_summary", "confidence": 0.82})
            if self._is_contextual_table_summary_query(text, available_sources) and not any(_metric_name(item) == "dataset_summary" for item in plan.metrics):
                plan.metrics.append({"name": "dataset_summary", "confidence": 0.82})
            plan.clarification_needed = False
            plan.clarification_question = None
            plan.confidence = max(normalize_confidence(plan.confidence), 0.72)
            plan.reasoning_short = "Data query stabilized to table analysis because a table source is available."
        should_force_document = plan.intent in {"", "general_finance"} or (
            plan.clarification_needed and not plan.required_source_type
        )
        if should_force_document and self._is_contextual_document_summary_query(text, available_sources):
            plan.intent = "summarize_document"
            plan.required_source_type = "document"
            plan.clarification_needed = False
            plan.clarification_question = None
            plan.confidence = max(normalize_confidence(plan.confidence), 0.76)
            plan.reasoning_short = "Summary request stabilized to document summarization because a document source is available."
        return plan

    def _trend_aggregation(self, metrics: List[Dict[str, Any]]) -> Optional[str]:
        names = {_metric_name(item) for item in metrics or []}
        if names.intersection({"revenue", "sales", "profit", "income", "expenses", "cost", "runs"}):
            return "sum"
        if names.intersection({"price", "margin", "strike_rate", "mileage", "rate"}):
            return "mean"
        return None

    def _is_table_insight_query(self, text: str, available_sources: List[Any]) -> bool:
        if not self._has_source_type(available_sources, TABLE_FILE_TYPES):
            return False
        has_table_signal = any(signal in text for signal in TABLE_INSIGHT_SIGNALS) or (
            "insight" in text and not any(signal in text for signal in DOCUMENT_SIGNALS)
        )
        if not has_table_signal:
            return False
        has_document_signal = any(signal in text for signal in DOCUMENT_SIGNALS)
        has_document_source = self._has_source_type(available_sources, DOCUMENT_FILE_TYPES)
        return not (has_document_signal and has_document_source)

    def _has_source_type(self, available_sources: List[Any], file_types: set) -> bool:
        for source in available_sources or []:
            payload = source if isinstance(source, dict) else self._source_payloads([source])[0]
            file_type = str(payload.get("file_type") or payload.get("source_type") or "").lower()
            metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
            source_category = str(payload.get("source_category") or payload.get("category") or metadata.get("source_category") or "").lower()
            if file_type in file_types:
                return True
            if file_types == TABLE_FILE_TYPES and source_category == "table":
                return True
            if file_types == DOCUMENT_FILE_TYPES and source_category == "document":
                return True
        return False

    def _is_contextual_document_summary_query(self, text: str, available_sources: List[Any]) -> bool:
        if not self._has_source_type(available_sources, DOCUMENT_FILE_TYPES):
            return False
        if not any(signal in text for signal in SUMMARY_SIGNALS):
            return False
        if any(signal in text for signal in URL_SIGNALS | COMPARISON_SIGNALS | CHART_SIGNALS):
            return False
        has_table_source = self._has_source_type(available_sources, TABLE_FILE_TYPES)
        has_table_signal = any(signal in text for signal in DATA_REFERENCE_SIGNALS | TABLE_ANALYSIS_SIGNALS | FINANCE_SURVEY_SIGNALS)
        return not (has_table_source and has_table_signal)

    def _is_contextual_table_summary_query(self, text: str, available_sources: List[Any]) -> bool:
        if not self._has_source_type(available_sources, TABLE_FILE_TYPES):
            return False
        if not any(signal in text for signal in SUMMARY_SIGNALS):
            return False
        return not self._has_source_type(available_sources, DOCUMENT_FILE_TYPES)

    def _is_mixed_compare_query(self, text: str, available_sources: List[Any]) -> bool:
        if not any(signal in text for signal in COMPARISON_SIGNALS):
            return False
        if not self._has_source_type(available_sources, TABLE_FILE_TYPES):
            return False
        if not self._has_source_type(available_sources, DOCUMENT_FILE_TYPES):
            return False
        return any(signal in text for signal in {"annual", "docx", "document", "pdf", "report"}) or (
            "csv" in text and any(signal in text for signal in {"document", "pdf", "report"})
        )

    def _extract_aggregations(self, text: str) -> List[Dict[str, Any]]:
        aggregations = []
        token_set = set(_tokens(text))
        for aggregation, terms in AGGREGATION_GROUPS.items():
            if any(_term_present(term, text, token_set) for term in terms):
                aggregations.append({"operation": aggregation, "confidence": 0.78})
        return aggregations

    def _extract_grouping(self, text: str, token_set: set) -> List[str]:
        grouping = []
        if "monthly" in text or "month" in token_set:
            grouping.append("month")
        if "manual" in token_set and "automatic" in token_set:
            grouping.append("transmission")
        if "quarter" in token_set:
            grouping.append("quarter")
        if "gender" in token_set or "male" in token_set or "female" in token_set:
            grouping.append("gender")
        if "avenue" in token_set or "avenues" in token_set or "preferred" in token_set or "preference" in token_set:
            grouping.append("avenue")
        if "objective" in token_set or "objectives" in token_set:
            grouping.append("objective")
        if "purpose" in token_set:
            grouping.append("purpose")
        if "duration" in token_set:
            grouping.append("duration")
        if "expect" in token_set or "expected" in token_set or ("return" in token_set and "expected" in text):
            grouping.append("expected return")
        if "monitor" in token_set or "monitoring" in token_set:
            grouping.append("investment monitor")
        if "source" in token_set and ("finance" in text or "financial" in text or "information" in text):
            grouping.append("source")
        return grouping

    def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        entities = []
        if "virat" in text:
            entities.append({"text": "Virat", "normalized": "Virat Kohli", "type": "person", "confidence": 0.85})
        if "rcb" in text or "royal challengers bangalore" in text:
            entities.append({"text": "RCB", "normalized": "Royal Challengers Bangalore", "type": "team", "confidence": 0.82})
        if "manual" in text:
            entities.append(
                {
                    "text": "manual",
                    "normalized": "Manual",
                    "type": "category_value",
                    "field": "transmission",
                    "confidence": 0.82,
                }
            )
        if "automatic" in text:
            entities.append(
                {
                    "text": "automatic",
                    "normalized": "Automatic",
                    "type": "category_value",
                    "field": "transmission",
                    "confidence": 0.82,
                }
            )
        return entities

    def _extract_filters(self, text: str, token_set: set) -> List[Dict[str, Any]]:
        filters: List[Dict[str, Any]] = []
        if "stock market" in text or "stock_market" in text or ("stock" in token_set and "market" in token_set):
            filters.append({"field": "stock market", "operator": "equals", "value": "Yes", "confidence": 0.82})
        return filters

    def _extract_sorting(self, text: str) -> Dict[str, Any]:
        if any(signal in text for signal in {"bottom", "lowest", "minimum", "weakest"}):
            return {"direction": "asc"}
        if any(signal in text for signal in {"common", "contribute", "dominant", "frequent", "highest", "maximum", "most", "preferred", "top"}):
            return {"direction": "desc"}
        return {}

    def _extract_chart_types(self, text: str) -> List[str]:
        chart_types: List[str] = []
        token_set = set(_tokens(text))
        for chart_type, signals in [
            ("bar", {"bar", "bar graph", "bar chart"}),
            ("pie", {"pie", "pie graph", "pie chart"}),
            ("line", {"line graph", "line chart"}),
            ("scatter", {"scatter", "scatter plot"}),
            ("histogram", {"histogram"}),
        ]:
            if any(_term_present(signal, text, token_set) for signal in signals):
                chart_types.append(chart_type)
        if any(signal in text for signal in CHART_SIGNALS) and not chart_types:
            chart_types.append("bar")
        return chart_types

    def _normalize_chart_types(self, value: Any) -> List[str]:
        raw_values = value if isinstance(value, (list, tuple, set)) else [value]
        aliases = {
            "bar": "bar",
            "bar_chart": "bar",
            "bar_graph": "bar",
            "pie": "pie",
            "pie_chart": "pie",
            "pie_graph": "pie",
            "line": "line",
            "line_chart": "line",
            "line_graph": "line",
            "scatter": "scatter",
            "scatter_chart": "scatter",
            "scatter_plot": "scatter",
            "histogram": "histogram",
        }
        normalized: List[str] = []
        for item in raw_values:
            key = str(item or "").strip().lower().replace(" ", "_")
            chart_type = aliases.get(key)
            if chart_type and chart_type not in normalized:
                normalized.append(chart_type)
        return normalized

    def _extract_comparison(self, text: str, intent: str) -> Dict[str, Any]:
        if intent != "compare_documents":
            return {}
        comparison_type = "revenue_trend" if "revenue" in text and "trend" in text else "cross_source_comparison"
        return {"type": comparison_type, "required_source_types": ["table", "document"]}

    def _extract_limit(self, text: str) -> Optional[int]:
        match = re.search(r"\btop\s+(\d+)\b", text)
        if match:
            return int(match.group(1))
        ranked_match = re.search(r"\b(\d+)\b", text)
        if ranked_match and any(signal in text for signal in {"bottom", "lowest", "top", "weakest"}):
            return int(ranked_match.group(1))
        word_numbers = {
            "one": 1,
            "two": 2,
            "three": 3,
            "four": 4,
            "five": 5,
            "six": 6,
            "seven": 7,
            "eight": 8,
            "nine": 9,
            "ten": 10,
        }
        word_match = re.search(r"\btop\s+({0})\b".format("|".join(word_numbers)), text)
        if word_match:
            return word_numbers[word_match.group(1)]
        ranked_word_match = re.search(r"\b({0})\b".format("|".join(word_numbers)), text)
        if ranked_word_match and any(signal in text for signal in {"bottom", "lowest", "top", "weakest"}):
            return word_numbers[ranked_word_match.group(1)]
        return None

    def _looks_like_table_question(
        self,
        text: str,
        metrics: List[Dict[str, Any]],
        aggregations: List[Dict[str, Any]],
        grouping: List[str],
        entities: List[Dict[str, Any]],
        available_sources: List[Any],
    ) -> bool:
        if not self._has_source_type(available_sources, TABLE_FILE_TYPES):
            return False

        has_data_reference = any(signal in text for signal in DATA_REFERENCE_SIGNALS)
        has_analysis_signal = any(signal in text for signal in TABLE_ANALYSIS_SIGNALS)
        has_finance_survey_signal = any(signal in text for signal in FINANCE_SURVEY_SIGNALS)
        has_structured_plan = bool(metrics or aggregations or grouping or entities)

        if has_data_reference and (has_analysis_signal or has_finance_survey_signal or has_structured_plan):
            return True
        if has_finance_survey_signal and (has_analysis_signal or aggregations or grouping):
            return True
        if has_structured_plan and has_analysis_signal:
            return True
        return False

    def _fallback_confidence(
        self,
        intent: str,
        metrics: List[Dict[str, Any]],
        aggregations: List[Dict[str, Any]],
        grouping: List[str],
        entities: List[Dict[str, Any]],
        available_sources: List[Any],
    ) -> float:
        score = 0.45
        if intent != "general_finance":
            score += 0.15
        if metrics:
            score += 0.12
        if aggregations:
            score += 0.08
        if grouping:
            score += 0.06
        if entities:
            score += 0.06
        if available_sources:
            score += 0.04
        return normalize_confidence(score)

    def _normalize_items(self, value: Any, key_name: str) -> List[Dict[str, Any]]:
        if not value:
            return []
        items = value if isinstance(value, list) else [value]
        normalized = []
        for item in items:
            if isinstance(item, dict):
                normalized.append(item)
            else:
                normalized.append({"name" if key_name in {"metric", "aggregation"} else "text": str(item)})
        return normalized

    def _normalize_string_list(self, value: Any) -> List[str]:
        if not value:
            return []
        if isinstance(value, list):
            return [str(item) for item in value]
        return [str(value)]

    def _merge_items(self, existing: Any, additions: Any) -> List[Dict[str, Any]]:
        merged: List[Dict[str, Any]] = []
        seen = set()
        for item in self._normalize_items(existing, "metric") + self._normalize_items(additions, "metric"):
            key = (
                str(item.get("name") or item.get("text") or item.get("field") or item.get("column") or "").lower(),
                str(item.get("normalized") or item.get("value") or item.get("operation") or "").lower(),
                str(item.get("operator") or "").lower(),
            )
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)
        return merged

    def _merge_strings(self, existing: Any, additions: Any) -> List[str]:
        merged: List[str] = []
        positions: Dict[str, int] = {}
        for item in self._normalize_string_list(existing) + self._normalize_string_list(additions):
            value = str(item).strip()
            key = value.casefold()
            if not value:
                continue
            if key not in positions:
                positions[key] = len(merged)
                merged.append(value)
            elif merged[positions[key]].islower() and not value.islower():
                # Prefer the source-profile spelling (for example ``Gender``)
                # over a generic deterministic hint such as ``gender``.
                merged[positions[key]] = value
        return merged

    def _rewritten_payload(self, rewritten_query: RewrittenQuery) -> Dict[str, Any]:
        if hasattr(rewritten_query, "model_dump"):
            return rewritten_query.model_dump()
        return rewritten_query.dict()

    def _source_payloads(self, available_sources: List[Any]) -> List[Dict[str, Any]]:
        payloads = []
        for source in available_sources:
            if isinstance(source, dict):
                payloads.append(source)
            elif hasattr(source, "model_dump"):
                payloads.append(source.model_dump())
            elif hasattr(source, "dict"):
                payloads.append(source.dict())
            else:
                payloads.append({"source": str(source)})
        return payloads

    def _table_profile_payloads(self, table_profiles: List[Any]) -> List[Dict[str, Any]]:
        payloads = []
        for profile in table_profiles or []:
            if isinstance(profile, dict):
                payload = dict(profile)
            elif hasattr(profile, "model_dump"):
                payload = profile.model_dump()
            elif hasattr(profile, "dict"):
                payload = profile.dict()
            else:
                continue
            payloads.append(
                {
                    "source_id": payload.get("source_id"),
                    "filename": payload.get("filename"),
                    "columns": payload.get("columns", []),
                    "normalized_columns": payload.get("normalized_columns", {}),
                    "numeric_columns": payload.get("numeric_columns", []),
                    "categorical_columns": payload.get("categorical_columns", []),
                    "entity_candidate_columns": payload.get("entity_candidate_columns", []),
                    "metric_candidate_columns": payload.get("metric_candidate_columns", []),
                    "sample_values": payload.get("sample_values", {}),
                    "semantic_summary": payload.get("semantic_summary", ""),
                }
            )
        return payloads
