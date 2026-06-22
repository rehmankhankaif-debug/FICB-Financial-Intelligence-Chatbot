from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from src.agents.query_planner import QueryPlannerAgent
from src.agents.query_rewriter import QueryRewriterAgent
from src.agents.source_selector import SourceSelector
from src.agents.tool_planner import ToolPlannerAgent
from src.evaluation.metrics import (
    citation_presence_score,
    contains_all_score,
    exact_match,
    hallucination_risk_score,
    pass_fail,
    sequence_match_score,
    source_selection_score,
    table_value_score,
)
from src.llm.gemini_client import GeminiClient
from src.models.table import TableProfile
from src.models.tool import ToolResult
from src.table_intelligence.pandas_executor import PandasExecutor
from src.tools.compare_tool import CompareTool
from src.tools.manager import ToolManager
from src.tools.rag_qa_tool import RagQATool


METRIC_KEYS = [
    "query_rewrite_quality",
    "intent_accuracy",
    "semantic_plan_accuracy",
    "source_selection_accuracy",
    "tool_selection_accuracy",
    "csv_answer_accuracy",
    "rag_citation_presence",
    "hallucination_safety",
    "document_comparison_accuracy",
    "error_handling",
]


@dataclass
class EvaluationCaseResult:
    case_id: str
    query: str
    passed: bool
    scores: Dict[str, float] = field(default_factory=dict)
    failures: List[str] = field(default_factory=list)
    actual: Dict[str, Any] = field(default_factory=dict)
    expected: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationReport:
    total_cases: int
    passed_cases: int
    accuracy_percentage: float
    metric_accuracy: Dict[str, float]
    failed_cases: List[Dict[str, Any]]
    improvement_suggestions: List[str]
    cases: List[EvaluationCaseResult]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_cases": self.total_cases,
            "passed_cases": self.passed_cases,
            "accuracy_percentage": self.accuracy_percentage,
            "metric_accuracy": self.metric_accuracy,
            "failed_cases": self.failed_cases,
            "improvement_suggestions": self.improvement_suggestions,
            "cases": [
                {
                    "case_id": item.case_id,
                    "query": item.query,
                    "passed": item.passed,
                    "scores": item.scores,
                    "failures": item.failures,
                    "actual": item.actual,
                    "expected": item.expected,
                }
                for item in self.cases
            ],
        }

    def to_markdown(self) -> str:
        lines = [
            "# Evaluation Report",
            "",
            "> This is a reproducible controlled benchmark, not a universal production-accuracy guarantee. Real annual reports, browser/device coverage, load, security, OCR, and deployment evaluation remain separate gates.",
            "",
            "## Assignment Requirement Coverage",
            "",
            "The project covers the assignment's core functional requirements:",
            "",
            "- CSV support: implemented through table loading, profiling, semantic mapping, and pandas execution.",
            "- Excel support: implemented through pandas and `openpyxl`.",
            "- PDF support: implemented through PyMuPDF/`pypdf`, OCR fallback, chunking, embeddings, ChromaDB, retrieval, and citations.",
            "- DOCX support: implemented through `python-docx`, including paragraph and table extraction.",
            "- URL support: implemented through `requests`, BeautifulSoup cleaning, chunking, indexing, and retrieval.",
            "- Natural language and Hinglish support: query rewrite, planning, safe language detection, and deterministic fallbacks.",
            "- Financial and statistical analysis: sum, mean, median, min, max, count, nunique, ranking, filtering, and cited cross-document numeric comparisons.",
            "- Charts: Plotly chart generation from pandas-grounded table results.",
            "- Summaries and RAG QA: document chunk retrieval with citation-bearing answers.",
            "- Autonomous tool invocation: `QueryPlan -> ToolPlanner -> ExecutionPlan -> ToolChainExecutor`.",
            "- Chat history, export, logging, observability, tests, and documentation are included.",
            "",
            "## Benchmark Results",
            "",
            "Latest evaluation:",
            "",
            "- Total cases: {0}".format(self.total_cases),
            "- Passed cases: {0}".format(self.passed_cases),
            "- Accuracy: {0:.2f}%".format(self.accuracy_percentage),
            "",
            "## Metric Accuracy",
            "",
        ]
        for metric, score in sorted(self.metric_accuracy.items()):
            lines.append("- {0}: {1:.2f}%".format(metric, score))
        lines.extend(["", "## Failed Cases", ""])
        if not self.failed_cases:
            lines.append("No failed cases.")
        for item in self.failed_cases:
            lines.append("- {0}: {1}".format(item["case_id"], "; ".join(item["failures"])))
        lines.extend(["", "## Improvement Suggestions", ""])
        if not self.improvement_suggestions:
            lines.append("No improvement suggestions.")
        for suggestion in self.improvement_suggestions:
            lines.append("- {0}".format(suggestion))
        lines.extend(
            [
                "",
                "## Evaluation Scope",
                "",
                "The committed benchmark covers table analysis, chart planning, English/Hinglish/Spanish planning, source selection, tool selection, CSV answer accuracy, executed RAG citations, grounded-number safety, cited PDF arithmetic, and safe error handling.",
                "",
                "## Challenges",
                "",
                "- Preventing hallucinated numeric answers required strict separation between Gemini narration and pandas calculation.",
                "- Supporting vague and Hinglish queries required deterministic fallback rules plus confidence scoring.",
                "- RAG citation integrity required citations to be part of the structured `ToolResult` contract.",
                "- Streamlit safety required every UI stage to handle exceptions and display friendly recovery messages.",
                "- Tool chaining required dependency-aware execution, especially for chart and comparison workflows.",
                "",
                "## Future Improvements",
                "",
                "- Add larger independently reviewed multilingual benchmark sets.",
                "- Add more real-world annual reports and financial statements to the benchmark.",
                "- Add richer YoY, margin, variance, and trend analytics.",
                "- Add dashboard-level trace visualization.",
                "- Add stricter enterprise privacy controls.",
                "- Add browser, accessibility, load, security, and deployment verification.",
            ]
        )
        return "\n".join(lines).strip() + "\n"


class Evaluator:
    def __init__(self, gemini_client: Optional[GeminiClient] = None) -> None:
        self.gemini_client = gemini_client or GeminiClient(api_key="", client=None)
        self.rewriter = QueryRewriterAgent(gemini_client=self.gemini_client)
        self.planner = QueryPlannerAgent(gemini_client=self.gemini_client)
        self.source_selector = SourceSelector()
        self.tool_planner = ToolPlannerAgent(ToolManager().get_registry())
        self.pandas_executor = PandasExecutor()

    def load_cases(self, paths: List[Path]) -> List[Dict[str, Any]]:
        cases: List[Dict[str, Any]] = []
        for path in paths:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                cases.extend(payload.get("cases", []))
            elif isinstance(payload, list):
                cases.extend(payload)
        return cases

    def evaluate_files(self, paths: List[Path]) -> EvaluationReport:
        return self.evaluate_cases(self.load_cases(paths))

    def evaluate_cases(self, cases: List[Dict[str, Any]]) -> EvaluationReport:
        results = [self.evaluate_case(case) for case in cases]
        total = len(results)
        passed = len([result for result in results if result.passed])
        metric_accuracy = self._metric_accuracy(results)
        failed_cases = [
            {"case_id": result.case_id, "query": result.query, "failures": result.failures}
            for result in results
            if not result.passed
        ]
        return EvaluationReport(
            total_cases=total,
            passed_cases=passed,
            accuracy_percentage=round((passed / float(total) * 100.0) if total else 0.0, 2),
            metric_accuracy=metric_accuracy,
            failed_cases=failed_cases,
            improvement_suggestions=self._suggestions(results, metric_accuracy),
            cases=results,
        )

    def evaluate_case(self, case: Dict[str, Any]) -> EvaluationCaseResult:
        case_id = str(case.get("id") or case.get("case_id") or "case")
        query = str(case.get("query") or "")
        expected = dict(case.get("expected") or {})
        scores: Dict[str, float] = {}
        failures: List[str] = []
        actual: Dict[str, Any] = {}

        try:
            sources = list(case.get("available_sources") or [])
            table_profiles = self._table_profiles(case.get("table_profiles") or [])
            document_metadata = list(case.get("document_metadata") or [])

            rewritten = self.rewriter.rewrite(query, language=case.get("language"))
            scores["query_rewrite_quality"] = contains_all_score(
                rewritten.rewritten_query,
                expected.get("rewritten_contains") or [],
            )

            plan = self.planner.plan(query, rewritten, available_sources=sources, table_profiles=table_profiles)
            scores["intent_accuracy"] = exact_match(expected.get("intent"), plan.intent) if expected.get("intent") else 1.0
            scores["semantic_plan_accuracy"] = self._semantic_plan_score(expected, plan)

            source_selection = self.source_selector.select_source(
                plan,
                sources,
                table_profiles=table_profiles,
                document_metadata=document_metadata,
            )
            scores["source_selection_accuracy"] = source_selection_score(
                expected.get("source_id"),
                source_selection.selected_source_id,
                expected.get("source_type"),
                source_selection.source_type,
            )

            selected_for_tools = source_selection if source_selection.selected_source_id else None
            execution_plan = self.tool_planner.create_execution_plan(plan, selected_for_tools)
            actual_tools = [tool_call.tool_name for tool_call in execution_plan.tool_calls]
            scores["tool_selection_accuracy"] = sequence_match_score(
                expected.get("tools") or [],
                actual_tools,
                ordered=True,
            ) if expected.get("tools") is not None else 1.0

            if case.get("csv_case"):
                scores["csv_answer_accuracy"] = self._csv_answer_score(case)
            else:
                scores["csv_answer_accuracy"] = 1.0

            rag_result = self._execute_rag_case(case, plan)
            scores["rag_citation_presence"] = self._rag_citation_score(case, rag_result)
            scores["hallucination_safety"] = self._hallucination_score(case, rag_result)
            comparison_result = self._execute_comparison_case(case, plan, rag_result)
            if case.get("comparison_case"):
                scores["document_comparison_accuracy"] = self._document_comparison_score(case, comparison_result)
            scores["error_handling"] = self._error_handling_score(case, execution_plan)

            actual = {
                "rewritten_query": rewritten.rewritten_query,
                "intent": plan.intent,
                "source_id": source_selection.selected_source_id,
                "source_type": source_selection.source_type,
                "tools": actual_tools,
                "warnings": list(execution_plan.warnings or []),
            }
            if rag_result is not None:
                actual["rag"] = {
                    "success": rag_result.success,
                    "answer": rag_result.answer,
                    "citation_count": len(rag_result.citations or []),
                }
            if comparison_result is not None:
                actual["comparison"] = {
                    "success": comparison_result.success,
                    "answer": comparison_result.answer,
                    "data": comparison_result.data,
                    "citation_count": len(comparison_result.citations or []),
                }
        except Exception as exc:
            scores["error_handling"] = 0.0
            failures.append("Evaluator raised uncaught exception: {0}".format(str(exc)))
            actual = {"error": str(exc)}

        for metric, score in scores.items():
            if not pass_fail(score, self._threshold(metric)):
                failures.append("{0} failed with score {1:.2f}".format(metric, score))

        return EvaluationCaseResult(
            case_id=case_id,
            query=query,
            passed=not failures,
            scores=scores,
            failures=failures,
            actual=actual,
            expected=expected,
        )

    def _csv_answer_score(self, case: Dict[str, Any]) -> float:
        csv_case = case.get("csv_case") or {}
        dataframe = pd.DataFrame(csv_case.get("data") or [])
        operation = csv_case.get("operation") or {}
        result = self.pandas_executor.execute(dataframe, operation)
        expected = csv_case.get("expected") or {}
        if not result.success:
            return 0.0
        data_score = table_value_score(expected.get("data"), result.data)
        table_score = table_value_score(expected.get("table"), result.table)
        return round((data_score + table_score) / 2.0, 4)

    def _execute_rag_case(self, case: Dict[str, Any], plan: Any) -> Optional[ToolResult]:
        rag_case = case.get("rag_case") or {}
        chunks = list(rag_case.get("retrieved_chunks") or [])
        if not chunks:
            return None
        payload = {
            "query_plan": plan.model_dump() if hasattr(plan, "model_dump") else plan.dict(),
            "retrieved_chunks": chunks,
            "minimum_score": float(rag_case.get("minimum_score", 0.0)),
        }
        return RagQATool().safe_run(payload)

    def _rag_citation_score(self, case: Dict[str, Any], rag_result: Optional[ToolResult]) -> float:
        rag_case = case.get("rag_case") or {}
        requires = bool(rag_case.get("requires_citations") or (case.get("expected") or {}).get("requires_citations"))
        citation_count = len(rag_result.citations or []) if rag_result is not None else 0
        return citation_presence_score(requires, citation_count)

    def _hallucination_score(self, case: Dict[str, Any], rag_result: Optional[ToolResult]) -> float:
        hallucination_case = case.get("hallucination_case") or {}
        if not hallucination_case and rag_result is None:
            return 1.0
        answer = rag_result.answer if rag_result is not None else hallucination_case.get("answer") or ""
        grounded_payload = (
            (case.get("rag_case") or {}).get("retrieved_chunks")
            or hallucination_case.get("grounded_payload")
            or case.get("csv_case")
            or {}
        )
        return hallucination_risk_score(answer, grounded_payload)

    def _execute_comparison_case(
        self,
        case: Dict[str, Any],
        plan: Any,
        rag_result: Optional[ToolResult],
    ) -> Optional[ToolResult]:
        if not case.get("comparison_case") or rag_result is None:
            return None
        plan_payload = plan.model_dump() if hasattr(plan, "model_dump") else plan.dict()
        rag_payload = rag_result.model_dump() if hasattr(rag_result, "model_dump") else rag_result.dict()
        return CompareTool().safe_run(
            {
                "query_plan": plan_payload,
                "dependency_results": {"rag_qa_tool": rag_payload},
            }
        )

    def _document_comparison_score(
        self,
        case: Dict[str, Any],
        comparison_result: Optional[ToolResult],
    ) -> float:
        if comparison_result is None or not comparison_result.success:
            return 0.0
        expected = dict((case.get("comparison_case") or {}).get("expected") or {})
        checks = [
            table_value_score(expected.get("absolute_difference"), comparison_result.data.get("absolute_difference")),
            table_value_score(expected.get("percentage_change"), comparison_result.data.get("percentage_change")),
            exact_match(expected.get("direction"), comparison_result.data.get("direction")) if expected.get("direction") else 1.0,
            1.0 if len(comparison_result.citations or []) >= int(expected.get("minimum_citations", 0)) else 0.0,
        ]
        return round(sum(checks) / float(len(checks)), 4)

    def _error_handling_score(self, case: Dict[str, Any], execution_plan: Any) -> float:
        error_case = case.get("error_case") or {}
        if not error_case:
            return 1.0
        expected_safe = bool(error_case.get("expects_safe_failure", True))
        has_warning = bool(getattr(execution_plan, "warnings", []))
        has_no_calls = not bool(getattr(execution_plan, "tool_calls", []))
        return 1.0 if expected_safe and (has_warning or has_no_calls) else 0.0

    def _semantic_plan_score(self, expected: Dict[str, Any], plan: Any) -> float:
        checks: List[float] = []
        if expected.get("metrics_contains") is not None:
            actual_metrics = [
                str(item.get("name") or item.get("metric") or item.get("text") or "")
                for item in getattr(plan, "metrics", [])
                if isinstance(item, dict)
            ]
            checks.append(self._contains_any_form_score(expected.get("metrics_contains") or [], actual_metrics))
        if expected.get("aggregations_contains") is not None:
            actual_aggregations = [
                str(item.get("operation") or item.get("agg") or item.get("name") or "")
                for item in getattr(plan, "aggregations", [])
                if isinstance(item, dict)
            ]
            checks.append(self._contains_any_form_score(expected.get("aggregations_contains") or [], actual_aggregations))
        if expected.get("grouping_contains") is not None:
            checks.append(self._contains_any_form_score(expected.get("grouping_contains") or [], getattr(plan, "grouping", [])))
        if expected.get("entities_contains") is not None:
            actual_entities = [
                str(item.get("normalized") or item.get("value") or item.get("text") or item.get("name") or "")
                for item in getattr(plan, "entities", [])
                if isinstance(item, dict)
            ]
            checks.append(self._contains_any_form_score(expected.get("entities_contains") or [], actual_entities))
        if expected.get("chart_requested") is not None:
            checks.append(1.0 if bool(expected.get("chart_requested")) == bool(getattr(plan, "chart_requested", False)) else 0.0)
        if expected.get("chart_type") is not None:
            checks.append(exact_match(expected.get("chart_type"), getattr(plan, "chart_type", "")))
        if expected.get("limit") is not None:
            checks.append(1.0 if int(expected.get("limit")) == int(getattr(plan, "limit", 0) or 0) else 0.0)
        if expected.get("sorting_direction") is not None:
            sorting = getattr(plan, "sorting", {}) or {}
            checks.append(exact_match(expected.get("sorting_direction"), sorting.get("direction")))
        if expected.get("analysis_type") is not None:
            comparison = getattr(plan, "comparison", {}) or {}
            checks.append(
                1.0
                if str(expected.get("analysis_type")).lower()
                in {
                    str(comparison.get("type") or "").lower(),
                    str(comparison.get("analysis_type") or "").lower(),
                }
                else 0.0
            )
        return round(sum(checks) / float(len(checks)), 4) if checks else 1.0

    def _contains_any_form_score(self, expected_values: Iterable[Any], actual_values: Iterable[Any]) -> float:
        expected_list = [self._normalized_key(value) for value in expected_values or [] if self._normalized_key(value)]
        actual_list = [self._normalized_key(value) for value in actual_values or [] if self._normalized_key(value)]
        if not expected_list:
            return 1.0
        matched = 0
        for expected in expected_list:
            if any(expected == actual or expected in actual or actual in expected for actual in actual_list):
                matched += 1
        return matched / float(len(expected_list))

    def _normalized_key(self, value: Any) -> str:
        return str(value or "").lower().replace(" ", "_").replace("-", "_")

    def _table_profiles(self, payloads: List[Dict[str, Any]]) -> List[TableProfile]:
        profiles = []
        for payload in payloads:
            try:
                profiles.append(TableProfile(**payload))
            except Exception:
                continue
        return profiles

    def _threshold(self, metric: str) -> float:
        if metric == "query_rewrite_quality":
            return 0.5
        return 1.0

    def _metric_accuracy(self, results: List[EvaluationCaseResult]) -> Dict[str, float]:
        metric_accuracy: Dict[str, float] = {}
        for metric in METRIC_KEYS:
            relevant = [result.scores[metric] for result in results if metric in result.scores]
            if not relevant:
                metric_accuracy[metric] = 0.0
                continue
            passed = len([score for score in relevant if pass_fail(score, self._threshold(metric))])
            metric_accuracy[metric] = round(passed / float(len(relevant)) * 100.0, 2)
        return metric_accuracy

    def _suggestions(self, results: List[EvaluationCaseResult], metric_accuracy: Dict[str, float]) -> List[str]:
        suggestions = []
        suggestion_map = {
            "query_rewrite_quality": "Improve query rewrite prompts and deterministic synonym handling.",
            "intent_accuracy": "Add more intent examples and ambiguity handling in QueryPlannerAgent.",
            "semantic_plan_accuracy": "Improve semantic metric, entity, grouping, and analytical-operation extraction.",
            "source_selection_accuracy": "Enrich table/document metadata and tune SourceSelector scoring.",
            "tool_selection_accuracy": "Review tool capability metadata and tool chain rules.",
            "csv_answer_accuracy": "Add focused pandas operation fixtures for failing table calculations.",
            "rag_citation_presence": "Require citation-bearing retrieval output before document narration.",
            "hallucination_safety": "Strengthen groundedness checks for unverified numbers or claims.",
            "document_comparison_accuracy": "Improve cited financial metric extraction and cross-document arithmetic.",
            "error_handling": "Add safer fallback behavior for missing sources and malformed plans.",
        }
        for metric, accuracy in metric_accuracy.items():
            if accuracy < 100.0:
                suggestions.append(suggestion_map[metric])
        if not suggestions and results:
            suggestions.append("All benchmark cases passed. Add harder multilingual and noisy-query cases next.")
        return suggestions
