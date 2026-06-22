from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.agents.confidence import combine_confidence, is_low_confidence, normalize_confidence
from src.models.execution import ExecutionPlan, ToolCall
from src.models.query import QueryPlan
from src.models.source import SourceSelection
from src.tools.manager import ToolManager
from src.tools.registry import ToolRegistry


INTENT_TOOL_CHAINS = {
    "table_analysis": ["table_analysis_tool"],
    "chart_request": ["table_analysis_tool", "chart_tool"],
    "summarize_document": ["summarize_tool"],
    "compare_documents": ["table_analysis_tool", "rag_qa_tool", "compare_tool"],
    "rag_question": ["rag_qa_tool"],
    "url_lookup": ["url_lookup_tool"],
    "general_finance": ["general_finance_tool"],
}

SOURCE_REQUIRED_INTENTS = {
    "table_analysis",
    "chart_request",
    "summarize_document",
    "compare_documents",
    "rag_question",
    "url_lookup",
}


def _model_payload(model: Any) -> Dict[str, Any]:
    if model is None:
        return {}
    if isinstance(model, dict):
        return dict(model)
    if hasattr(model, "model_dump"):
        return model.model_dump()
    if hasattr(model, "dict"):
        return model.dict()
    return {"value": str(model)}


class ToolPlannerAgent:
    def __init__(self, registry: Optional[ToolRegistry] = None) -> None:
        self.registry = registry or ToolManager().get_registry()

    def create_execution_plan(
        self,
        query_plan: QueryPlan,
        source_selection: Optional[SourceSelection] = None,
        available_tools: Optional[List[Dict[str, Any]]] = None,
    ) -> ExecutionPlan:
        try:
            warnings: List[str] = []
            intent = query_plan.intent or ""
            selected_sources = [source_selection] if source_selection and source_selection.selected_source_id else []

            if is_low_confidence(query_plan.confidence):
                warnings.append("Query plan confidence is low; execution should proceed cautiously or ask clarification.")

            if intent not in INTENT_TOOL_CHAINS:
                return ExecutionPlan(
                    query_plan=query_plan,
                    selected_sources=selected_sources,
                    tool_calls=[],
                    requires_tool_chain=False,
                    confidence=0.0,
                    warnings=warnings + ["Unknown intent cannot be mapped to tools safely: {0}".format(intent)],
                )

            if self._requires_source(intent) and not selected_sources:
                return ExecutionPlan(
                    query_plan=query_plan,
                    selected_sources=[],
                    tool_calls=[],
                    requires_tool_chain=False,
                    confidence=combine_confidence([query_plan.confidence, 0.0]),
                    warnings=warnings + ["Missing required uploaded source for intent: {0}".format(intent)],
                )

            requested_chain = self._chain_for_sources(intent, source_selection)
            chain = self._filter_available_chain(requested_chain, available_tools)
            missing = [tool_name for tool_name in INTENT_TOOL_CHAINS[intent] if not self.registry.validate_tool_exists(tool_name)]
            if missing:
                warnings.append("Required tool(s) are not registered: {0}".format(", ".join(missing)))

            tool_calls = self._build_tool_calls(chain, query_plan, source_selection)
            tool_confidences = [
                self.registry.capability_metadata(tool_name).confidence
                for tool_name in chain
                if self.registry.capability_metadata(tool_name) is not None
            ]
            source_confidence = source_selection.confidence if source_selection else (1.0 if intent == "general_finance" else 0.0)
            confidence = combine_confidence([query_plan.confidence, source_confidence] + tool_confidences)

            return ExecutionPlan(
                query_plan=query_plan,
                selected_sources=selected_sources,
                tool_calls=tool_calls,
                requires_tool_chain=len(tool_calls) > 1,
                confidence=normalize_confidence(confidence),
                warnings=warnings,
            )
        except Exception as exc:
            return ExecutionPlan(
                query_plan=query_plan if query_plan is not None else QueryPlan(),
                selected_sources=[],
                tool_calls=[],
                requires_tool_chain=False,
                confidence=0.0,
                warnings=["Tool planning failed safely: {0}".format(str(exc))],
            )

    def plan_tools(
        self,
        query_plan: QueryPlan,
        source_selection: Optional[SourceSelection] = None,
        available_tools: Optional[List[Dict[str, Any]]] = None,
    ) -> ExecutionPlan:
        return self.create_execution_plan(query_plan, source_selection, available_tools)

    def _requires_source(self, intent: str) -> bool:
        return intent in SOURCE_REQUIRED_INTENTS

    def _chain_for_sources(self, intent: str, source_selection: Optional[SourceSelection]) -> List[str]:
        chain = list(INTENT_TOOL_CHAINS[intent])
        if intent != "compare_documents" or source_selection is None:
            return chain
        source_types = set((source_selection.selected_source_types or {}).values())
        if source_types == {"document"}:
            return ["rag_qa_tool", "compare_tool"]
        if source_types == {"table"}:
            return ["table_analysis_tool", "compare_tool"]
        return chain

    def _filter_available_chain(self, chain: List[str], available_tools: Optional[List[Dict[str, Any]]]) -> List[str]:
        if not available_tools:
            return [tool_name for tool_name in chain if self.registry.validate_tool_exists(tool_name)]
        available_names = {
            str(tool.get("name") or tool.get("tool_name"))
            for tool in available_tools
            if isinstance(tool, dict)
        }
        return [
            tool_name
            for tool_name in chain
            if tool_name in available_names and self.registry.validate_tool_exists(tool_name)
        ]

    def _build_tool_calls(
        self,
        chain: List[str],
        query_plan: QueryPlan,
        source_selection: Optional[SourceSelection],
    ) -> List[ToolCall]:
        calls: List[ToolCall] = []
        base_payload = {
            "query_plan": _model_payload(query_plan),
            "source_selection": _model_payload(source_selection),
        }

        for tool_name in chain:
            depends_on: List[str] = []
            if tool_name == "chart_tool":
                depends_on = ["table_analysis_tool"]
            elif tool_name == "compare_tool":
                depends_on = [name for name in chain if name != "compare_tool"]

            calls.append(
                ToolCall(
                    tool_name=tool_name,
                    input_payload=dict(base_payload),
                    reason=self._reason_for_tool(tool_name, query_plan.intent),
                    confidence=self._tool_confidence(tool_name),
                    depends_on=depends_on,
                )
            )
        return calls

    def _tool_confidence(self, tool_name: str) -> float:
        capability = self.registry.capability_metadata(tool_name)
        return capability.confidence if capability else 0.0

    def _reason_for_tool(self, tool_name: str, intent: str) -> str:
        reasons = {
            "table_analysis_tool": "Structured table evidence is required for intent {0}.".format(intent),
            "chart_tool": "Chart rendering requires table output first.",
            "summarize_tool": "Document summary or outline is required.",
            "rag_qa_tool": "Document evidence retrieval is required.",
            "compare_tool": "Comparison requires prior table and document evidence.",
            "url_lookup_tool": "URL lookup is required.",
            "general_finance_tool": "General finance question does not require uploaded source.",
        }
        return reasons.get(tool_name, "Tool selected from capability metadata.")
