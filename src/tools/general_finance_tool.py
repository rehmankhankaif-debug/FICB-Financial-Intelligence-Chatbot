from __future__ import annotations

from typing import Any, Dict, Optional

from src.llm.gemini_client import GeminiClient
from src.models.query import QueryPlan
from src.models.tool import ToolResult
from src.tools.base import BaseTool


FALLBACK_EXPLANATIONS = {
    "ebitda": "EBITDA means earnings before interest, taxes, depreciation, and amortization. It is commonly used to discuss operating performance before financing, tax, and non-cash accounting effects.",
    "net profit margin": "Net profit margin measures how much net profit a company keeps from each unit of revenue. It is usually discussed as net profit divided by revenue.",
    "cash flow": "Cash flow describes money moving into and out of a business. Positive cash flow means inflows exceed outflows during the period being discussed.",
}


class GeneralFinanceTool(BaseTool):
    name = "general_finance_tool"
    description = "Answer general conceptual finance questions when no uploaded source is relevant."
    supported_intents = ["general_finance"]
    supported_source_types = ["none", "general"]
    input_types = ["QueryPlan"]
    output_types = ["text", "metadata"]
    input_requirements = ["query_plan"]
    capabilities = ["financial_concept_explanation", "general_guidance"]
    positive_examples = ["what is EBITDA?", "Explain net profit margin."]
    negative_examples = ["calculate revenue from csv"]
    confidence = 0.8

    def __init__(self, gemini_client: Optional[GeminiClient] = None) -> None:
        super().__init__()
        self.gemini_client = gemini_client or GeminiClient()

    def run(self, input_payload: Dict[str, Any]) -> ToolResult:
        try:
            query_plan = QueryPlan(**(input_payload.get("query_plan") or {}))
            question = (query_plan.rewritten_query or query_plan.original_query or input_payload.get("question") or "").strip()
            if not question:
                return ToolResult(success=False, tool_name=self.name, confidence=0.0, error_msg="No general finance question was provided.", metadata={})

            answer = self._gemini_answer(question) or self._fallback_answer(question)
            return ToolResult(
                success=True,
                tool_name=self.name,
                data={"source": "gemini" if self.gemini_client.is_available() else "fallback"},
                answer=answer,
                table=[],
                confidence=self.confidence,
                warnings=[] if self.gemini_client.is_available() else ["Used deterministic fallback explanation because Gemini was unavailable."],
                metadata={"general_finance_only": True},
            )
        except Exception as exc:
            return ToolResult(success=False, tool_name=self.name, confidence=0.0, error_msg="General finance tool failed safely: {0}".format(str(exc)), metadata={})

    def _gemini_answer(self, question: str) -> str:
        if not self.gemini_client.is_available():
            return ""
        prompt = (
            "Answer this general conceptual finance question concisely. "
            "Do not calculate user data. Do not invent uploaded-source facts. Question: {0}"
        ).format(question)
        return self.gemini_client.generate(prompt).strip()

    def _fallback_answer(self, question: str) -> str:
        lower = question.lower()
        for key, answer in FALLBACK_EXPLANATIONS.items():
            if key in lower:
                return answer
        return "This is a general finance concept question. Please ask about a specific finance term such as EBITDA, net profit margin, or cash flow."
