from __future__ import annotations

from src.llm.gemini_client import GeminiClient
from src.tools.general_finance_tool import GeneralFinanceTool


class FakeGemini(GeminiClient):
    def is_available(self) -> bool:
        return True

    def generate(self, prompt: str) -> str:
        return "EBITDA is a measure of operating performance before certain expenses."


def test_general_finance_tool_uses_fallback_when_gemini_missing() -> None:
    tool = GeneralFinanceTool(gemini_client=GeminiClient(api_key="", client=None))

    result = tool.safe_run({"query_plan": {"intent": "general_finance", "original_query": "What is EBITDA?"}})

    assert result.success is True
    assert "earnings before interest" in result.answer.lower()
    assert result.warnings


def test_general_finance_tool_uses_gemini_when_available() -> None:
    tool = GeneralFinanceTool(gemini_client=FakeGemini())

    result = tool.safe_run({"query_plan": {"intent": "general_finance", "original_query": "What is EBITDA?"}})

    assert result.success is True
    assert "operating performance" in result.answer
    assert result.warnings == []


def test_general_finance_tool_missing_question_fails_safely() -> None:
    result = GeneralFinanceTool(gemini_client=GeminiClient(api_key="", client=None)).safe_run({"query_plan": {"intent": "general_finance"}})

    assert result.success is False
    assert result.error_msg
