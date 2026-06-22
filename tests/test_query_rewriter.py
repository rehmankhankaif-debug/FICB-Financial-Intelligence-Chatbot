from __future__ import annotations

from src.agents.query_rewriter import QueryRewriterAgent
from src.llm.gemini_client import GeminiClient
from src.models.query import RewrittenQuery


class FakeGemini(GeminiClient):
    def __init__(self, payload):
        self.payload = payload

    def is_available(self) -> bool:
        return True

    def generate_json(self, prompt, fallback=None):
        return self.payload


def test_empty_query_handled_safely() -> None:
    rewritten = QueryRewriterAgent(gemini_client=GeminiClient(api_key="", client=None)).rewrite("")

    assert isinstance(rewritten, RewrittenQuery)
    assert rewritten.confidence == 0.0
    assert rewritten.notes


def test_hinglish_query_rewritten_with_fallback() -> None:
    rewritten = QueryRewriterAgent(gemini_client=GeminiClient(api_key="", client=None)).rewrite(
        "virat ne kitne run maare rcb jeeti kya?"
    )

    assert isinstance(rewritten, RewrittenQuery)
    assert "Virat Kohli" in rewritten.rewritten_query
    assert "Royal Challengers Bangalore" in rewritten.rewritten_query
    assert 0.0 <= rewritten.confidence <= 1.0


def test_fallback_profit_query_is_cleaned() -> None:
    rewritten = QueryRewriterAgent(gemini_client=GeminiClient(api_key="", client=None)).rewrite(
        "bhai profit ka scene kya tha?"
    )

    assert "profit performance" in rewritten.rewritten_query.lower()
    assert rewritten.detected_language in {"en", "hi-en", "hi"}


def test_mocked_gemini_response_returns_rewritten_query() -> None:
    agent = QueryRewriterAgent(
        gemini_client=FakeGemini(
            {
                "original_query": "outline this report",
                "rewritten_query": "Create a structured outline of the uploaded report.",
                "language": "en",
                "detected_language": "en",
                "confidence": 0.91,
                "notes": ["mocked"],
            }
        )
    )

    rewritten = agent.rewrite("outline this report")

    assert rewritten.rewritten_query.startswith("Create a structured outline")
    assert rewritten.confidence == 0.91


def test_spanish_csv_average_profit_is_rewritten_for_table_analysis() -> None:
    rewritten = QueryRewriterAgent(gemini_client=GeminiClient(api_key="", client=None)).rewrite(
        "¿Cuál es el beneficio mensual promedio según el archivo CSV?",
        language="es",
    )

    assert rewritten.language == "es"
    assert "average monthly profit" in rewritten.rewritten_query.lower()
    assert "table data" in rewritten.rewritten_query.lower()


def test_spanish_excel_revenue_trend_preserves_periods() -> None:
    rewritten = QueryRewriterAgent(gemini_client=GeminiClient(api_key="", client=None)).rewrite(
        "Muestra las tendencias de ingresos para Q1 y Q2 del archivo Excel.",
        language="es",
    )

    assert "revenue trends" in rewritten.rewritten_query.lower()
    assert "q1 and q2" in rewritten.rewritten_query.lower()


def test_follow_up_reference_reuses_previous_user_question() -> None:
    rewritten = QueryRewriterAgent(gemini_client=GeminiClient(api_key="", client=None)).rewrite(
        "what about my previous asked question?",
        conversation_context={
            "previous_user_query": "What is the total revenue in the uploaded CSV?",
            "previous_answer": "The total revenue is 124000.",
        },
    )

    assert rewritten.rewritten_query == "What is the total revenue in the uploaded CSV?"
    assert rewritten.confidence >= 0.88
    assert any("follow-up reference" in note.lower() for note in rewritten.notes)
