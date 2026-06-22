from __future__ import annotations

from src.agents.response_narrator import ResponseNarrator
from src.llm.gemini_client import GeminiClient
from src.models import Citation, ExecutionPlan, FinalResponse, QueryPlan, SourceSelection, ToolCall, ToolResult, ValidationResult


class EmptyGemini(GeminiClient):
    def is_available(self) -> bool:
        return False

    def generate(self, prompt: str) -> str:
        return ""


class UnsafeNumberGemini(GeminiClient):
    def is_available(self) -> bool:
        return True

    def generate(self, prompt: str) -> str:
        return "The verified answer is 9999."


class CapturingPolishGemini(GeminiClient):
    def __init__(self) -> None:
        self.prompt = ""

    def is_available(self) -> bool:
        return True

    def generate(self, prompt: str) -> str:
        self.prompt = prompt
        return "The resume shows Python, SQL, and financial analytics experience."


def _source(source_type: str = "table") -> SourceSelection:
    return SourceSelection(selected_source_id="source_1", source_type=source_type, confidence=0.9)


def _execution(query_plan: QueryPlan, *tool_names: str) -> ExecutionPlan:
    return ExecutionPlan(
        query_plan=query_plan,
        tool_calls=[ToolCall(tool_name=name) for name in tool_names],
        confidence=0.88,
    )


def test_table_answer_narration_preserves_table() -> None:
    query_plan = QueryPlan(intent="table_analysis", language="en", confidence=0.9)
    result = ToolResult(
        success=True,
        tool_name="table_analysis_tool",
        answer="Average profit is 200.",
        table=[{"mean_profit": 200}],
        data={"value": 200},
        confidence=0.92,
    )

    response = ResponseNarrator(EmptyGemini()).narrate(
        "average profit",
        "Calculate average profit.",
        query_plan,
        _source("table"),
        _execution(query_plan, "table_analysis_tool"),
        [result],
        ValidationResult(is_valid=True, confidence=0.91),
        "en",
    )

    assert isinstance(response, FinalResponse)
    assert response.table == [{"mean_profit": 200}]
    assert "Average profit is 200." in response.answer
    assert response.confidence > 0.8


def test_chart_answer_narration_preserves_chart() -> None:
    query_plan = QueryPlan(intent="chart_request", chart_requested=True, chart_type="bar", language="en", confidence=0.88)
    chart = {"type": "plotly", "chart": "bar"}
    result = ToolResult(
        success=True,
        tool_name="chart_tool",
        answer="Bar chart generated successfully.",
        table=[{"transmission": "Manual", "count_rows": 3}],
        chart=chart,
        confidence=0.86,
    )

    response = ResponseNarrator(EmptyGemini()).narrate(
        "make bar chart",
        "Create a bar chart.",
        query_plan,
        _source("table"),
        _execution(query_plan, "chart_tool"),
        [result],
        ValidationResult(is_valid=True, confidence=0.85),
        "en",
    )

    assert response.chart == chart
    assert response.table == [{"transmission": "Manual", "count_rows": 3}]
    assert "chart is attached" in response.answer.lower()


def test_citation_answer_narration_preserves_citations() -> None:
    query_plan = QueryPlan(intent="rag_question", language="en", confidence=0.86)
    citation = Citation(source_id="s1", filename="report.pdf", page=2, chunk_id="c1", text_snippet="Revenue improved.")
    result = ToolResult(
        success=True,
        tool_name="rag_qa_tool",
        answer="Revenue improved according to the report.",
        citations=[citation],
        confidence=0.86,
    )

    response = ResponseNarrator(EmptyGemini()).narrate(
        "what happened to revenue",
        "Explain revenue change.",
        query_plan,
        _source("document"),
        _execution(query_plan, "rag_qa_tool"),
        [result],
        ValidationResult(is_valid=True, confidence=0.86),
        "en",
    )

    assert response.citations == [citation]
    assert "Revenue improved according to the report." in response.answer
    assert "Sources are attached below." in response.answer
    assert "report.pdf page 2 chunk c1" not in response.answer
    assert response.metadata["narration_mode"] == "deterministic_fallback"
    assert any("not configured" in warning.lower() for warning in response.warnings)


def test_failed_validation_narration_explains_limitation() -> None:
    query_plan = QueryPlan(intent="table_analysis", language="en", confidence=0.8)
    result = ToolResult(
        success=False,
        tool_name="table_analysis_tool",
        error_msg="Missing profit column.",
        confidence=0.0,
    )
    validation = ValidationResult(
        is_valid=False,
        confidence=0.2,
        issues=["Table answer has no pandas-grounded data."],
        clarification_needed=True,
        clarification_question="Which metric should I use?",
    )

    response = ResponseNarrator(EmptyGemini()).narrate(
        "average profit",
        "Calculate average profit.",
        query_plan,
        _source("table"),
        _execution(query_plan, "table_analysis_tool"),
        [result],
        validation,
        "en",
    )

    assert response.metadata["validation_is_valid"] is False
    assert "could not produce a fully validated answer" in response.answer
    assert any("Issue:" in warning for warning in response.warnings)
    assert response.confidence == 0.2


def test_hinglish_response_uses_hinglish_fallback() -> None:
    query_plan = QueryPlan(intent="table_analysis", language="hi-en", confidence=0.9)
    result = ToolResult(
        success=True,
        tool_name="table_analysis_tool",
        answer="Average profit is 200.",
        table=[{"mean_profit": 200}],
        confidence=0.9,
    )

    response = ResponseNarrator(EmptyGemini()).narrate(
        "profit batao",
        "Calculate profit.",
        query_plan,
        _source("table"),
        _execution(query_plan, "table_analysis_tool"),
        [result],
        ValidationResult(is_valid=True, confidence=0.9),
        "hi-en",
    )

    assert response.metadata["language"] == "hi-en"
    assert response.answer.lower().startswith("yeh verified result hai")


def test_hinglish_response_detects_original_query_without_explicit_preference() -> None:
    query_plan = QueryPlan(intent="table_analysis", language="en", confidence=0.9)
    result = ToolResult(
        success=True,
        tool_name="table_analysis_tool",
        answer="Average profit is 200.",
        table=[{"mean_profit": 200}],
        confidence=0.9,
    )

    response = ResponseNarrator(EmptyGemini()).narrate(
        "profit batao",
        "Calculate profit.",
        query_plan,
        _source("table"),
        _execution(query_plan, "table_analysis_tool"),
        [result],
        ValidationResult(is_valid=True, confidence=0.9),
    )

    assert response.metadata["language"] == "hi-en"
    assert response.answer.lower().startswith("yeh verified result hai")


def test_gemini_fallback_rejects_unverified_numbers() -> None:
    query_plan = QueryPlan(intent="table_analysis", language="en", confidence=0.9)
    result = ToolResult(
        success=True,
        tool_name="table_analysis_tool",
        answer="Average profit is 200.",
        table=[{"mean_profit": 200}],
        data={"value": 200},
        confidence=0.9,
    )

    response = ResponseNarrator(UnsafeNumberGemini()).narrate(
        "average profit",
        "Calculate average profit.",
        query_plan,
        _source("table"),
        _execution(query_plan, "table_analysis_tool"),
        [result],
        ValidationResult(is_valid=True, confidence=0.9),
        "en",
    )

    assert "9999" not in response.answer
    assert "Average profit is 200." in response.answer
    assert response.metadata["used_gemini"] is False
    assert response.metadata["gemini_fallback_used"] is True


def test_document_narration_prompts_gemini_to_polish_retrieved_evidence() -> None:
    gemini = CapturingPolishGemini()
    query_plan = QueryPlan(intent="rag_question", language="en", confidence=0.9)
    citation = Citation(
        source_id="resume",
        filename="resume.pdf",
        page=1,
        chunk_id="resume_c1",
        text_snippet="Skills Python, SQL, and financial analytics.",
    )
    result = ToolResult(
        success=True,
        tool_name="rag_qa_tool",
        answer=(
            "Skills\n"
            "-----------------------------------------------\n"
            "Python, SQL, and financial analytics experience."
        ),
        citations=[citation],
        data={"retrieved_chunks": [{"content": "Skills Python, SQL, and financial analytics experience."}]},
        confidence=0.9,
    )

    response = ResponseNarrator(gemini).narrate(
        "what skills are in this resume",
        "List resume skills.",
        query_plan,
        _source("document"),
        _execution(query_plan, "rag_qa_tool"),
        [result],
        ValidationResult(is_valid=True, confidence=0.9),
        "en",
    )

    assert response.metadata["used_gemini"] is True
    assert response.metadata["narration_mode"] == "gemini"
    assert "--------" not in response.answer
    assert "polished prose" in gemini.prompt
    assert "resume separators" in gemini.prompt


def test_summary_narration_prompt_requests_full_document_coverage() -> None:
    gemini = CapturingPolishGemini()
    query_plan = QueryPlan(intent="summarize_document", language="en", confidence=0.9)
    result = ToolResult(
        success=True,
        tool_name="summarize_tool",
        answer="Extractive fallback.",
        data={"summary_context": [{"page": 1, "content": "Opening."}, {"page": 6, "content": "Final verdict."}]},
        confidence=0.9,
    )

    response = ResponseNarrator(gemini).narrate(
        "give me summary",
        "Summarize the document.",
        query_plan,
        _source("document"),
        _execution(query_plan, "summarize_tool"),
        [result],
        ValidationResult(is_valid=True, confidence=0.9),
        "en",
    )

    assert response.metadata["used_gemini"] is True
    assert "cover the full document" in gemini.prompt
    assert "Final verdict" in gemini.prompt
