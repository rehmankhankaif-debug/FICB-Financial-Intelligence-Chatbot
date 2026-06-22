from __future__ import annotations

from src.models.document import RetrievedChunk
from src.models.query import QueryPlan
from src.tools.summarize_tool import SummarizeTool


def _dump(model):
    return model.model_dump() if hasattr(model, "model_dump") else model.dict()


def test_summarize_tool_creates_outline_with_citations() -> None:
    chunks = [
        RetrievedChunk(
            chunk_id="c1",
            source_id="s1",
            filename="report.pdf",
            page=1,
            content="Executive summary says revenue improved.",
            score=0.9,
        ),
        RetrievedChunk(
            chunk_id="c2",
            source_id="s1",
            filename="report.pdf",
            page=2,
            content="Market outlook remains positive.",
            score=0.8,
        ),
    ]
    plan = QueryPlan(intent="summarize_document", original_query="outline this report")

    result = SummarizeTool().safe_run({"query_plan": _dump(plan), "retrieved_chunks": [_dump(chunk) for chunk in chunks]})

    assert result.success is True
    assert "1." in result.answer
    assert len(result.citations) == 2
    assert result.citations[0].page == 1


def test_summarize_tool_polishes_pdf_style_chunks() -> None:
    chunks = [
        RetrievedChunk(
            chunk_id="c1",
            source_id="s1",
            filename="assignment.pdf",
            page=1,
            content=(
                "Financial Intelligence Chatbot Assignment Objective Develop a robust, scalable financial chatbot "
                "capable of processing both structured and unstructured financial documents. "
                "\u25cf User Query Support: \u25cb Respond to queries regarding specific financial metrics, "
                "comparisons, or trends. \u25cf Intent Recognition: \u25cb Automatically detect user intent "
                "to decide if a specialized tool is required. \u25cf Tool Selection: \u25cb Integrate with tools "
                "for summarization, CSV/Excel analysis, table querying, visualizations, etc."
            ),
            score=0.9,
        )
    ]
    plan = QueryPlan(intent="summarize_document", original_query="summarize this document")

    result = SummarizeTool().safe_run({"query_plan": _dump(plan), "document_chunks": [_dump(chunk) for chunk in chunks]})

    assert result.success is True
    assert result.answer.startswith("Summary:")
    assert "Key points:" in result.answer
    assert "Intent Recognition" in result.answer
    assert "chunk" not in result.answer.lower()
    assert "Citations:" not in result.answer


def test_summarize_tool_extracts_key_fact_statement_fields() -> None:
    chunks = [
        RetrievedChunk(
            chunk_id="c1",
            source_id="s1",
            filename="loan.pdf",
            page=1,
            content=(
                "Ver-Apr'26 Page 1 of 35 Key Fact Statement Ref No: DELDP01152532 Date: 19/05/2026 "
                "Part 1 (Interest rate and fees/charges) 1 Loan proposal/ account No. DELDP01152532 "
                "Type of Loan Education Loan (Domestic) 2 Sanctioned Loan amount (in Rupees) INR 489,500 "
                "3 Disbursal schedule Rs. 489,500 If disbursed in tranches: Tranches Disbursement Month "
                "Loan Amount Tranche-1 May 2026 489,500 Tranche-2 Tranche-3 Tranche-4 "
                "4 Loan term (year/months/days) 36 Months 5 Installment details Type of installments "
                "Number of EPIs EPI (INR) 1 Commencement of repayment, post sanction Monthly 1 - 36 "
                "(EMI) INR 17,347 (EMI) 04/06/26 6 Interest rate (%) and type (fixed or floating or hybrid) "
                "17.07% FIXED (Interest calculated on reducing balance method) 6A Total Interest charged "
                "during entire tenor of the Loan 1. Payable by the Borrower: Rs. 134972"
            ),
            score=0.9,
        ),
        RetrievedChunk(
            chunk_id="c2",
            source_id="s1",
            filename="loan.pdf",
            page=2,
            content=(
                "8 Fee/ Charges Processing fees (Non-Refundable) One time Rs. 0 "
                "Insurance/Wellness charges One time Rs. 0 Valuation fees One time Rs. 0 "
                "9 Annual Percentage Rate (APR) (%) 16.57 % 10 Details of Contingent Charges "
                "(i) Penal charges, if any, in case of delayed payment 2% p.m. is on outstanding emi overdue "
                "(ii) Other penal charges, if any NIL (iii) Foreclosure charges , if applicable NIL "
                "(v) Part-payment Charges NIL This KFS shall be valid for a period of five working days"
            ),
            score=0.8,
        ),
    ]
    plan = QueryPlan(intent="summarize_document", original_query="summarize this loan pdf")

    result = SummarizeTool().safe_run({"query_plan": _dump(plan), "document_chunks": [_dump(chunk) for chunk in chunks]})

    assert result.success is True
    assert "Education Loan (Domestic)" in result.answer
    assert "sanctioned amount INR 489,500" in result.answer
    assert "EMI INR 17,347" in result.answer
    assert "17.07% FIXED" in result.answer
    assert "APR 16.57%" in result.answer
    assert "2% p.m." in result.answer
    assert "processing fee Rs. 0" in result.answer
    assert "Page 1 of 35" not in result.answer
    assert "Payable by the Borrower" not in result.answer


def test_summarize_tool_cleans_resume_separator_artifacts() -> None:
    chunks = [
        RetrievedChunk(
            chunk_id="resume_c1",
            source_id="resume",
            filename="resume.pdf",
            page=1,
            content=(
                "Professional Experience\n"
                "-----------------------------------------------\n"
                "Created financial intelligence dashboards for monthly reporting and executive decisions. "
                "Automated reconciliation checks across invoices, statements, and ledger exports."
            ),
            score=0.9,
        )
    ]
    plan = QueryPlan(intent="summarize_document", original_query="summarize this resume")

    result = SummarizeTool().safe_run({"query_plan": _dump(plan), "document_chunks": [_dump(chunk) for chunk in chunks]})

    assert result.success is True
    assert "--------" not in result.answer
    assert "financial intelligence dashboards" in result.answer


def test_summarize_tool_no_chunks_fails_safely() -> None:
    result = SummarizeTool().safe_run({"query_plan": _dump(QueryPlan(intent="summarize_document"))})

    assert result.success is False
    assert result.error_msg


def test_summarize_tool_formats_resume_fallback_into_sections() -> None:
    chunks = [
        RetrievedChunk(
            chunk_id="resume-1",
            source_id="resume",
            filename="resume.pdf",
            page=1,
            content=(
                "KAIF REHMAN KHAN\n"
                "PROFESSIONAL SUMMARY\nAI/ML Engineer experienced in NLP and predictive models.\n"
                "EDUCATION\nBachelor of Technology in CSE specialised in AI & ML, UPES.\n"
                "TECHNICAL SKILLS\n· Programming Languages: Python, SQL.\n· ML Frameworks: PyTorch, TensorFlow.\n"
                "PROFESSIONAL EXPERIENCE\n· Built NLP pipelines for summarisation and sentiment analysis.\n"
                "KEY PROJECTS & ACHIEVEMENTS\n· Developed an enterprise RAG chatbot.\n"
                "ACHIEVEMENTS\n· Published machine-learning research.\n"
                "PERSONAL DETAILS\nPassport Number: secret"
            ),
            score=0.9,
        )
    ]
    plan = QueryPlan(intent="summarize_document", original_query="summarise it please")

    result = SummarizeTool().safe_run({"query_plan": _dump(plan), "document_chunks": [_dump(chunk) for chunk in chunks]})

    assert result.success is True
    assert "## Resume summary" in result.answer
    assert "### Core skills" in result.answer
    assert "### Experience highlights" in result.answer
    assert "Passport Number" not in result.answer


def test_summarize_tool_covers_whole_document_and_exposes_bounded_context() -> None:
    chunks = [
        RetrievedChunk(chunk_id="c1", source_id="s1", filename="review.pdf", page=1, content="The document requests an engineering review."),
        RetrievedChunk(chunk_id="c2", source_id="s1", filename="review.pdf", page=2, content="Reliability and security must be scored."),
        RetrievedChunk(chunk_id="c3", source_id="s1", filename="review.pdf", page=3, content="Enterprise observability features must be identified."),
        RetrievedChunk(chunk_id="c4", source_id="s1", filename="review.pdf", page=4, content="Architecture and source code require detailed review."),
        RetrievedChunk(chunk_id="c5", source_id="s1", filename="review.pdf", page=5, content="At least 100 failure scenarios must be documented."),
        RetrievedChunk(chunk_id="c6", source_id="s1", filename="review.pdf", page=6, content="The final verdict must decide production approval."),
    ]
    plan = QueryPlan(intent="summarize_document", original_query="give me summary")

    result = SummarizeTool().safe_run({"query_plan": _dump(plan), "document_chunks": [_dump(chunk) for chunk in chunks]})

    assert result.success is True
    assert "final verdict" in result.answer.lower()
    assert result.data["chunk_count"] == 6
    assert result.data["summary_context"][-1]["page"] == 6
    assert result.data["context_truncated"] is False
