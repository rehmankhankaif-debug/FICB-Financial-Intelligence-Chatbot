from __future__ import annotations

from src.models.document import RetrievedChunk
from src.rag.citations import CitationBuilder


def test_citation_builder_converts_retrieved_chunks() -> None:
    chunks = [
        RetrievedChunk(
            chunk_id="chunk_1",
            source_id="source_1",
            filename="report.pdf",
            source_type="pdf",
            score=0.9,
            content="Revenue increased by ten percent according to the report.",
            page=3,
            metadata={"page": 3},
        )
    ]

    citations = CitationBuilder().build(chunks)

    assert len(citations) == 1
    assert citations[0].source_id == "source_1"
    assert citations[0].filename == "report.pdf"
    assert citations[0].page == 3
    assert citations[0].chunk_id == "chunk_1"
    assert "Revenue increased" in citations[0].text_snippet


def test_citation_builder_truncates_long_snippets() -> None:
    chunk = RetrievedChunk(
        chunk_id="chunk_1",
        source_id="source_1",
        filename="report.pdf",
        content="word " * 100,
        score=0.9,
    )

    citation = CitationBuilder().build([chunk], snippet_length=25)[0]

    assert citation.text_snippet.endswith("...")
    assert len(citation.text_snippet) <= 28


def test_citation_builder_removes_decorative_separator_lines() -> None:
    chunk = RetrievedChunk(
        chunk_id="resume_chunk",
        source_id="resume",
        filename="resume.pdf",
        content=(
            "Experience\n"
            "--------------------------------------\n"
            "Designed retrieval pipelines for financial documents and dashboards."
        ),
        score=0.9,
    )

    citation = CitationBuilder().build([chunk])[0]

    assert "--------" not in citation.text_snippet
    assert "retrieval pipelines" in citation.text_snippet
