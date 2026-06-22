from __future__ import annotations

from src.models.document import DocumentChunkSource
from src.rag.chunker import DocumentChunker


def test_chunker_generates_sentence_aware_chunks_with_metadata() -> None:
    source = DocumentChunkSource(
        source_id="src_1",
        filename="report.pdf",
        source_type="pdf",
        content=(
            "Revenue increased in the first quarter. "
            "Profit margins improved because costs declined. "
            "Management expects steady cash flow."
        ),
        metadata={"page": 2, "section": "Overview"},
    )

    chunks = DocumentChunker(chunk_size=70, overlap=15).chunk_source(source)

    assert len(chunks) >= 2
    assert all(chunk.source_id == "src_1" for chunk in chunks)
    assert all(chunk.page == 2 for chunk in chunks)
    assert chunks[0].metadata["section"] == "Overview"
    assert chunks[0].chunk_id


def test_chunker_handles_empty_text() -> None:
    source = DocumentChunkSource(source_id="src_1", filename="empty.pdf", source_type="pdf", content="")

    chunks = DocumentChunker().chunk_source(source)

    assert chunks == []


def test_chunker_removes_decorative_pdf_separator_lines() -> None:
    source = DocumentChunkSource(
        source_id="resume",
        filename="resume.pdf",
        source_type="pdf",
        content=(
            "Professional Summary\n"
            "------------------------------\n"
            "Built financial dashboards with Python and SQL for executive reporting. "
            "Reduced manual analysis time with automated data validation."
        ),
    )

    chunks = DocumentChunker(chunk_size=300, overlap=20).chunk_source(source)

    combined = " ".join(chunk.content for chunk in chunks)
    assert "----------------" not in combined
    assert "financial dashboards" in combined
