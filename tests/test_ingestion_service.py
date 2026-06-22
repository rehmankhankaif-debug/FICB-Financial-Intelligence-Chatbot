from __future__ import annotations

from pathlib import Path

from src.models.document import DocumentSource
from src.rag.chunker import DocumentChunker
from src.services.ingestion_service import IngestionService


class FakeVectorStore:
    def __init__(self) -> None:
        self.deleted_sources = []
        self.added_chunks = []

    def delete_source(self, source_id: str) -> None:
        self.deleted_sources.append(source_id)

    def add_documents(self, chunks, embedding_service=None) -> int:
        self.added_chunks.extend(chunks)
        return len(chunks)


def test_ingestion_service_profiles_table_source(tmp_path: Path) -> None:
    path = tmp_path / "finance.csv"
    path.write_text("month,revenue,profit\nJan,100,30\nFeb,140,50\n", encoding="utf-8")
    source = DocumentSource(source_id="table_1", filename="finance.csv", file_type="csv", path=str(path), status="uploaded")

    result = IngestionService(vector_store=FakeVectorStore()).ingest_source(source)

    assert result.source_category == "table"
    assert result.dataframe.shape == (2, 3)
    assert result.table_profile.source_id == "table_1"
    assert result.source.metadata["row_count"] == 2


def test_ingestion_service_indexes_html_document_source(tmp_path: Path) -> None:
    path = tmp_path / "report.html"
    path.write_text(
        "<html><head><title>Report</title></head><body><nav>ignore</nav><p>Revenue improved because demand increased.</p></body></html>",
        encoding="utf-8",
    )
    source = DocumentSource(source_id="doc_1", filename="report.html", file_type="html", path=str(path), status="uploaded")
    vector_store = FakeVectorStore()

    result = IngestionService(
        chunker=DocumentChunker(chunk_size=200, overlap=20),
        vector_store=vector_store,
        embedding_service=object(),
    ).ingest_source(source)

    assert result.source_category == "document"
    assert result.indexed_count == len(result.document_chunks)
    assert "demand increased" in result.document_chunks[0].content
    assert "ignore" not in result.document_chunks[0].content
    assert vector_store.deleted_sources == ["doc_1"]
