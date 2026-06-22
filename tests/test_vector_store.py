from __future__ import annotations

from src.models.document import DocumentChunk
from src.rag.embeddings import EmbeddingService
from src.rag.vector_store import VectorStore


def _chunks():
    return [
        DocumentChunk(
            chunk_id="chunk_revenue",
            source_id="source_a",
            filename="report.pdf",
            source_type="pdf",
            content="Revenue growth and profit improved this quarter.",
            page=1,
            metadata={"page": 1},
        ),
        DocumentChunk(
            chunk_id="chunk_expense",
            source_id="source_b",
            filename="expenses.pdf",
            source_type="pdf",
            content="Operating expenses declined due to vendor savings.",
            page=2,
            metadata={"page": 2},
        ),
    ]


def test_vector_store_add_search_delete_and_clear(tmp_path) -> None:
    service = EmbeddingService(prefer_fallback=True)
    store = VectorStore(persist_directory=tmp_path, collection_name="test_docs")
    store.clear()

    count = store.add_documents(_chunks(), embedding_service=service)
    results = store.search("revenue profit", embedding_service=service, top_k=1)

    assert count == 2
    assert store.count() == 2
    assert len(results) == 1
    assert results[0].chunk_id == "chunk_revenue"
    assert results[0].source_id == "source_a"
    assert results[0].score > 0

    store.delete_source("source_a")
    assert store.count() == 1

    store.clear()
    assert store.count() == 0


def test_vector_store_metadata_filter(tmp_path) -> None:
    service = EmbeddingService(prefer_fallback=True)
    store = VectorStore(persist_directory=tmp_path, collection_name="test_filter_docs")
    store.clear()
    store.add_documents(_chunks(), embedding_service=service)

    results = store.search(
        "cost savings",
        embedding_service=service,
        top_k=3,
        metadata_filter={"source_id": "source_b"},
    )

    assert len(results) == 1
    assert results[0].source_id == "source_b"
