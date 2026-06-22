from __future__ import annotations

import src.tools.url_lookup_tool as url_lookup_tool
from src.models.document import DocumentChunkSource
from src.rag.embeddings import EmbeddingService
from src.rag.vector_store import VectorStore
from src.tools.url_lookup_tool import UrlLookupTool


def test_url_lookup_tool_loads_chunks_retrieves_and_cites(monkeypatch, tmp_path) -> None:
    def fake_load_url(url, source_id=None):
        return [
            DocumentChunkSource(
                source_id=source_id or "url_source",
                filename="market_report",
                source_type="url",
                content="Market trends show revenue growth and improving profit margins.",
                metadata={"url": url, "title": "Market Report"},
            )
        ]

    monkeypatch.setattr(url_lookup_tool, "load_url", fake_load_url)
    vector_store = VectorStore(persist_directory=tmp_path, collection_name="url_lookup_test")
    vector_store.clear()

    result = UrlLookupTool().safe_run(
        {
            "url": "https://example.com/report",
            "query_plan": {"intent": "url_lookup", "original_query": "market trends"},
            "embedding_service": EmbeddingService(prefer_fallback=True),
            "vector_store": vector_store,
        }
    )

    assert result.success is True
    assert "revenue growth" in result.answer
    assert result.citations
    assert result.metadata["loaded_chunks"] >= 1


def test_url_lookup_tool_missing_url_fails_safely() -> None:
    result = UrlLookupTool().safe_run({"query_plan": {"intent": "url_lookup"}})

    assert result.success is False
    assert result.error_msg
