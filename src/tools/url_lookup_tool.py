from __future__ import annotations

from typing import Any, Dict

from src.ingestion.url_loader import load_url
from src.models.query import QueryPlan
from src.models.tool import ToolResult
from src.rag.chunker import DocumentChunker
from src.rag.embeddings import EmbeddingService
from src.rag.retriever import Retriever
from src.rag.vector_store import VectorStore
from src.tools.base import BaseTool
from src.tools.rag_qa_tool import RagQATool


def _dump_model(model):
    if hasattr(model, "model_dump"):
        return model.model_dump()
    if hasattr(model, "dict"):
        return model.dict()
    return {}


class UrlLookupTool(BaseTool):
    name = "url_lookup_tool"
    description = "Load URL content and return grounded lookup results from retrieved chunks."
    supported_intents = ["url_lookup"]
    supported_source_types = ["url", "document"]
    input_types = ["url", "QueryPlan"]
    output_types = ["retrieved_context", "metadata"]
    input_requirements = ["url"]
    capabilities = ["url_lookup", "web_content_lookup"]
    positive_examples = ["latest market trends from this report link"]
    negative_examples = ["group sales by month"]
    confidence = 0.82

    def run(self, input_payload: Dict[str, Any]) -> ToolResult:
        try:
            query_plan = QueryPlan(**(input_payload.get("query_plan") or {}))
            url = input_payload.get("url") or (input_payload.get("source_selection") or {}).get("url")
            retriever = input_payload.get("retriever")
            if isinstance(retriever, Retriever):
                rag_result = RagQATool().safe_run(
                    {
                        "query_plan": _dump_model(query_plan),
                        "retriever": retriever,
                        "top_k": input_payload.get("top_k", 5),
                        "metadata_filter": input_payload.get("metadata_filter"),
                    }
                )
                rag_result.tool_name = self.name
                rag_result.metadata.update({"url": url, "reused_existing_retriever": True})
                return rag_result

            if not url:
                return ToolResult(success=False, tool_name=self.name, confidence=0.0, error_msg="No URL was provided.", metadata={})

            sources = load_url(str(url), source_id=input_payload.get("source_id"))
            chunks = DocumentChunker().chunk_sources(sources)
            embedding_service = input_payload.get("embedding_service") or EmbeddingService(prefer_fallback=True)
            vector_store = input_payload.get("vector_store") or VectorStore(collection_name="url_lookup_tool")
            vector_store.delete_source(sources[0].source_id)
            vector_store.add_documents(chunks, embedding_service=embedding_service)
            retriever = Retriever(vector_store, embedding_service)
            rag_result = RagQATool().safe_run(
                {
                    "query_plan": _dump_model(query_plan),
                    "retriever": retriever,
                    "top_k": input_payload.get("top_k", 5),
                    "metadata_filter": {"source_id": sources[0].source_id},
                }
            )
            rag_result.tool_name = self.name
            rag_result.metadata.update({"url": url, "loaded_chunks": len(chunks), "source_id": sources[0].source_id})
            return rag_result
        except Exception as exc:
            return ToolResult(success=False, tool_name=self.name, confidence=0.0, error_msg="URL lookup failed safely: {0}".format(str(exc)), metadata={})
