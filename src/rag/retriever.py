from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set

from src.models.document import RetrievedChunk
from src.rag.embeddings import EmbeddingService
from src.rag.vector_store import VectorStore
from src.utils.text_summary import clean_document_text


_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_%-]*")
DEFAULT_CANDIDATE_MULTIPLIER = 3
VECTOR_WEIGHT = 0.55
LEXICAL_WEIGHT = 0.45


class Retriever:
    def __init__(self, vector_store: VectorStore, embedding_service: EmbeddingService) -> None:
        self.vector_store = vector_store
        self.embedding_service = embedding_service

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
        minimum_score: float = 0.0,
    ) -> List[RetrievedChunk]:
        requested_top_k = max(1, int(top_k))
        candidate_count = max(requested_top_k, requested_top_k * DEFAULT_CANDIDATE_MULTIPLIER)
        candidates = self.vector_store.search(
            query=query,
            embedding_service=self.embedding_service,
            top_k=candidate_count,
            metadata_filter=metadata_filter,
        )
        reranked = self._rerank(query, candidates)
        if minimum_score > 0:
            reranked = [chunk for chunk in reranked if chunk.score >= minimum_score]
        return reranked[:requested_top_k]

    def _rerank(self, query: str, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        query_terms = _tokens(query)
        seen: Set[str] = set()
        reranked: List[RetrievedChunk] = []
        for chunk in chunks or []:
            key = chunk.chunk_id or "{0}:{1}".format(chunk.source_id, clean_document_text(chunk.content)[:120])
            if key in seen:
                continue
            seen.add(key)
            vector_score = _bounded_score(chunk.score)
            lexical_score = _lexical_score(query_terms, chunk.content)
            combined_score = (VECTOR_WEIGHT * vector_score) + (LEXICAL_WEIGHT * lexical_score)
            chunk.metadata = dict(chunk.metadata or {})
            chunk.metadata["vector_score"] = vector_score
            chunk.metadata["lexical_score"] = lexical_score
            chunk.metadata["combined_score"] = combined_score
            chunk.score = combined_score
            reranked.append(chunk)
        return sorted(reranked, key=lambda item: item.score, reverse=True)


def _tokens(text: str) -> Set[str]:
    return {token.lower() for token in _TOKEN_RE.findall(str(text or "")) if len(token) > 2}


def _bounded_score(value: float) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return 0.0


def _lexical_score(query_terms: Set[str], content: str) -> float:
    if not query_terms:
        return 0.0
    content_terms = _tokens(clean_document_text(content))
    if not content_terms:
        return 0.0
    overlap = len(query_terms.intersection(content_terms))
    return max(0.0, min(1.0, overlap / float(len(query_terms))))
