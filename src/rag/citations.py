from __future__ import annotations

from typing import List

from src.models.citation import Citation
from src.models.document import RetrievedChunk
from src.utils.text_summary import clean_document_text


class CitationBuilder:
    def build(self, retrieved_chunks: List[RetrievedChunk], snippet_length: int = 240) -> List[Citation]:
        citations: List[Citation] = []
        for chunk in retrieved_chunks:
            snippet = " ".join(clean_document_text(chunk.content).split())
            if len(snippet) > snippet_length:
                snippet = snippet[:snippet_length].rstrip() + "..."
            citations.append(
                Citation(
                    source_id=chunk.source_id,
                    filename=chunk.filename,
                    page=chunk.page,
                    chunk_id=chunk.chunk_id,
                    text_snippet=snippet,
                )
            )
        return citations
