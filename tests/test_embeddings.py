from __future__ import annotations

import numpy as np

from src.models.document import DocumentChunk
from src.rag.embeddings import DEFAULT_EMBEDDING_DIMENSION, EmbeddingService


class FakeModel:
    def __init__(self) -> None:
        self.calls = 0

    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True):
        self.calls += 1
        return np.ones((len(texts), 3), dtype=float)


def test_fallback_embedding_generates_stable_vectors() -> None:
    service = EmbeddingService(prefer_fallback=True)

    first = service.embed_text("revenue profit growth")
    second = service.embed_query("revenue profit growth")

    assert len(first) == DEFAULT_EMBEDDING_DIMENSION
    assert first == second
    assert any(value != 0.0 for value in first)


def test_embed_chunks_uses_chunk_content() -> None:
    service = EmbeddingService(prefer_fallback=True)
    chunks = [
        DocumentChunk(chunk_id="c1", content="revenue growth"),
        DocumentChunk(chunk_id="c2", content="expense reduction"),
    ]

    embeddings = service.embed_chunks(chunks)

    assert len(embeddings) == 2
    assert len(embeddings[0]) == DEFAULT_EMBEDDING_DIMENSION


def test_embedding_service_uses_injected_model_without_reloading() -> None:
    model = FakeModel()
    service = EmbeddingService(model=model)

    embeddings = service.embed_texts(["a", "b"])

    assert model.calls == 1
    assert embeddings == [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
