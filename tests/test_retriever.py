from __future__ import annotations

from src.models.document import DocumentChunk, RetrievedChunk
from src.rag.embeddings import EmbeddingService
from src.rag.retriever import Retriever
from src.rag.validator import RetrievalValidator
from src.rag.vector_store import VectorStore


class FakeVectorStore:
    def __init__(self, chunks):
        self.chunks = chunks
        self.requested_top_k = None

    def search(self, query, embedding_service, top_k=5, metadata_filter=None):
        self.requested_top_k = top_k
        return list(self.chunks)[:top_k]


def test_retriever_returns_top_k_chunks(tmp_path) -> None:
    service = EmbeddingService(prefer_fallback=True)
    store = VectorStore(persist_directory=tmp_path, collection_name="test_retriever_docs")
    store.clear()
    store.add_documents(
        [
            DocumentChunk(
                chunk_id="finance_chunk",
                source_id="source_1",
                filename="report.pdf",
                source_type="pdf",
                content="The report discusses revenue growth and profit expansion.",
                page=1,
                metadata={"page": 1},
            ),
            DocumentChunk(
                chunk_id="people_chunk",
                source_id="source_1",
                filename="report.pdf",
                source_type="pdf",
                content="The appendix lists employee training sessions.",
                page=5,
                metadata={"page": 5},
            ),
        ],
        embedding_service=service,
    )

    chunks = Retriever(store, service).retrieve("profit revenue", top_k=1)

    assert len(chunks) == 1
    assert chunks[0].chunk_id == "finance_chunk"


def test_retriever_reranks_with_lexical_evidence_and_expands_candidates() -> None:
    irrelevant = RetrievedChunk(
        chunk_id="irrelevant",
        source_id="s1",
        filename="report.pdf",
        score=0.9,
        content="The appendix lists employee onboarding notes.",
    )
    relevant = RetrievedChunk(
        chunk_id="relevant",
        source_id="s1",
        filename="report.pdf",
        score=0.4,
        content="Revenue and profit improved after customer demand increased.",
    )
    store = FakeVectorStore([irrelevant, relevant])

    chunks = Retriever(store, object()).retrieve("revenue profit", top_k=1)

    assert store.requested_top_k == 3
    assert chunks[0].chunk_id == "relevant"
    assert chunks[0].metadata["lexical_score"] == 1.0
    assert chunks[0].metadata["combined_score"] == chunks[0].score


def test_retriever_filters_below_minimum_score() -> None:
    store = FakeVectorStore(
        [
            RetrievedChunk(
                chunk_id="weak",
                source_id="s1",
                filename="report.pdf",
                score=0.01,
                content="Unrelated appendix content.",
            )
        ]
    )

    chunks = Retriever(store, object()).retrieve("revenue profit", top_k=1, minimum_score=0.8)

    assert chunks == []


def test_retrieval_validator_detects_empty_low_score_and_duplicates() -> None:
    validator = RetrievalValidator()

    empty = validator.validate([])
    assert empty.is_valid is False
    assert "Retrieval returned no chunks." in empty.issues

    duplicate_chunks = [
        RetrievedChunk(chunk_id="c1", source_id="s1", filename="a.pdf", score=0.5, content="Useful retrieved content."),
        RetrievedChunk(chunk_id="c1", source_id="s1", filename="a.pdf", score=0.5, content="Useful retrieved content."),
    ]
    deduped = validator.deduplicate(duplicate_chunks)
    validation = validator.validate(duplicate_chunks)

    assert len(deduped) == 1
    assert validation.is_valid is True
    assert validation.warnings

    low_score = validator.validate(
        [RetrievedChunk(chunk_id="c2", source_id="s1", filename="a.pdf", score=0.01, content="Weak content here.")]
    )
    assert low_score.is_valid is False
