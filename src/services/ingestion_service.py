from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from bs4 import BeautifulSoup

from src.evaluation.benchmark_generator import CsvBenchmarkGenerator
from src.ingestion.docx_loader import load_docx
from src.ingestion.pdf_loader import load_pdf
from src.ingestion.table_loader import load_table
from src.ingestion.url_loader import load_url
from src.models.document import DocumentChunk, DocumentChunkSource, DocumentSource
from src.models.table import TableProfile
from src.rag.chunker import DocumentChunker
from src.rag.embeddings import EmbeddingService
from src.rag.vector_store import VectorStore
from src.table_intelligence.profiler import TableProfiler
from src.utils.logging import log_error


TABLE_FILE_TYPES = {"csv", "xlsx", "xls"}
DOCUMENT_FILE_TYPES = {"pdf", "docx", "txt", "html", "url"}


def dump_model(model: Any) -> Dict[str, Any]:
    if model is None:
        return {}
    if isinstance(model, dict):
        return dict(model)
    if hasattr(model, "model_dump"):
        return model.model_dump()
    if hasattr(model, "dict"):
        return model.dict()
    return {"value": str(model)}


@dataclass
class IngestionResult:
    source: DocumentSource
    source_category: str
    dataframe: Optional[pd.DataFrame] = None
    table_profile: Optional[TableProfile] = None
    table_benchmarks: List[Dict[str, Any]] = field(default_factory=list)
    document_chunks: List[DocumentChunk] = field(default_factory=list)
    document_metadata: Dict[str, Any] = field(default_factory=dict)
    indexed_count: int = 0

    @property
    def chunk_payloads(self) -> List[Dict[str, Any]]:
        return [dump_model(chunk) for chunk in self.document_chunks]


class IngestionService:
    def __init__(
        self,
        *,
        chunker: Optional[DocumentChunker] = None,
        vector_store: Optional[VectorStore] = None,
        embedding_service: Optional[EmbeddingService] = None,
        table_profiler: Optional[TableProfiler] = None,
        benchmark_generator: Optional[CsvBenchmarkGenerator] = None,
    ) -> None:
        self.chunker = chunker or DocumentChunker()
        self.vector_store = vector_store or VectorStore(collection_name="financial_documents")
        self.embedding_service = embedding_service or EmbeddingService(local_files_only=True)
        self.table_profiler = table_profiler or TableProfiler()
        self.benchmark_generator = benchmark_generator or CsvBenchmarkGenerator()

    def ingest_source(self, source: DocumentSource) -> IngestionResult:
        if source.file_type in TABLE_FILE_TYPES:
            return self._ingest_table_source(source)
        if source.file_type in DOCUMENT_FILE_TYPES:
            return self._ingest_document_source(source)
        raise ValueError("Unsupported source type: {0}".format(source.file_type))

    def _ingest_table_source(self, source: DocumentSource) -> IngestionResult:
        dataframe = load_table(source.path)
        profile = self.table_profiler.profile(dataframe, source_id=source.source_id, filename=source.filename)
        benchmarks = self.benchmark_generator.generate_for_dataframe(
            dataframe,
            profile,
            include_answer_checks=False,
        )
        source.metadata.update(
            {
                "row_count": int(dataframe.shape[0]),
                "column_count": int(dataframe.shape[1]),
                "benchmark_question_count": len(benchmarks),
                "source_category": "table",
            }
        )
        return IngestionResult(
            source=source,
            source_category="table",
            dataframe=dataframe,
            table_profile=profile,
            table_benchmarks=benchmarks,
        )

    def _ingest_document_source(self, source: DocumentSource) -> IngestionResult:
        chunk_sources = self._load_document_source(source)
        chunks = self.chunker.chunk_sources(chunk_sources)
        try:
            self.vector_store.delete_source(source.source_id)
        except Exception as exc:
            log_error(exc, {"stage": "delete_existing_vector_source", "source_id": source.source_id})
        indexed_count = self.vector_store.add_documents(chunks, embedding_service=self.embedding_service)
        metadata = {
            "source_id": source.source_id,
            "filename": source.filename,
            "file_type": source.file_type,
            "chunk_count": len(chunks),
            "indexed_chunk_count": indexed_count,
            "source_category": "document",
            "metadata": dict(source.metadata),
        }
        source.metadata.update(
            {
                "chunk_count": len(chunks),
                "indexed_chunk_count": indexed_count,
                "source_category": "document",
            }
        )
        return IngestionResult(
            source=source,
            source_category="document",
            document_chunks=chunks,
            document_metadata=metadata,
            indexed_count=indexed_count,
        )

    def _load_document_source(self, source: DocumentSource) -> List[DocumentChunkSource]:
        if source.file_type == "pdf":
            return load_pdf(source.path, source_id=source.source_id)
        if source.file_type == "docx":
            return load_docx(source.path, source_id=source.source_id)
        if source.file_type in {"txt", "html"}:
            return self._load_text_or_html(source)
        if source.file_type == "url":
            return load_url(source.path, source_id=source.source_id)
        raise ValueError("Unsupported document source type: {0}".format(source.file_type))

    def _load_text_or_html(self, source: DocumentSource) -> List[DocumentChunkSource]:
        path = Path(source.path)
        content = path.read_text(encoding="utf-8", errors="replace")
        metadata = {"source_path": str(path), **dict(source.metadata or {})}
        if source.file_type == "html":
            soup = BeautifulSoup(content, "html.parser")
            for tag in soup.find_all(["script", "style", "nav", "header", "footer", "aside", "noscript", "svg"]):
                tag.decompose()
            title = soup.title.string.strip() if soup.title and soup.title.string else source.filename
            text_parts = [part.strip() for part in soup.get_text(separator="\n").splitlines() if part.strip()]
            content = "\n".join(text_parts)
            metadata["title"] = title
        return [
            DocumentChunkSource(
                source_id=source.source_id,
                filename=source.filename,
                source_type=source.file_type,
                content=content,
                metadata=metadata,
            )
        ]
