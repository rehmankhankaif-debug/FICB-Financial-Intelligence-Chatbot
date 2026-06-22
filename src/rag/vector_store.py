from __future__ import annotations

import importlib
import os
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import settings
from src.models.document import DocumentChunk, RetrievedChunk
from src.rag.embeddings import EmbeddingService


_CHROMADB = None
_CHROMA_IMPORT_LOCK = threading.Lock()


def _chromadb_module() -> Any:
    global _CHROMADB
    if _CHROMADB is not None:
        return _CHROMADB
    with _CHROMA_IMPORT_LOCK:
        if _CHROMADB is not None:
            return _CHROMADB
        original_cwd = Path.cwd()
        neutral_cwd = Path(tempfile.gettempdir())
        dotenv_module = None
        dotenv_main_module = None
        original_dotenv_values = None
        original_main_dotenv_values = None
        try:
            try:
                dotenv_module = importlib.import_module("dotenv")
                dotenv_main_module = importlib.import_module("dotenv.main")
                original_dotenv_values = getattr(dotenv_module, "dotenv_values", None)
                original_main_dotenv_values = getattr(dotenv_main_module, "dotenv_values", None)
                dotenv_module.dotenv_values = lambda *args, **kwargs: {}
                dotenv_main_module.dotenv_values = lambda *args, **kwargs: {}
            except Exception:
                dotenv_module = None
                dotenv_main_module = None
            os.chdir(neutral_cwd)
            _CHROMADB = importlib.import_module("chromadb")
        finally:
            os.chdir(original_cwd)
            if dotenv_module is not None and original_dotenv_values is not None:
                dotenv_module.dotenv_values = original_dotenv_values
            if dotenv_main_module is not None and original_main_dotenv_values is not None:
                dotenv_main_module.dotenv_values = original_main_dotenv_values
        return _CHROMADB


def _metadata_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _chunk_metadata(chunk: DocumentChunk) -> Dict[str, Any]:
    metadata = {
        "source_id": chunk.source_id,
        "filename": chunk.filename,
        "source_type": chunk.source_type,
        "page": chunk.page if chunk.page is not None else "",
        "chunk_id": chunk.chunk_id,
    }
    for key, value in chunk.metadata.items():
        if key not in metadata:
            metadata[key] = _metadata_value(value)
    return {key: _metadata_value(value) for key, value in metadata.items()}


def _score_from_distance(distance: Any) -> float:
    try:
        distance_value = float(distance)
    except Exception:
        return 0.0
    return 1.0 / (1.0 + max(distance_value, 0.0))


class VectorStore:
    def __init__(self, persist_directory: Optional[Path] = None, collection_name: str = "financial_documents") -> None:
        self.persist_directory = Path(persist_directory or settings.chroma_dir)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        chromadb = _chromadb_module()
        self.client = chromadb.PersistentClient(path=str(self.persist_directory))
        self.collection = self.client.get_or_create_collection(name=self.collection_name)

    def add_documents(
        self,
        chunks: List[DocumentChunk],
        embeddings: Optional[List[List[float]]] = None,
        embedding_service: Optional[EmbeddingService] = None,
    ) -> int:
        if not chunks:
            return 0

        vectors = embeddings
        if vectors is None:
            service = embedding_service or EmbeddingService()
            vectors = service.embed_chunks(chunks)

        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.content for chunk in chunks]
        metadatas = [_chunk_metadata(chunk) for chunk in chunks]

        self.collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=vectors)
        return len(chunks)

    def search(
        self,
        query: str,
        embedding_service: EmbeddingService,
        top_k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievedChunk]:
        query_embedding = embedding_service.embed_query(query)
        kwargs: Dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": max(1, int(top_k)),
            "include": ["documents", "metadatas", "distances"],
        }
        if metadata_filter:
            kwargs["where"] = metadata_filter

        results = self.collection.query(**kwargs)
        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        retrieved: List[RetrievedChunk] = []
        for index, chunk_id in enumerate(ids):
            metadata = metadatas[index] or {}
            page = metadata.get("page")
            retrieved.append(
                RetrievedChunk(
                    chunk_id=str(chunk_id),
                    source_id=str(metadata.get("source_id", "")),
                    filename=str(metadata.get("filename", "")),
                    source_type=str(metadata.get("source_type", "")),
                    score=_score_from_distance(distances[index] if index < len(distances) else 0.0),
                    content=documents[index] or "",
                    page=int(page) if isinstance(page, int) or (isinstance(page, str) and page.isdigit()) else None,
                    metadata=dict(metadata),
                )
            )
        return retrieved

    def delete_source(self, source_id: str) -> None:
        self.collection.delete(where={"source_id": source_id})

    def clear(self) -> None:
        try:
            self.client.delete_collection(self.collection_name)
        except Exception as exc:
            _ignored_error = exc
        self.collection = self.client.get_or_create_collection(name=self.collection_name)

    def count(self) -> int:
        return int(self.collection.count())
