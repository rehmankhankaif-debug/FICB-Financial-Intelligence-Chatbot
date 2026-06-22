from __future__ import annotations

import hashlib
import re
from typing import List

from src.models.document import DocumentChunk, DocumentChunkSource
from src.utils.text_summary import clean_document_text


DEFAULT_CHUNK_SIZE = 800
DEFAULT_OVERLAP = 100


def _sentence_units(text: str) -> List[str]:
    clean_text = re.sub(r"\s+", " ", clean_document_text(text).strip())
    if not clean_text:
        return []
    units = re.split(r"(?<=[.!?])\s+", clean_text)
    return [unit.strip() for unit in units if unit.strip()]


def _split_long_unit(unit: str, chunk_size: int) -> List[str]:
    if len(unit) <= chunk_size:
        return [unit]
    words = unit.split()
    parts: List[str] = []
    current: List[str] = []
    current_length = 0
    for word in words:
        additional_length = len(word) + (1 if current else 0)
        if current and current_length + additional_length > chunk_size:
            parts.append(" ".join(current))
            current = [word]
            current_length = len(word)
        else:
            current.append(word)
            current_length += additional_length
    if current:
        parts.append(" ".join(current))
    return parts


def _tail_overlap(text: str, overlap: int) -> str:
    if overlap <= 0 or not text:
        return ""
    if len(text) <= overlap:
        return text
    tail = text[-overlap:]
    first_space = tail.find(" ")
    if first_space > 0:
        return tail[first_space + 1 :]
    return tail


class DocumentChunker:
    def __init__(self, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")
        if overlap < 0:
            raise ValueError("overlap cannot be negative.")
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size.")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_sources(self, sources: List[DocumentChunkSource]) -> List[DocumentChunk]:
        chunks: List[DocumentChunk] = []
        for source in sources:
            chunks.extend(self.chunk_source(source, start_index=len(chunks)))
        return chunks

    def chunk_source(self, source: DocumentChunkSource, start_index: int = 0) -> List[DocumentChunk]:
        text_chunks = self.chunk_text(source.content)
        chunks: List[DocumentChunk] = []
        page = source.metadata.get("page")

        for index, content in enumerate(text_chunks):
            chunk_index = start_index + index
            digest = hashlib.sha1(
                "{0}:{1}:{2}".format(source.source_id, chunk_index, content).encode("utf-8")
            ).hexdigest()[:12]
            chunks.append(
                DocumentChunk(
                    chunk_id="{0}_{1}_{2}".format(source.source_id, chunk_index, digest),
                    source_id=source.source_id,
                    filename=source.filename,
                    source_type=source.source_type,
                    content=content,
                    page=int(page) if page is not None else None,
                    metadata=dict(source.metadata),
                )
            )
        return chunks

    def chunk_text(self, text: str) -> List[str]:
        units: List[str] = []
        for unit in _sentence_units(text):
            units.extend(_split_long_unit(unit, self.chunk_size))

        chunks: List[str] = []
        current = ""

        for unit in units:
            if not current:
                current = unit
                continue

            candidate = "{0} {1}".format(current, unit)
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                chunks.append(current.strip())
                overlap_text = _tail_overlap(current, self.overlap)
                current = "{0} {1}".format(overlap_text, unit).strip() if overlap_text else unit

        if current:
            chunks.append(current.strip())

        return chunks
