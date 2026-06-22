from __future__ import annotations

import hashlib
import math
import re
from typing import Any, List, Optional

import numpy as np

from src.config import DEFAULT_EMBEDDING_MODEL
from src.models.document import DocumentChunk


DEFAULT_EMBEDDING_DIMENSION = 384


class EmbeddingService:
    _cached_models = {}

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        dimension: int = DEFAULT_EMBEDDING_DIMENSION,
        model: Optional[Any] = None,
        prefer_fallback: bool = False,
        local_files_only: bool = True,
    ) -> None:
        self.model_name = model_name
        self.dimension = dimension
        self._model = model
        self.prefer_fallback = prefer_fallback
        self.local_files_only = local_files_only
        self.using_fallback = prefer_fallback

    def embed_text(self, text: str) -> List[float]:
        return self.embed_texts([text])[0]

    def embed_query(self, query: str) -> List[float]:
        return self.embed_text(query)

    def embed_chunks(self, chunks: List[DocumentChunk]) -> List[List[float]]:
        return self.embed_texts([chunk.content for chunk in chunks])

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        safe_texts = [text or "" for text in texts]
        if self.prefer_fallback:
            return [self._fallback_embedding(text) for text in safe_texts]

        model = self._load_model()
        if model is None:
            return [self._fallback_embedding(text) for text in safe_texts]

        vectors = model.encode(safe_texts, convert_to_numpy=True, normalize_embeddings=True)
        array = np.asarray(vectors)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        return [[float(value) for value in row.tolist()] for row in array]

    def _load_model(self) -> Optional[Any]:
        if self._model is not None:
            return self._model
        if self.model_name in self._cached_models:
            self._model = self._cached_models[self.model_name]
            return self._model

        try:
            from sentence_transformers import SentenceTransformer

            try:
                model = SentenceTransformer(self.model_name, local_files_only=self.local_files_only)
            except TypeError:
                if self.local_files_only:
                    self.using_fallback = True
                    return None
                model = SentenceTransformer(self.model_name)

            self._cached_models[self.model_name] = model
            self._model = model
            self.using_fallback = False
            return model
        except Exception:
            self.using_fallback = True
            return None

    def _fallback_embedding(self, text: str) -> List[float]:
        vector = [0.0] * self.dimension
        tokens = re.findall(r"[a-zA-Z0-9]+", (text or "").lower())
        if not tokens:
            tokens = [""]

        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [float(value / norm) for value in vector]
