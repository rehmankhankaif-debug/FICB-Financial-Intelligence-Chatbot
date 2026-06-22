from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class DocumentSource(BaseModel):
    source_id: str = ""
    filename: str = ""
    file_type: str = ""
    path: str = ""
    uploaded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = Field(default_factory=dict)
    status: str = "pending"
    error_msg: Optional[str] = None


class DocumentChunkSource(BaseModel):
    source_id: str = ""
    filename: str = ""
    source_type: str = ""
    content: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentChunk(BaseModel):
    chunk_id: str = ""
    source_id: str = ""
    filename: str = ""
    source_type: str = ""
    content: str = ""
    page: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RetrievedChunk(BaseModel):
    chunk_id: str = ""
    source_id: str = ""
    filename: str = ""
    source_type: str = ""
    score: float = 0.0
    content: str = ""
    page: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
