from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, Field

from src.models.citation import Citation


class FinalResponse(BaseModel):
    answer: str = ""
    table: Any = None
    chart: Any = None
    citations: List[Citation] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    confidence: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)
