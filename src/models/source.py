from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class SourceSelection(BaseModel):
    selected_source_id: Optional[str] = None
    selected_source_ids: List[str] = Field(default_factory=list)
    selected_source_types: Dict[str, str] = Field(default_factory=dict)
    source_type: str = ""
    confidence: float = 0.0
    reason: str = ""
    alternatives: List[Dict[str, Any]] = Field(default_factory=list)
