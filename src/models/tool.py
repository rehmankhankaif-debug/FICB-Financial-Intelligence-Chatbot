from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from src.models.citation import Citation


class ToolResult(BaseModel):
    success: bool = False
    tool_name: str = ""
    data: Dict[str, Any] = Field(default_factory=dict)
    answer: Optional[str] = None
    table: Any = None
    chart: Any = None
    citations: List[Citation] = Field(default_factory=list)
    confidence: float = 0.0
    warnings: List[str] = Field(default_factory=list)
    error_msg: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
