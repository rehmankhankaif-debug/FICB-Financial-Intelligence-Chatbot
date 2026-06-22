from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, Field

from src.models.query import QueryPlan
from src.models.source import SourceSelection


class ToolCall(BaseModel):
    tool_name: str = ""
    input_payload: Dict[str, Any] = Field(default_factory=dict)
    reason: str = ""
    confidence: float = 0.0
    depends_on: List[str] = Field(default_factory=list)


class ExecutionPlan(BaseModel):
    query_plan: QueryPlan = Field(default_factory=QueryPlan)
    selected_sources: List[SourceSelection] = Field(default_factory=list)
    tool_calls: List[ToolCall] = Field(default_factory=list)
    requires_tool_chain: bool = False
    confidence: float = 0.0
    warnings: List[str] = Field(default_factory=list)
