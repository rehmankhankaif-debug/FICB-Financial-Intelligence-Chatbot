from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from src.models.citation import Citation
from src.models.execution import ExecutionPlan
from src.models.query import QueryPlan, RewrittenQuery
from src.models.tool import ToolResult


class ChatMessage(BaseModel):
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    role: str = "user"
    content: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ChatHistoryRecord(BaseModel):
    session_id: str = ""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    user_query: str = ""
    rewritten_query: RewrittenQuery = Field(default_factory=RewrittenQuery)
    query_plan: QueryPlan = Field(default_factory=QueryPlan)
    selected_source: Dict[str, Any] = Field(default_factory=dict)
    selected_tools: List[str] = Field(default_factory=list)
    execution_plan: ExecutionPlan = Field(default_factory=ExecutionPlan)
    execution_time_ms: float = 0.0
    confidence_scores: Dict[str, float] = Field(default_factory=dict)
    tool_results: List[ToolResult] = Field(default_factory=list)
    final_answer: str = ""
    citations: List[Citation] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    errors: List[Dict[str, Any]] = Field(default_factory=list)
    document_source_ids: List[str] = Field(default_factory=list)
    trace_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
