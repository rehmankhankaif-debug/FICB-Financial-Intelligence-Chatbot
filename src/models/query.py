from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class RewrittenQuery(BaseModel):
    original_query: str = ""
    rewritten_query: str = ""
    language: str = "en"
    detected_language: str = "en"
    confidence: float = 0.0
    notes: List[str] = Field(default_factory=list)


class QueryPlan(BaseModel):
    original_query: str = ""
    rewritten_query: str = ""
    language: str = "en"
    intent: str = ""
    required_source_type: Optional[str] = None
    entities: List[Dict[str, Any]] = Field(default_factory=list)
    metrics: List[Dict[str, Any]] = Field(default_factory=list)
    filters: List[Dict[str, Any]] = Field(default_factory=list)
    aggregations: List[Dict[str, Any]] = Field(default_factory=list)
    grouping: List[str] = Field(default_factory=list)
    sorting: Dict[str, Any] = Field(default_factory=dict)
    comparison: Dict[str, Any] = Field(default_factory=dict)
    chart_requested: bool = False
    chart_type: Optional[str] = None
    chart_types: List[str] = Field(default_factory=list)
    limit: Optional[int] = None
    confidence: float = 0.0
    clarification_needed: bool = False
    clarification_question: Optional[str] = None
    reasoning_short: str = ""
