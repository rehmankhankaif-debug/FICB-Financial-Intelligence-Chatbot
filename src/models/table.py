from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


class TableProfile(BaseModel):
    source_id: str = ""
    filename: str = ""
    shape: Tuple[int, int] = (0, 0)
    columns: List[str] = Field(default_factory=list)
    normalized_columns: Dict[str, str] = Field(default_factory=dict)
    dtypes: Dict[str, str] = Field(default_factory=dict)
    numeric_columns: List[str] = Field(default_factory=list)
    categorical_columns: List[str] = Field(default_factory=list)
    datetime_columns: List[str] = Field(default_factory=list)
    boolean_columns: List[str] = Field(default_factory=list)
    entity_candidate_columns: List[str] = Field(default_factory=list)
    metric_candidate_columns: List[str] = Field(default_factory=list)
    result_candidate_columns: List[str] = Field(default_factory=list)
    sample_values: Dict[str, List[Any]] = Field(default_factory=dict)
    unique_values: Dict[str, List[Any]] = Field(default_factory=dict)
    missing_values: Dict[str, int] = Field(default_factory=dict)
    numeric_stats: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    semantic_summary: str = ""


class ColumnMatch(BaseModel):
    requested_term: str = ""
    matched_column: Optional[str] = None
    confidence: float = 0.0
    strategy: str = "none"
    reason: str = ""
    alternatives: List[Dict[str, Any]] = Field(default_factory=list)


class ValueMatch(BaseModel):
    requested_value: str = ""
    matched_column: str = ""
    matched_value: Any = None
    confidence: float = 0.0
    strategy: str = "none"
    reason: str = ""
    alternatives: List[Dict[str, Any]] = Field(default_factory=list)
