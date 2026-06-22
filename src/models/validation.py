from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class ValidationResult(BaseModel):
    is_valid: bool = False
    confidence: float = 0.0
    issues: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    requires_retry: bool = False
    clarification_needed: bool = False
    clarification_question: Optional[str] = None
