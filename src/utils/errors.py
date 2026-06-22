from __future__ import annotations

from typing import Any, Dict, Optional


class AppError(Exception):
    def __init__(self, message: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.message = message
        self.metadata = metadata or {}

    def __str__(self) -> str:
        if self.metadata:
            return f"{self.message} | metadata={self.metadata}"
        return self.message


class IngestionError(AppError):
    """Raised when a source cannot be ingested safely."""


class PlanningError(AppError):
    """Raised when query or execution planning fails."""


class ToolExecutionError(AppError):
    """Raised when a tool cannot complete its operation."""


class ValidationError(AppError):
    """Raised when validation fails."""


class UnsupportedFileError(AppError):
    """Raised when the uploaded file type is unsupported."""


class LowConfidenceError(AppError):
    """Raised when confidence is too low to answer safely."""
