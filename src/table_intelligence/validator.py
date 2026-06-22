from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.models.tool import ToolResult
from src.models.validation import ValidationResult


class TableResultValidator:
    def validate(
        self,
        result: ToolResult,
        operation: Optional[Dict[str, Any]] = None,
        minimum_confidence: float = 0.55,
    ) -> ValidationResult:
        issues: List[str] = []
        warnings: List[str] = []

        if result is None:
            return ValidationResult(
                is_valid=False,
                confidence=0.0,
                issues=["ToolResult is missing."],
                requires_retry=True,
            )

        if not result.success:
            issues.append(result.error_msg or "Tool execution failed.")

        if result.confidence < minimum_confidence:
            warnings.append("Result confidence is below the minimum threshold.")

        if result.table == []:
            issues.append("Result table is empty.")

        if operation:
            missing_columns = self._missing_columns_from_metadata(result)
            if missing_columns:
                issues.append("Missing columns: {0}".format(", ".join(missing_columns)))

            expected_agg = operation.get("agg") or operation.get("aggregation")
            if operation.get("operation") == "aggregate" and expected_agg and not result.data.get("aggregation"):
                issues.append("Aggregation did not produce structured aggregation metadata.")

        warnings.extend(result.warnings)

        is_valid = result.success and not issues and result.confidence >= minimum_confidence
        return ValidationResult(
            is_valid=is_valid,
            confidence=result.confidence if result.success else 0.0,
            issues=issues,
            warnings=warnings,
            requires_retry=not is_valid,
            clarification_needed=False,
            clarification_question=None,
        )

    def _missing_columns_from_metadata(self, result: ToolResult) -> List[str]:
        missing = result.metadata.get("missing_columns") if result.metadata else None
        if isinstance(missing, list):
            return [str(item) for item in missing]
        column = result.metadata.get("column") if result.metadata and not result.success else None
        if column and "column does not exist" in (result.error_msg or "").lower():
            return [str(column)]
        return []
