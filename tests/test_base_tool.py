from __future__ import annotations

from typing import Any

from src.models.tool import ToolResult
from src.tools.base import BaseTool


class SuccessfulDummyTool(BaseTool):
    name = "dummy.success"

    def run(self, input_payload: dict[str, Any]) -> ToolResult:
        return ToolResult(success=True, tool_name=self.name, data={"echo": input_payload})


class FailingDummyTool(BaseTool):
    name = "dummy.failure"

    def run(self, input_payload: dict[str, Any]) -> ToolResult:
        raise RuntimeError("boom")


class InvalidDummyTool(BaseTool):
    name = "dummy.invalid"

    def run(self, input_payload: dict[str, Any]) -> ToolResult:
        return None  # type: ignore[return-value]


def test_successful_dummy_tool_returns_tool_result() -> None:
    result = SuccessfulDummyTool().safe_run({"value": 1})
    assert isinstance(result, ToolResult)
    assert result.success is True
    assert result.data["echo"] == {"value": 1}


def test_failing_dummy_tool_is_caught_by_safe_run() -> None:
    result = FailingDummyTool().safe_run({"value": 1})
    assert isinstance(result, ToolResult)
    assert result.success is False
    assert result.error_msg == "boom"


def test_safe_run_returns_failed_tool_result_on_exception() -> None:
    result = FailingDummyTool().safe_run()
    assert result.success is False
    assert result.tool_name == "dummy.failure"


def test_safe_run_never_raises_uncaught_exception() -> None:
    result = InvalidDummyTool().safe_run({"value": 1})
    assert isinstance(result, ToolResult)
    assert result.success is False
    assert "invalid result" in (result.error_msg or "")
