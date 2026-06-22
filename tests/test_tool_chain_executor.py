from __future__ import annotations

from typing import Any, Dict

from src.agents.tool_chain_executor import ToolChainExecutor
from src.models.execution import ExecutionPlan, ToolCall
from src.models.tool import ToolResult
from src.tools.base import BaseTool
from src.tools.registry import ToolRegistry


class EchoTool(BaseTool):
    def __init__(self, name: str) -> None:
        super().__init__(name=name, confidence=0.9)

    def run(self, input_payload: Dict[str, Any]) -> ToolResult:
        return ToolResult(
            success=True,
            tool_name=self.name,
            data={"input_payload": input_payload},
            answer="ok",
            confidence=0.9,
            metadata={"received_dependency_results": input_payload.get("dependency_results", {})},
        )


class FailingTool(BaseTool):
    def __init__(self, name: str) -> None:
        super().__init__(name=name, confidence=0.2)

    def run(self, input_payload: Dict[str, Any]) -> ToolResult:
        raise RuntimeError("tool exploded")


def _registry_with(*tools: BaseTool) -> ToolRegistry:
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    return registry


def test_executor_executes_single_tool() -> None:
    registry = _registry_with(EchoTool("first_tool"))
    plan = ExecutionPlan(tool_calls=[ToolCall(tool_name="first_tool", input_payload={"value": 1})])

    results = ToolChainExecutor(registry).execute(plan)

    assert len(results) == 1
    assert results[0].success is True
    assert results[0].tool_name == "first_tool"
    assert results[0].metadata["execution_index"] == 0


def test_executor_executes_multi_tool_chain() -> None:
    registry = _registry_with(EchoTool("first_tool"), EchoTool("second_tool"))
    plan = ExecutionPlan(
        tool_calls=[
            ToolCall(tool_name="first_tool", input_payload={"value": 1}),
            ToolCall(tool_name="second_tool", depends_on=["first_tool"]),
        ],
        requires_tool_chain=True,
    )

    results = ToolChainExecutor(registry).execute(plan)

    assert len(results) == 2
    assert all(result.success for result in results)
    assert results[1].metadata["depends_on"] == ["first_tool"]


def test_executor_passes_previous_output_metadata_to_dependent_tools() -> None:
    registry = _registry_with(EchoTool("first_tool"), EchoTool("second_tool"))
    plan = ExecutionPlan(
        tool_calls=[
            ToolCall(tool_name="first_tool", input_payload={"value": 1}),
            ToolCall(tool_name="second_tool", depends_on=["first_tool"]),
        ]
    )

    results = ToolChainExecutor(registry).execute(plan)
    dependency_payload = results[1].metadata["received_dependency_results"]

    assert "first_tool" in dependency_payload
    assert dependency_payload["first_tool"]["success"] is True


def test_executor_handles_failing_tool_safely() -> None:
    registry = _registry_with(FailingTool("bad_tool"))
    plan = ExecutionPlan(tool_calls=[ToolCall(tool_name="bad_tool")])

    results = ToolChainExecutor(registry).execute(plan)

    assert len(results) == 1
    assert results[0].success is False
    assert "tool exploded" in (results[0].error_msg or "")


def test_executor_skips_dependent_tool_after_dependency_failure() -> None:
    registry = _registry_with(FailingTool("bad_tool"), EchoTool("dependent_tool"))
    plan = ExecutionPlan(
        tool_calls=[
            ToolCall(tool_name="bad_tool"),
            ToolCall(tool_name="dependent_tool", depends_on=["bad_tool"]),
        ]
    )

    results = ToolChainExecutor(registry).execute(plan)

    assert len(results) == 2
    assert results[0].success is False
    assert results[1].success is False
    assert results[1].metadata["skipped"] is True
    assert results[1].metadata["failed_dependencies"] == ["bad_tool"]


def test_executor_returns_tool_result_for_missing_dependency_skip() -> None:
    registry = _registry_with(EchoTool("dependent_tool"))
    plan = ExecutionPlan(tool_calls=[ToolCall(tool_name="dependent_tool", depends_on=["missing_tool"])])

    results = ToolChainExecutor(registry).execute(plan)

    assert len(results) == 1
    assert isinstance(results[0], ToolResult)
    assert results[0].success is False
    assert results[0].metadata["missing_dependencies"] == ["missing_tool"]


def test_executor_never_raises_uncaught_exception_for_missing_tool() -> None:
    registry = ToolRegistry()
    plan = ExecutionPlan(tool_calls=[ToolCall(tool_name="not_registered")])

    results = ToolChainExecutor(registry).execute(plan)

    assert len(results) == 1
    assert results[0].success is False
    assert results[0].error_msg == "Tool is not registered."


def test_executor_preserves_execution_metadata() -> None:
    registry = _registry_with(EchoTool("first_tool"))
    plan = ExecutionPlan(
        tool_calls=[
            ToolCall(
                tool_name="first_tool",
                input_payload={"value": 1},
                reason="testing metadata",
                confidence=0.77,
            )
        ]
    )

    results = ToolChainExecutor(registry).execute(plan)

    assert results[0].metadata["execution_index"] == 0
    assert results[0].metadata["tool_call"]["reason"] == "testing metadata"
    assert results[0].metadata["tool_call"]["confidence"] == 0.77
