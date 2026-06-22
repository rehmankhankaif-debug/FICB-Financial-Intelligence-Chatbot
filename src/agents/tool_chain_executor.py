from __future__ import annotations

import time
from typing import Any, Dict, List

from src.models.execution import ExecutionPlan, ToolCall
from src.models.tool import ToolResult
from src.tools.registry import ToolRegistry
from src.utils.logging import log_tool_call


def _model_payload(model: Any) -> Dict[str, Any]:
    if model is None:
        return {}
    if isinstance(model, dict):
        return dict(model)
    if hasattr(model, "model_dump"):
        return model.model_dump()
    if hasattr(model, "dict"):
        return model.dict()
    return {"value": str(model)}


class ToolChainExecutor:
    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry

    def execute(self, execution_plan: ExecutionPlan) -> List[ToolResult]:
        results: List[ToolResult] = []
        result_by_tool: Dict[str, ToolResult] = {}

        try:
            for index, tool_call in enumerate(execution_plan.tool_calls):
                failed_dependencies = self._failed_dependencies(tool_call, result_by_tool)
                missing_dependencies = self._missing_dependencies(tool_call, result_by_tool)

                if failed_dependencies or missing_dependencies:
                    skipped = self._skipped_result(tool_call, index, failed_dependencies, missing_dependencies)
                    self._log_tool_trace(tool_call, skipped, 0.0, index)
                    results.append(skipped)
                    result_by_tool[tool_call.tool_name] = skipped
                    continue

                tool = self.registry.get_tool(tool_call.tool_name)
                if tool is None:
                    failed = ToolResult(
                        success=False,
                        tool_name=tool_call.tool_name,
                        error_msg="Tool is not registered.",
                        confidence=0.0,
                        warnings=["Tool execution skipped because the tool is missing."],
                        metadata={
                            "execution_index": index,
                            "tool_call": _model_payload(tool_call),
                            "registered": False,
                        },
                    )
                    self._log_tool_trace(tool_call, failed, 0.0, index)
                    results.append(failed)
                    result_by_tool[tool_call.tool_name] = failed
                    continue

                payload = self._payload_for_call(tool_call, result_by_tool, results, index)
                started_at = time.perf_counter()
                result = tool.safe_run(payload)
                elapsed_ms = round((time.perf_counter() - started_at) * 1000.0, 4)
                if result is None:
                    result = ToolResult(
                        success=False,
                        tool_name=tool_call.tool_name,
                        error_msg="Tool returned None.",
                        confidence=0.0,
                        metadata={"execution_index": index},
                    )
                if not result.tool_name:
                    result.tool_name = tool_call.tool_name
                result.metadata.update(
                    {
                        "execution_index": index,
                        "execution_time_ms": elapsed_ms,
                        "tool_call": _model_payload(tool_call),
                        "depends_on": list(tool_call.depends_on),
                    }
                )
                self._log_tool_trace(tool_call, result, elapsed_ms, index)
                results.append(result)
                result_by_tool[tool_call.tool_name] = result
            return results
        except Exception as exc:
            results.append(
                ToolResult(
                    success=False,
                    tool_name="tool_chain_executor",
                    error_msg="Tool chain executor failed safely: {0}".format(str(exc)),
                    confidence=0.0,
                    metadata={"executor_failure": True},
                )
            )
            return results

    def _log_tool_trace(self, tool_call: ToolCall, result: ToolResult, elapsed_ms: float, index: int) -> None:
        log_tool_call(
            tool_call.tool_name,
            {
                "execution_index": index,
                "execution_time_ms": elapsed_ms,
                "success": result.success,
                "confidence": result.confidence,
                "warning_count": len(result.warnings or []),
                "error_msg": result.error_msg,
                "depends_on": list(tool_call.depends_on),
                "trace_id": (tool_call.input_payload or {}).get("trace_id"),
            },
        )

    def _payload_for_call(
        self,
        tool_call: ToolCall,
        result_by_tool: Dict[str, ToolResult],
        previous_results: List[ToolResult],
        index: int,
    ) -> Dict[str, Any]:
        payload = dict(tool_call.input_payload or {})
        dependency_results = {
            dependency: _model_payload(result_by_tool[dependency])
            for dependency in tool_call.depends_on
            if dependency in result_by_tool
        }
        payload.update(
            {
                "dependency_results": dependency_results,
                "previous_results": [_model_payload(result) for result in previous_results],
                "execution_metadata": {
                    "execution_index": index,
                    "depends_on": list(tool_call.depends_on),
                    "reason": tool_call.reason,
                    "tool_call_confidence": tool_call.confidence,
                },
            }
        )
        return payload

    def _failed_dependencies(self, tool_call: ToolCall, result_by_tool: Dict[str, ToolResult]) -> List[str]:
        return [
            dependency
            for dependency in tool_call.depends_on
            if dependency in result_by_tool and not result_by_tool[dependency].success
        ]

    def _missing_dependencies(self, tool_call: ToolCall, result_by_tool: Dict[str, ToolResult]) -> List[str]:
        return [
            dependency
            for dependency in tool_call.depends_on
            if dependency not in result_by_tool
        ]

    def _skipped_result(
        self,
        tool_call: ToolCall,
        index: int,
        failed_dependencies: List[str],
        missing_dependencies: List[str],
    ) -> ToolResult:
        warnings = []
        if failed_dependencies:
            warnings.append("Skipped because dependency failed: {0}".format(", ".join(failed_dependencies)))
        if missing_dependencies:
            warnings.append("Skipped because dependency result is missing: {0}".format(", ".join(missing_dependencies)))

        return ToolResult(
            success=False,
            tool_name=tool_call.tool_name,
            error_msg="Tool skipped due to dependency failure or missing dependency.",
            confidence=0.0,
            warnings=warnings,
            metadata={
                "skipped": True,
                "execution_index": index,
                "tool_call": _model_payload(tool_call),
                "depends_on": list(tool_call.depends_on),
                "failed_dependencies": failed_dependencies,
                "missing_dependencies": missing_dependencies,
            },
        )
