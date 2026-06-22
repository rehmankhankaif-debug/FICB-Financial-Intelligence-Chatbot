from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from src.models.tool import ToolResult


class BaseTool(ABC):
    name: str = ""
    description: str = ""
    input_types: List[str] = []
    output_types: List[str] = []
    supported_intents: List[str] = []
    supported_source_types: List[str] = []
    input_requirements: List[str] = []
    capabilities: List[str] = []
    positive_examples: List[str] = []
    negative_examples: List[str] = []
    required_context: List[str] = []
    can_chain_after: List[str] = []
    can_chain_before: List[str] = []
    confidence: float = 0.0

    def __init__(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
        input_types: Optional[List[str]] = None,
        output_types: Optional[List[str]] = None,
        supported_intents: Optional[List[str]] = None,
        supported_source_types: Optional[List[str]] = None,
        input_requirements: Optional[List[str]] = None,
        capabilities: Optional[List[str]] = None,
        positive_examples: Optional[List[str]] = None,
        negative_examples: Optional[List[str]] = None,
        required_context: Optional[List[str]] = None,
        can_chain_after: Optional[List[str]] = None,
        can_chain_before: Optional[List[str]] = None,
        confidence: Optional[float] = None,
    ) -> None:
        self.name = name if name is not None else self.name
        self.description = description if description is not None else self.description
        self.input_types = list(input_types if input_types is not None else self.input_types)
        self.output_types = list(output_types if output_types is not None else self.output_types)
        self.supported_intents = list(
            supported_intents if supported_intents is not None else self.supported_intents
        )
        self.supported_source_types = list(
            supported_source_types if supported_source_types is not None else self.supported_source_types
        )
        self.input_requirements = list(
            input_requirements if input_requirements is not None else self.input_requirements
        )
        self.capabilities = list(capabilities if capabilities is not None else self.capabilities)
        self.positive_examples = list(
            positive_examples if positive_examples is not None else self.positive_examples
        )
        self.negative_examples = list(
            negative_examples if negative_examples is not None else self.negative_examples
        )
        self.required_context = list(
            required_context if required_context is not None else self.required_context
        )
        self.can_chain_after = list(
            can_chain_after if can_chain_after is not None else self.can_chain_after
        )
        self.can_chain_before = list(
            can_chain_before if can_chain_before is not None else self.can_chain_before
        )
        self.confidence = float(confidence if confidence is not None else self.confidence)

    def capability_metadata(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "supported_intents": list(self.supported_intents),
            "supported_source_types": list(self.supported_source_types),
            "input_requirements": list(self.input_requirements),
            "output_types": list(self.output_types),
            "capabilities": list(self.capabilities),
            "positive_examples": list(self.positive_examples),
            "negative_examples": list(self.negative_examples),
            "can_chain_after": list(self.can_chain_after),
            "can_chain_before": list(self.can_chain_before),
            "confidence": max(0.0, min(1.0, self.confidence)),
        }

    @abstractmethod
    def run(self, input_payload: Dict[str, Any]) -> ToolResult:
        raise NotImplementedError("Tools must implement run() and return ToolResult.")

    def safe_run(self, input_payload: Optional[Dict[str, Any]] = None) -> ToolResult:
        payload = input_payload or {}
        try:
            result = self.run(payload)
            if isinstance(result, ToolResult):
                if not result.tool_name:
                    result.tool_name = self.name
                return result

            return ToolResult(
                success=False,
                tool_name=self.name,
                error_msg="Tool returned an invalid result. Expected ToolResult.",
                metadata={"result_type": type(result).__name__},
            )
        except Exception as exc:
            return ToolResult(
                success=False,
                tool_name=self.name,
                error_msg=str(exc),
                metadata={"error_type": exc.__class__.__name__},
            )
