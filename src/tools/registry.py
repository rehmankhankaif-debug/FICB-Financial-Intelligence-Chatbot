from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from src.tools.base import BaseTool


class ToolCapability(BaseModel):
    name: str = ""
    description: str = ""
    supported_intents: List[str] = Field(default_factory=list)
    supported_source_types: List[str] = Field(default_factory=list)
    input_requirements: List[str] = Field(default_factory=list)
    output_types: List[str] = Field(default_factory=list)
    capabilities: List[str] = Field(default_factory=list)
    positive_examples: List[str] = Field(default_factory=list)
    negative_examples: List[str] = Field(default_factory=list)
    can_chain_after: List[str] = Field(default_factory=list)
    can_chain_before: List[str] = Field(default_factory=list)
    confidence: float = 0.0


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        if not isinstance(tool, BaseTool):
            raise TypeError("Only BaseTool instances can be registered.")
        if not tool.name:
            raise ValueError("Tool must define a non-empty name.")
        self._tools[tool.name] = tool

    def list_tools(self) -> List[BaseTool]:
        return list(self._tools.values())

    def list_tool_names(self) -> List[str]:
        return sorted(self._tools.keys())

    def get_tool(self, name: str) -> Optional[BaseTool]:
        return self._tools.get(name)

    def get(self, name: str) -> Optional[BaseTool]:
        return self.get_tool(name)

    def validate_tool_exists(self, name: str) -> bool:
        return name in self._tools

    def find_tools_by_intent(self, intent: str) -> List[BaseTool]:
        return [
            tool
            for tool in self._tools.values()
            if intent in tool.supported_intents
        ]

    def find_tools_by_source_type(self, source_type: str) -> List[BaseTool]:
        normalized = (source_type or "").lower()
        return [
            tool
            for tool in self._tools.values()
            if normalized in [item.lower() for item in tool.supported_source_types]
        ]

    def capability_metadata(self, name: str) -> Optional[ToolCapability]:
        tool = self.get_tool(name)
        if tool is None:
            return None
        return ToolCapability(**tool.capability_metadata())

    def list_capabilities(self) -> List[ToolCapability]:
        return [ToolCapability(**tool.capability_metadata()) for tool in self.list_tools()]
