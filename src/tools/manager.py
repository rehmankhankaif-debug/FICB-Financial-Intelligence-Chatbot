from __future__ import annotations

from typing import List, Optional

from src.tools.base import BaseTool
from src.tools.chart_tool import ChartTool
from src.tools.compare_tool import CompareTool
from src.tools.general_finance_tool import GeneralFinanceTool
from src.tools.rag_qa_tool import RagQATool
from src.tools.registry import ToolRegistry
from src.tools.summarize_tool import SummarizeTool
from src.tools.table_analysis_tool import TableAnalysisTool
from src.tools.url_lookup_tool import UrlLookupTool


def default_tools() -> List[BaseTool]:
    return [
        TableAnalysisTool(),
        ChartTool(),
        SummarizeTool(),
        CompareTool(),
        RagQATool(),
        UrlLookupTool(),
        GeneralFinanceTool(),
    ]


class ToolManager:
    def __init__(self, tools: Optional[List[BaseTool]] = None) -> None:
        self.registry = ToolRegistry()
        for tool in tools if tools is not None else default_tools():
            self.registry.register(tool)

    def get_registry(self) -> ToolRegistry:
        return self.registry

    def get_tool(self, name: str) -> Optional[BaseTool]:
        return self.registry.get_tool(name)

    def validate_available_tools(self, required_tool_names: List[str]) -> bool:
        return all(self.registry.validate_tool_exists(name) for name in required_tool_names)

    def list_tool_names(self) -> List[str]:
        return self.registry.list_tool_names()
