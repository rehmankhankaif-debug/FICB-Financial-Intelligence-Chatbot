from __future__ import annotations

from src.tools.manager import default_tools
from src.tools.table_analysis_tool import TableAnalysisTool
from src.tools.registry import ToolCapability, ToolRegistry


def test_registry_registers_and_retrieves_tools() -> None:
    registry = ToolRegistry()
    tool = TableAnalysisTool()

    registry.register(tool)

    assert registry.get_tool("table_analysis_tool") is tool
    assert registry.validate_tool_exists("table_analysis_tool") is True


def test_registry_filters_by_intent() -> None:
    registry = ToolRegistry()
    for tool in default_tools():
        registry.register(tool)

    tools = registry.find_tools_by_intent("chart_request")
    names = [tool.name for tool in tools]

    assert "table_analysis_tool" in names
    assert "chart_tool" in names


def test_registry_filters_by_source_type() -> None:
    registry = ToolRegistry()
    for tool in default_tools():
        registry.register(tool)

    tools = registry.find_tools_by_source_type("pdf")
    names = [tool.name for tool in tools]

    assert "summarize_tool" in names
    assert "rag_qa_tool" in names


def test_registry_rejects_missing_tool_safely() -> None:
    registry = ToolRegistry()

    assert registry.get_tool("missing_tool") is None
    assert registry.validate_tool_exists("missing_tool") is False
    assert registry.capability_metadata("missing_tool") is None


def test_registry_exposes_capability_metadata() -> None:
    registry = ToolRegistry()
    registry.register(TableAnalysisTool())

    capability = registry.capability_metadata("table_analysis_tool")

    assert isinstance(capability, ToolCapability)
    assert capability.name == "table_analysis_tool"
    assert "table_analysis" in capability.supported_intents
    assert capability.output_types
