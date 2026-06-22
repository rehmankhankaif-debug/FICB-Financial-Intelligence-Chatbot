from __future__ import annotations

from src.tools.base import BaseTool
from src.tools.manager import ToolManager


REQUIRED_TOOL_NAMES = {
    "table_analysis_tool",
    "chart_tool",
    "summarize_tool",
    "compare_tool",
    "rag_qa_tool",
    "url_lookup_tool",
    "general_finance_tool",
}


def test_tool_manager_initializes_default_tools() -> None:
    manager = ToolManager()

    assert set(manager.list_tool_names()) == REQUIRED_TOOL_NAMES
    assert isinstance(manager.get_tool("table_analysis_tool"), BaseTool)


def test_tool_manager_exposes_registry() -> None:
    manager = ToolManager()

    registry = manager.get_registry()

    assert registry.validate_tool_exists("table_analysis_tool") is True


def test_tool_manager_returns_expected_tool_names() -> None:
    manager = ToolManager()

    assert "general_finance_tool" in manager.list_tool_names()
    assert manager.get_tool("missing") is None


def test_tool_manager_validates_available_tools() -> None:
    manager = ToolManager()

    assert manager.validate_available_tools(["table_analysis_tool", "chart_tool"]) is True
    assert manager.validate_available_tools(["table_analysis_tool", "missing"]) is False
