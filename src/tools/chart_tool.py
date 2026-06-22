from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px

from src.models.query import QueryPlan
from src.models.tool import ToolResult
from src.tools.base import BaseTool


class ChartTool(BaseTool):
    name = "chart_tool"
    description = "Generate Plotly charts from deterministic table results."
    supported_intents = ["chart_request"]
    supported_source_types = ["table", "csv", "xlsx", "xls"]
    input_types = ["table_analysis_tool result", "table records"]
    output_types = ["chart", "metadata"]
    input_requirements = ["table data"]
    capabilities = ["bar_chart", "line_chart", "pie_chart", "scatter_chart", "histogram"]
    positive_examples = ["manual aur automatic cars kitni hain bar graph bnao"]
    negative_examples = ["what is EBITDA?"]
    can_chain_after = ["table_analysis_tool"]
    confidence = 0.86

    def run(self, input_payload: Dict[str, Any]) -> ToolResult:
        try:
            query_plan = QueryPlan(**(input_payload.get("query_plan") or {}))
            table = self._extract_table(input_payload or {})
            if not table:
                return self._failed("No table data available for chart generation.")

            dataframe = pd.DataFrame(table)
            if dataframe.empty:
                return self._failed("Chart data is empty.")

            chart_types = self._requested_chart_types(query_plan, input_payload)
            x_column, y_column, color_column = self._choose_axes(dataframe, input_payload)
            figures = [
                self._build_figure(dataframe, chart_type, x_column, y_column, color_column)
                for chart_type in chart_types
            ]
            chart_value = figures[0] if len(figures) == 1 else figures
            chart_names = " and ".join(chart_type.title() for chart_type in chart_types)

            return ToolResult(
                success=True,
                tool_name=self.name,
                data={
                    "chart_type": chart_types[0],
                    "chart_types": chart_types,
                    "x": x_column,
                    "y": y_column,
                    "color": color_column,
                    "row_count": len(dataframe),
                },
                answer="{0} chart{1} generated successfully.".format(chart_names, "s" if len(figures) > 1 else ""),
                table=table,
                chart=chart_value,
                confidence=self.confidence,
                warnings=[],
                metadata={
                    "plotly_spec": figures[0].to_dict(),
                    "plotly_specs": [figure.to_dict() for figure in figures],
                    "source_rows": len(dataframe),
                },
            )
        except Exception as exc:
            return self._failed("Chart generation failed safely: {0}".format(str(exc)))

    def _extract_table(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        if isinstance(payload.get("table"), list):
            return payload["table"]
        dependency = (payload.get("dependency_results") or {}).get("table_analysis_tool") or {}
        if isinstance(dependency, dict) and isinstance(dependency.get("table"), list):
            return dependency["table"]
        previous_results = payload.get("previous_results") or []
        for result in reversed(previous_results):
            if isinstance(result, dict) and isinstance(result.get("table"), list) and result.get("table"):
                return result["table"]
        return []

    def _choose_axes(self, dataframe: pd.DataFrame, payload: Dict[str, Any]) -> tuple:
        x_column = payload.get("x")
        y_column = payload.get("y")
        color_column = payload.get("color")
        columns = list(dataframe.columns)
        numeric_columns = [column for column in columns if pd.api.types.is_numeric_dtype(dataframe[column])]
        category_columns = [column for column in columns if column not in numeric_columns]
        if x_column not in columns:
            x_column = None
        if y_column not in columns:
            y_column = None
        if color_column not in columns:
            color_column = None
        if not x_column:
            x_column = category_columns[0] if category_columns else columns[0]
        if not y_column:
            y_column = numeric_columns[0] if numeric_columns else columns[-1]
        if not color_column:
            color_column = next((column for column in category_columns if column != x_column), None)
        return x_column, y_column, color_column

    def _requested_chart_types(self, query_plan: QueryPlan, payload: Dict[str, Any]) -> List[str]:
        raw_types = query_plan.chart_types or payload.get("chart_types") or query_plan.chart_type or payload.get("chart_type") or "bar"
        if not isinstance(raw_types, (list, tuple, set)):
            raw_types = [raw_types]
        aliases = {
            "bar_chart": "bar",
            "bar_graph": "bar",
            "pie_chart": "pie",
            "pie_graph": "pie",
            "line_chart": "line",
            "line_graph": "line",
            "scatter_chart": "scatter",
            "scatter_plot": "scatter",
        }
        chart_types: List[str] = []
        for item in raw_types:
            normalized = str(item or "").strip().lower().replace(" ", "_")
            normalized = aliases.get(normalized, normalized)
            if normalized in {"bar", "pie", "line", "scatter", "histogram"} and normalized not in chart_types:
                chart_types.append(normalized)
        return chart_types or ["bar"]

    def _build_figure(
        self,
        dataframe: pd.DataFrame,
        chart_type: str,
        x_column: str,
        y_column: str,
        color_column: Optional[str] = None,
    ):
        if chart_type == "line":
            return px.line(dataframe, x=x_column, y=y_column, color=color_column)
        if chart_type == "pie":
            return px.pie(dataframe, names=x_column, values=y_column)
        if chart_type == "scatter":
            return px.scatter(dataframe, x=x_column, y=y_column, color=color_column)
        if chart_type == "histogram":
            return px.histogram(dataframe, x=y_column)
        figure = px.bar(dataframe, x=x_column, y=y_column, color=color_column)
        if color_column:
            figure.update_layout(barmode="group")
        return figure

    def _failed(self, message: str) -> ToolResult:
        return ToolResult(success=False, tool_name=self.name, table=[], confidence=0.0, error_msg=message, metadata={"tool_name": self.name})
