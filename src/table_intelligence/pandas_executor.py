from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.models.tool import ToolResult


SUPPORTED_AGGREGATIONS = {"sum", "mean", "median", "count", "nunique", "min", "max"}
SUPPORTED_OPERATIONS = {
    "aggregate",
    "compare",
    "correlation",
    "filter",
    "groupby",
    "group_by",
    "sort",
    "top_k",
    "bottom_k",
}


def _to_builtin(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value


def _records(dataframe: pd.DataFrame) -> List[Dict[str, Any]]:
    return [
        {str(key): _to_builtin(value) for key, value in row.items()}
        for row in dataframe.to_dict(orient="records")
    ]


class PandasExecutor:
    tool_name = "table.pandas_executor"

    def execute(self, dataframe: pd.DataFrame, operation: Dict[str, Any]) -> ToolResult:
        try:
            if dataframe is None or not isinstance(dataframe, pd.DataFrame):
                return self._failed("Input is not a pandas DataFrame.")
            if operation is None or not isinstance(operation, dict):
                return self._failed("Operation must be a structured dictionary.")

            operation_name = str(operation.get("operation", "")).lower()
            if operation_name not in SUPPORTED_OPERATIONS:
                return self._failed(
                    "Unsupported pandas operation.",
                    metadata={"operation": operation_name, "supported_operations": sorted(SUPPORTED_OPERATIONS)},
                )

            working_df = self._apply_filters(dataframe, operation.get("filters", []))

            if operation_name == "filter":
                return self._result_from_dataframe(
                    working_df,
                    "Filtered rows: {0}.".format(len(working_df)),
                    metadata={"operation": operation_name, "row_count": len(working_df)},
                )
            if operation_name == "aggregate":
                return self._aggregate(working_df, operation)
            if operation_name in {"groupby", "group_by"}:
                return self._groupby(working_df, operation)
            if operation_name == "sort":
                return self._sort(working_df, operation)
            if operation_name == "top_k":
                return self._rank(working_df, operation, ascending=False)
            if operation_name == "bottom_k":
                return self._rank(working_df, operation, ascending=True)
            if operation_name == "correlation":
                return self._correlation(working_df, operation)
            if operation_name == "compare":
                return self._compare(dataframe, operation)

            return self._failed("Operation was not executed.")
        except Exception as exc:
            return self._failed(str(exc), metadata={"error_type": exc.__class__.__name__})

    def _apply_filters(self, dataframe: pd.DataFrame, filters: List[Dict[str, Any]]) -> pd.DataFrame:
        filtered_df = dataframe.copy()
        for item in filters or []:
            column = item.get("column")
            operator = str(item.get("operator", "equals")).lower()
            value = item.get("value")
            if column not in filtered_df.columns:
                raise KeyError("Filter column does not exist: {0}".format(column))

            series = filtered_df[column]
            if operator in {"equals", "eq", "=="}:
                mask = series.astype(str).str.lower() == str(value).lower()
            elif operator in {"not_equals", "ne", "!="}:
                mask = series.astype(str).str.lower() != str(value).lower()
            elif operator == "contains":
                mask = series.astype(str).str.contains(str(value), case=False, na=False, regex=False)
            elif operator in {"gt", "greater_than", ">"}:
                mask = pd.to_numeric(series, errors="coerce") > float(value)
            elif operator in {"gte", "greater_equal", ">="}:
                mask = pd.to_numeric(series, errors="coerce") >= float(value)
            elif operator in {"lt", "less_than", "<"}:
                mask = pd.to_numeric(series, errors="coerce") < float(value)
            elif operator in {"lte", "less_equal", "<="}:
                mask = pd.to_numeric(series, errors="coerce") <= float(value)
            elif operator == "in":
                value_set = {str(item).lower() for item in value}
                mask = series.astype(str).str.lower().isin(value_set)
            elif operator == "between":
                lower, upper = value
                numeric = pd.to_numeric(series, errors="coerce")
                mask = numeric.between(float(lower), float(upper))
            else:
                raise ValueError("Unsupported filter operator: {0}".format(operator))

            filtered_df = filtered_df[mask]
        return filtered_df

    def _aggregate(self, dataframe: pd.DataFrame, operation: Dict[str, Any]) -> ToolResult:
        agg = str(operation.get("agg", operation.get("aggregation", ""))).lower()
        column = operation.get("column")
        if agg not in SUPPORTED_AGGREGATIONS:
            return self._failed("Unsupported aggregation.", metadata={"aggregation": agg})
        if agg != "count" and column not in dataframe.columns:
            return self._failed("Aggregation column does not exist.", metadata={"column": column})

        value = self._aggregate_value(dataframe, column, agg)
        alias = operation.get("alias") or "{0}_{1}".format(agg, column or "rows")
        result_df = pd.DataFrame([{alias: value}])
        return ToolResult(
            success=True,
            tool_name=self.tool_name,
            data={"value": _to_builtin(value), "aggregation": agg, "column": column, "alias": alias},
            answer="{0} of {1}: {2}".format(agg, column or "rows", _to_builtin(value)),
            table=_records(result_df),
            confidence=0.95 if len(dataframe) > 0 else 0.55,
            warnings=["Aggregation ran on an empty dataframe."] if len(dataframe) == 0 else [],
            metadata={"operation": "aggregate", "row_count": len(dataframe)},
        )

    def _groupby(self, dataframe: pd.DataFrame, operation: Dict[str, Any]) -> ToolResult:
        group_by = operation.get("group_by") or operation.get("grouping") or []
        aggregations = operation.get("aggregations") or []
        if isinstance(group_by, str):
            group_by = [group_by]
        missing_group_columns = [column for column in group_by if column not in dataframe.columns]
        if missing_group_columns:
            return self._failed("Group-by column does not exist.", metadata={"missing_columns": missing_group_columns})
        if not aggregations:
            aggregations = [{"column": group_by[0] if group_by else None, "agg": "count", "alias": "count"}]

        for item in aggregations:
            agg = str(item.get("agg", item.get("aggregation", ""))).lower()
            column = item.get("column")
            if agg not in SUPPORTED_AGGREGATIONS:
                return self._failed("Unsupported aggregation.", metadata={"aggregation": agg})
            if agg != "count" and column not in dataframe.columns:
                return self._failed("Aggregation column does not exist.", metadata={"column": column})

        if dataframe.empty:
            result_df = pd.DataFrame(columns=list(group_by) + [item.get("alias") or item.get("agg") for item in aggregations])
        else:
            grouped = dataframe.groupby(group_by, dropna=False)
            result_df = grouped.size().reset_index(name="__row_count__")
            for item in aggregations:
                agg = str(item.get("agg", item.get("aggregation", ""))).lower()
                column = item.get("column")
                alias = item.get("alias") or "{0}_{1}".format(agg, column or "rows")
                if agg == "count":
                    series = grouped.size().rename(alias)
                else:
                    series = getattr(grouped[column], agg)().rename(alias)
                result_df = result_df.merge(series.reset_index(), on=group_by, how="left")
            if "__row_count__" in result_df.columns:
                result_df = result_df.drop(columns=["__row_count__"])

        result_df = self._apply_sort_and_limit(result_df, operation)
        return self._result_from_dataframe(
            result_df,
            "Grouped result contains {0} rows.".format(len(result_df)),
            metadata={"operation": "groupby", "group_by": group_by, "row_count": len(result_df)},
            confidence=0.94 if not result_df.empty else 0.55,
        )

    def _sort(self, dataframe: pd.DataFrame, operation: Dict[str, Any]) -> ToolResult:
        sort_by = operation.get("sort_by") or operation.get("column")
        ascending = bool(operation.get("ascending", True))
        if sort_by not in dataframe.columns:
            return self._failed("Sort column does not exist.", metadata={"column": sort_by})
        result_df = dataframe.sort_values(by=sort_by, ascending=ascending)
        result_df = self._limit(result_df, operation.get("limit"))
        return self._result_from_dataframe(result_df, "Sorted rows: {0}.".format(len(result_df)), metadata={"operation": "sort"})

    def _rank(self, dataframe: pd.DataFrame, operation: Dict[str, Any], ascending: bool) -> ToolResult:
        sort_by = operation.get("sort_by") or operation.get("column")
        if sort_by not in dataframe.columns:
            return self._failed("Ranking column does not exist.", metadata={"column": sort_by})
        k = int(operation.get("k", operation.get("limit", 5)))
        result_df = dataframe.sort_values(by=sort_by, ascending=ascending).head(k)
        label = "Bottom" if ascending else "Top"
        return self._result_from_dataframe(
            result_df,
            "{0} {1} rows by {2}.".format(label, len(result_df), sort_by),
            metadata={"operation": "bottom_k" if ascending else "top_k", "sort_by": sort_by, "k": k},
        )

    def _compare(self, dataframe: pd.DataFrame, operation: Dict[str, Any]) -> ToolResult:
        column = operation.get("column")
        agg = str(operation.get("agg", "sum")).lower()
        left_filters = operation.get("left_filters", [])
        right_filters = operation.get("right_filters", [])
        if agg not in SUPPORTED_AGGREGATIONS:
            return self._failed("Unsupported comparison aggregation.", metadata={"aggregation": agg})
        if agg != "count" and column not in dataframe.columns:
            return self._failed("Comparison column does not exist.", metadata={"column": column})

        left_df = self._apply_filters(dataframe, left_filters)
        right_df = self._apply_filters(dataframe, right_filters)
        left_value = self._aggregate_value(left_df, column, agg)
        right_value = self._aggregate_value(right_df, column, agg)
        difference = left_value - right_value if left_value is not None and right_value is not None else None
        result_df = pd.DataFrame(
            [
                {"side": "left", "value": left_value},
                {"side": "right", "value": right_value},
                {"side": "difference", "value": difference},
            ]
        )
        return ToolResult(
            success=True,
            tool_name=self.tool_name,
            data={
                "left_value": _to_builtin(left_value),
                "right_value": _to_builtin(right_value),
                "difference": _to_builtin(difference),
                "aggregation": agg,
                "column": column,
            },
            answer="Comparison difference: {0}".format(_to_builtin(difference)),
            table=_records(result_df),
            confidence=0.9,
            metadata={"operation": "compare"},
        )

    def _correlation(self, dataframe: pd.DataFrame, operation: Dict[str, Any]) -> ToolResult:
        columns = operation.get("columns") or []
        if not columns:
            columns = [operation.get("column_x"), operation.get("column_y")]
        columns = [column for column in columns if column]
        if len(columns) < 2:
            return self._failed("Correlation requires two numeric columns.")
        column_x, column_y = columns[:2]
        missing = [column for column in [column_x, column_y] if column not in dataframe.columns]
        if missing:
            return self._failed("Correlation column does not exist.", metadata={"missing_columns": missing})

        pair = dataframe[[column_x, column_y]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(pair) < 2:
            return self._failed(
                "Correlation requires at least two complete numeric rows.",
                metadata={"columns": [column_x, column_y], "complete_rows": len(pair)},
            )

        value = float(pair[column_x].corr(pair[column_y]))
        result_df = pd.DataFrame(
            [
                {
                    "column_x": column_x,
                    "column_y": column_y,
                    "correlation": value,
                    "complete_rows": int(len(pair)),
                }
            ]
        )
        return ToolResult(
            success=True,
            tool_name=self.tool_name,
            data={
                "column_x": column_x,
                "column_y": column_y,
                "correlation": _to_builtin(value),
                "complete_rows": int(len(pair)),
            },
            answer="Correlation between {0} and {1}: {2}".format(column_x, column_y, round(value, 4)),
            table=_records(result_df),
            confidence=0.9,
            metadata={"operation": "correlation", "columns": [column_x, column_y], "row_count": int(len(pair))},
        )

    def _aggregate_value(self, dataframe: pd.DataFrame, column: Optional[str], agg: str) -> Any:
        if agg == "count":
            return int(len(dataframe)) if column is None else int(dataframe[column].count())
        series = pd.to_numeric(dataframe[column], errors="coerce")
        if agg == "sum":
            return float(series.sum())
        if agg == "mean":
            return float(series.mean()) if not series.dropna().empty else None
        if agg == "median":
            return float(series.median()) if not series.dropna().empty else None
        if agg == "nunique":
            return int(dataframe[column].nunique(dropna=True))
        if agg == "min":
            return float(series.min()) if not series.dropna().empty else None
        if agg == "max":
            return float(series.max()) if not series.dropna().empty else None
        raise ValueError("Unsupported aggregation: {0}".format(agg))

    def _apply_sort_and_limit(self, dataframe: pd.DataFrame, operation: Dict[str, Any]) -> pd.DataFrame:
        sort_by = operation.get("sort_by")
        if sort_by and sort_by in dataframe.columns:
            dataframe = dataframe.sort_values(by=sort_by, ascending=bool(operation.get("ascending", True)))
        return self._limit(dataframe, operation.get("limit"))

    def _limit(self, dataframe: pd.DataFrame, limit: Any) -> pd.DataFrame:
        if limit is None:
            return dataframe
        return dataframe.head(int(limit))

    def _result_from_dataframe(
        self,
        dataframe: pd.DataFrame,
        answer: str,
        metadata: Optional[Dict[str, Any]] = None,
        confidence: float = 0.93,
    ) -> ToolResult:
        warnings = []
        if dataframe.empty:
            warnings.append("Result dataframe is empty.")
            confidence = min(confidence, 0.55)
        return ToolResult(
            success=True,
            tool_name=self.tool_name,
            data={"rows": _records(dataframe), "row_count": len(dataframe), "columns": [str(column) for column in dataframe.columns]},
            answer=answer,
            table=_records(dataframe),
            confidence=confidence,
            warnings=warnings,
            metadata=metadata or {},
        )

    def _failed(self, message: str, metadata: Optional[Dict[str, Any]] = None) -> ToolResult:
        return ToolResult(
            success=False,
            tool_name=self.tool_name,
            data={},
            answer=None,
            table=[],
            confidence=0.0,
            warnings=[],
            error_msg=message,
            metadata=metadata or {},
        )
