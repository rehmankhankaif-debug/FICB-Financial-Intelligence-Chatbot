from __future__ import annotations

import pandas as pd

from src.table_intelligence.pandas_executor import PandasExecutor
from src.table_intelligence.validator import TableResultValidator


def _sales_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "region": ["North", "South", "North", "West"],
            "product": ["A", "A", "B", "B"],
            "sales": [100, 200, 150, 50],
            "profit": [20, 50, 30, 10],
        }
    )


def test_aggregate_sum() -> None:
    result = PandasExecutor().execute(
        _sales_dataframe(),
        {"operation": "aggregate", "column": "sales", "agg": "sum"},
    )

    assert result.success is True
    assert result.data["value"] == 500.0
    assert result.data["aggregation"] == "sum"


def test_filtering_operation() -> None:
    result = PandasExecutor().execute(
        _sales_dataframe(),
        {"operation": "filter", "filters": [{"column": "region", "operator": "equals", "value": "North"}]},
    )

    assert result.success is True
    assert result.data["row_count"] == 2
    assert all(row["region"] == "North" for row in result.table)


def test_groupby_sum_and_sort() -> None:
    result = PandasExecutor().execute(
        _sales_dataframe(),
        {
            "operation": "groupby",
            "group_by": ["region"],
            "aggregations": [{"column": "sales", "agg": "sum", "alias": "total_sales"}],
            "sort_by": "total_sales",
            "ascending": False,
        },
    )

    assert result.success is True
    assert result.table[0]["region"] == "North"
    assert result.table[0]["total_sales"] == 250


def test_top_k() -> None:
    result = PandasExecutor().execute(
        _sales_dataframe(),
        {"operation": "top_k", "sort_by": "profit", "k": 2},
    )

    assert result.success is True
    assert len(result.table) == 2
    assert result.table[0]["profit"] == 50


def test_bottom_k() -> None:
    result = PandasExecutor().execute(
        _sales_dataframe(),
        {"operation": "bottom_k", "sort_by": "profit", "k": 1},
    )

    assert result.success is True
    assert result.table[0]["profit"] == 10


def test_mean_median_count_nunique_min_max() -> None:
    executor = PandasExecutor()

    mean_result = executor.execute(_sales_dataframe(), {"operation": "aggregate", "column": "profit", "agg": "mean"})
    median_result = executor.execute(_sales_dataframe(), {"operation": "aggregate", "column": "profit", "agg": "median"})
    count_result = executor.execute(_sales_dataframe(), {"operation": "aggregate", "column": "profit", "agg": "count"})
    nunique_result = executor.execute(_sales_dataframe(), {"operation": "aggregate", "column": "region", "agg": "nunique"})
    min_result = executor.execute(_sales_dataframe(), {"operation": "aggregate", "column": "profit", "agg": "min"})
    max_result = executor.execute(_sales_dataframe(), {"operation": "aggregate", "column": "profit", "agg": "max"})

    assert mean_result.data["value"] == 27.5
    assert median_result.data["value"] == 25.0
    assert count_result.data["value"] == 4
    assert nunique_result.data["value"] == 3
    assert min_result.data["value"] == 10.0
    assert max_result.data["value"] == 50.0


def test_compare_operation() -> None:
    result = PandasExecutor().execute(
        _sales_dataframe(),
        {
            "operation": "compare",
            "column": "sales",
            "agg": "sum",
            "left_filters": [{"column": "region", "operator": "equals", "value": "North"}],
            "right_filters": [{"column": "region", "operator": "equals", "value": "South"}],
        },
    )

    assert result.success is True
    assert result.data["left_value"] == 250.0
    assert result.data["right_value"] == 200.0
    assert result.data["difference"] == 50.0


def test_correlation_operation_returns_pearson_value() -> None:
    result = PandasExecutor().execute(
        pd.DataFrame({"price": [10, 20, 30], "cost": [1, 2, 3]}),
        {"operation": "correlation", "columns": ["price", "cost"]},
    )

    assert result.success is True
    assert result.data["column_x"] == "price"
    assert result.data["column_y"] == "cost"
    assert result.data["correlation"] == 1.0


def test_invalid_column_returns_failed_tool_result() -> None:
    result = PandasExecutor().execute(
        _sales_dataframe(),
        {"operation": "aggregate", "column": "missing", "agg": "sum"},
    )

    assert result.success is False
    assert result.error_msg


def test_empty_dataframe_result_can_be_validated() -> None:
    result = PandasExecutor().execute(
        _sales_dataframe(),
        {"operation": "filter", "filters": [{"column": "region", "operator": "equals", "value": "Unknown"}]},
    )
    validation = TableResultValidator().validate(result)

    assert result.success is True
    assert result.table == []
    assert validation.is_valid is False
    assert "Result table is empty." in validation.issues
