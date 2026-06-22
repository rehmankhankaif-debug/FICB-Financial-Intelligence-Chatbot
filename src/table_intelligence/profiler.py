from __future__ import annotations

import re
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.models.table import TableProfile
from src.utils.errors import IngestionError


ENTITY_NAME_HINTS = {
    "account",
    "batter",
    "batsman",
    "brand",
    "city",
    "client",
    "company",
    "country",
    "customer",
    "employee",
    "entity",
    "name",
    "player",
    "product",
    "region",
    "team",
    "user",
    "vendor",
}

METRIC_HINTS = {
    "amount",
    "average",
    "balance",
    "cost",
    "earnings",
    "expense",
    "income",
    "margin",
    "median",
    "price",
    "profit",
    "quantity",
    "rate",
    "revenue",
    "runs",
    "sales",
    "score",
    "sr",
    "strike_rate",
    "total",
    "value",
}

RESULT_HINTS = {
    "decision",
    "loss",
    "outcome",
    "result",
    "status",
    "verdict",
    "win",
    "winner",
}

BOOLEAN_TRUE_FALSE_VALUES = {"true", "false", "yes", "no", "y", "n"}
LOW_CARDINALITY_LIMIT = 20


def normalize_column_name(column_name: Any) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", str(column_name).strip().lower())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized


def _to_builtin(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _unique_non_null_values(series: pd.Series, limit: int = 10) -> List[Any]:
    values = []
    for value in series.dropna().unique()[:limit]:
        values.append(_to_builtin(value))
    return values


def _looks_like_datetime(series: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if pd.api.types.is_numeric_dtype(series) or series.dropna().empty:
        return False
    sample = series.dropna().astype(str).head(50)
    if sample.empty:
        return False
    combined_sample = " ".join(sample.tolist()).lower()
    date_like_pattern = r"(\d{4}[-/]\d{1,2})|(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})|(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)"
    if re.search(date_like_pattern, combined_sample) is None:
        return False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        parsed = pd.to_datetime(sample, errors="coerce")
    return float(parsed.notna().mean()) >= 0.8


def _looks_like_boolean(series: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(series):
        return True
    non_null = series.dropna()
    if non_null.empty:
        return False
    values = {str(value).strip().lower() for value in non_null.unique()}
    if values and values.issubset(BOOLEAN_TRUE_FALSE_VALUES):
        return True
    return values.issubset({"0", "1"}) and 0 < len(values) <= 2


def _name_contains_any(normalized_name: str, hints: set) -> bool:
    tokens = set(normalized_name.split("_"))
    return bool(tokens.intersection(hints)) or any(hint in normalized_name for hint in hints)


def _looks_like_person_or_entity_values(series: pd.Series) -> bool:
    sample = [str(value).strip() for value in series.dropna().head(25)]
    if not sample:
        return False
    multi_word_count = sum(1 for value in sample if len(value.split()) >= 2)
    title_like_count = sum(1 for value in sample if re.match(r"^[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)+$", value))
    return multi_word_count / float(len(sample)) >= 0.4 or title_like_count / float(len(sample)) >= 0.3


class TableProfiler:
    def profile(
        self,
        dataframe: pd.DataFrame,
        source_id: str = "",
        filename: str = "",
    ) -> TableProfile:
        if dataframe is None:
            raise IngestionError("Cannot profile a missing dataframe.")
        if not isinstance(dataframe, pd.DataFrame):
            raise IngestionError(
                "TableProfiler requires a pandas DataFrame.",
                metadata={"received_type": type(dataframe).__name__},
            )

        normalized_columns: Dict[str, str] = {}
        dtypes: Dict[str, str] = {}
        numeric_columns: List[str] = []
        categorical_columns: List[str] = []
        datetime_columns: List[str] = []
        boolean_columns: List[str] = []
        entity_candidate_columns: List[str] = []
        metric_candidate_columns: List[str] = []
        result_candidate_columns: List[str] = []
        sample_values: Dict[str, List[Any]] = {}
        unique_values: Dict[str, List[Any]] = {}
        missing_values: Dict[str, int] = {}
        numeric_stats: Dict[str, Dict[str, float]] = {}

        row_count = len(dataframe)

        for column in dataframe.columns:
            series = dataframe[column]
            column_name = str(column)
            normalized_name = normalize_column_name(column_name)

            normalized_columns[column_name] = normalized_name
            dtypes[column_name] = str(series.dtype)
            missing_values[column_name] = int(series.isna().sum())
            sample_values[column_name] = _unique_non_null_values(series, limit=8)

            unique_count = int(series.dropna().nunique())
            if unique_count <= LOW_CARDINALITY_LIMIT:
                unique_values[column_name] = _unique_non_null_values(series, limit=LOW_CARDINALITY_LIMIT)

            is_numeric = pd.api.types.is_numeric_dtype(series) and not _looks_like_boolean(series)
            is_datetime = _looks_like_datetime(series)
            is_boolean = _looks_like_boolean(series)
            is_categorical = (
                not is_numeric
                and not is_datetime
                and not is_boolean
                or unique_count <= min(LOW_CARDINALITY_LIMIT, max(1, row_count))
            )

            if is_numeric:
                numeric_columns.append(column_name)
                clean_numeric = pd.to_numeric(series, errors="coerce").dropna()
                if not clean_numeric.empty:
                    numeric_stats[column_name] = {
                        "count": float(clean_numeric.count()),
                        "mean": float(clean_numeric.mean()),
                        "median": float(clean_numeric.median()),
                        "min": float(clean_numeric.min()),
                        "max": float(clean_numeric.max()),
                        "sum": float(clean_numeric.sum()),
                    }
            if is_datetime:
                datetime_columns.append(column_name)
            if is_boolean:
                boolean_columns.append(column_name)
            if is_categorical:
                categorical_columns.append(column_name)

            if is_numeric and not normalized_name.endswith("_id") and normalized_name != "id":
                metric_candidate_columns.append(column_name)
            if _name_contains_any(normalized_name, METRIC_HINTS):
                if column_name not in metric_candidate_columns:
                    metric_candidate_columns.append(column_name)

            if _name_contains_any(normalized_name, RESULT_HINTS):
                result_candidate_columns.append(column_name)

            if (
                _name_contains_any(normalized_name, ENTITY_NAME_HINTS)
                or (is_categorical and _looks_like_person_or_entity_values(series))
            ) and column_name not in result_candidate_columns:
                entity_candidate_columns.append(column_name)

        summary_parts = [
            "Rows: {0}, columns: {1}.".format(dataframe.shape[0], dataframe.shape[1]),
            "Entity candidates: {0}.".format(", ".join(entity_candidate_columns) or "none"),
            "Metric candidates: {0}.".format(", ".join(metric_candidate_columns) or "none"),
            "Result/status candidates: {0}.".format(", ".join(result_candidate_columns) or "none"),
        ]

        return TableProfile(
            source_id=source_id,
            filename=filename,
            shape=(int(dataframe.shape[0]), int(dataframe.shape[1])),
            columns=[str(column) for column in dataframe.columns],
            normalized_columns=normalized_columns,
            dtypes=dtypes,
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
            datetime_columns=datetime_columns,
            boolean_columns=boolean_columns,
            entity_candidate_columns=entity_candidate_columns,
            metric_candidate_columns=metric_candidate_columns,
            result_candidate_columns=result_candidate_columns,
            sample_values=sample_values,
            unique_values=unique_values,
            missing_values=missing_values,
            numeric_stats=numeric_stats,
            semantic_summary=" ".join(summary_parts),
        )
