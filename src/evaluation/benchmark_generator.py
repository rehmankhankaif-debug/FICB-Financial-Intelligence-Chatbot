from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd

from src.ingestion.table_loader import load_table
from src.models.table import TableProfile
from src.table_intelligence.pandas_executor import PandasExecutor
from src.table_intelligence.profiler import TableProfiler, normalize_column_name


BENCHMARK_CATEGORIES = [
    "aggregation",
    "filtering",
    "grouping",
    "trend_analysis",
    "comparisons",
    "ranking",
    "top_k",
    "bottom_k",
    "anomaly_detection",
    "correlation_exploration",
    "segmentation",
    "executive_summaries",
    "chart_generation",
    "business_insights",
    "follow_up_questions",
    "multilingual_questions",
]


@dataclass
class ColumnChoices:
    metric: Optional[str] = None
    second_metric: Optional[str] = None
    category: Optional[str] = None
    second_category: Optional[str] = None
    entity: Optional[str] = None
    date: Optional[str] = None


def _dump_model(model: Any) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    if hasattr(model, "dict"):
        return model.dict()
    if isinstance(model, dict):
        return dict(model)
    return {}


def _safe_id(value: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return safe or "table"


def _first(items: Sequence[str]) -> Optional[str]:
    return items[0] if items else None


def _unique_values(dataframe: pd.DataFrame, column: Optional[str], limit: int = 3) -> List[Any]:
    if not column or column not in dataframe.columns:
        return []
    values = []
    for value in dataframe[column].dropna().unique():
        if str(value).strip() == "":
            continue
        values.append(value)
        if len(values) >= limit:
            break
    return values


def _profile_payload(profile: TableProfile) -> Dict[str, Any]:
    payload = _dump_model(profile)
    payload.setdefault("datetime_columns", [])
    payload.setdefault("boolean_columns", [])
    payload.setdefault("unique_values", {})
    payload.setdefault("missing_values", {})
    payload.setdefault("numeric_stats", {})
    return payload


class CsvBenchmarkGenerator:
    """Generate difficult, semantic benchmark questions for uploaded table data."""

    def __init__(self, profiler: Optional[TableProfiler] = None, executor: Optional[PandasExecutor] = None) -> None:
        self.profiler = profiler or TableProfiler()
        self.executor = executor or PandasExecutor()
        self.include_answer_checks = True

    def generate_for_path(
        self,
        path: Path,
        source_id: Optional[str] = None,
        include_answer_checks: bool = True,
    ) -> List[Dict[str, Any]]:
        dataframe = load_table(path)
        resolved_source_id = source_id or _safe_id(path.stem)
        profile = self.profiler.profile(dataframe, source_id=resolved_source_id, filename=path.name)
        return self.generate_for_dataframe(dataframe, profile, include_answer_checks=include_answer_checks)

    def generate_for_paths(self, paths: Iterable[Path], include_answer_checks: bool = True) -> List[Dict[str, Any]]:
        cases: List[Dict[str, Any]] = []
        for path in paths:
            try:
                cases.extend(self.generate_for_path(Path(path), include_answer_checks=include_answer_checks))
            except Exception:
                continue
        return cases

    def generate_for_dataframe(
        self,
        dataframe: pd.DataFrame,
        profile: TableProfile,
        include_answer_checks: bool = True,
    ) -> List[Dict[str, Any]]:
        previous_include_answer_checks = self.include_answer_checks
        self.include_answer_checks = include_answer_checks
        choices = self._choose_columns(profile)
        cases: List[Dict[str, Any]] = []
        try:
            for builder in [
                self._aggregation_case,
                self._filtering_case,
                self._grouping_case,
                self._trend_case,
                self._comparison_case,
                self._ranking_case,
                self._top_k_case,
                self._bottom_k_case,
                self._anomaly_case,
                self._correlation_case,
                self._segmentation_case,
                self._executive_summary_case,
                self._chart_case,
                self._business_insight_case,
                self._follow_up_case,
                self._multilingual_case,
            ]:
                case = builder(dataframe, profile, choices)
                if case:
                    cases.append(case)
        finally:
            self.include_answer_checks = previous_include_answer_checks
        return cases

    def write_cases(self, cases: List[Dict[str, Any]], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"cases": cases}, indent=2, ensure_ascii=True), encoding="utf-8")

    def _choose_columns(self, profile: TableProfile) -> ColumnChoices:
        metric_candidates = self._ordered_metrics(profile)
        category_candidates = self._ordered_categories(profile)
        entity_candidates = [
            column
            for column in profile.entity_candidate_columns
            if column in profile.columns and not self._is_weak_entity_column(column)
        ]
        entity = _first(entity_candidates) or _first(category_candidates)
        return ColumnChoices(
            metric=_first(metric_candidates),
            second_metric=metric_candidates[1] if len(metric_candidates) > 1 else None,
            category=_first(category_candidates),
            second_category=category_candidates[1] if len(category_candidates) > 1 else None,
            entity=entity,
            date=self._date_like_column(profile),
        )

    def _ordered_metrics(self, profile: TableProfile) -> List[str]:
        candidates = list(dict.fromkeys(profile.metric_candidate_columns + profile.numeric_columns))
        weak_tokens = {"id", "code", "number", "reg", "registration"}

        def score(column: str) -> int:
            normalized = normalize_column_name(column)
            tokens = set(normalized.split("_"))
            value = 0
            if any(token in normalized for token in ["price", "sales", "revenue", "profit", "amount", "cost", "spend"]):
                value += 5
            if any(token in normalized for token in ["runs", "rate", "quantity", "margin", "km", "mileage"]):
                value += 4
            if tokens.intersection(weak_tokens) or normalized.endswith("_id"):
                value -= 5
            return value

        return sorted(candidates, key=score, reverse=True)

    def _ordered_categories(self, profile: TableProfile) -> List[str]:
        candidates = list(dict.fromkeys(profile.categorical_columns + profile.entity_candidate_columns + profile.result_candidate_columns))
        weak_tokens = {"id", "title", "description", "address", "url", "number", "model"}
        candidates = [
            column
            for column in candidates
            if column in profile.columns
            and column not in profile.numeric_columns
            and not set(normalize_column_name(column).split("_")).intersection(weak_tokens)
        ]
        candidates = [
            column
            for column in candidates
            if 1 < self._known_unique_count(profile, column) <= 25
        ]

        def score(column: str) -> int:
            normalized = normalize_column_name(column)
            unique_count = self._known_unique_count(profile, column)
            value = 0
            if any(term in normalized for term in ["fuel", "transmission", "segment", "category", "type", "region"]):
                value += 10
            if any(term in normalized for term in ["ownership", "gender", "channel", "status", "winner", "result"]):
                value += 8
            if any(term in normalized for term in ["brand", "product", "customer", "team", "batter"]):
                value += 6
            value += max(0, 8 - abs(unique_count - 5))
            return value

        return sorted(candidates, key=score, reverse=True)

    def _known_unique_count(self, profile: TableProfile, column: str) -> int:
        values = profile.unique_values.get(column) or profile.sample_values.get(column) or []
        return len([value for value in values if value is not None])

    def _is_weak_entity_column(self, column: str) -> bool:
        normalized = normalize_column_name(column)
        weak_tokens = {"id", "code", "description", "transaction", "url"}
        return bool(set(normalized.split("_")).intersection(weak_tokens)) or normalized.endswith("_id") or normalized.endswith("id")

    def _date_like_column(self, profile: TableProfile) -> Optional[str]:
        if profile.datetime_columns:
            return profile.datetime_columns[0]
        for column in profile.columns:
            normalized = normalize_column_name(column)
            if any(term in normalized for term in ["date", "month", "quarter", "year", "period"]):
                return column
        return None

    def _base_case(
        self,
        profile: TableProfile,
        category: str,
        query: str,
        expected: Dict[str, Any],
        language: str = "en",
        csv_case: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        source_id = profile.source_id or _safe_id(profile.filename or "table")
        case = {
            "id": "{0}_{1}_{2}".format(_safe_id(profile.filename or source_id), category, len(query)),
            "category": category,
            "query": query,
            "language": language,
            "available_sources": [
                {"source_id": source_id, "filename": profile.filename or "{0}.csv".format(source_id), "file_type": "csv"}
            ],
            "table_profiles": [_profile_payload(profile)],
            "expected": {
                "intent": "table_analysis",
                "source_id": source_id,
                "tools": ["table_analysis_tool"],
                **expected,
            },
        }
        if csv_case:
            case["csv_case"] = csv_case
        return case

    def _operation_case(self, dataframe: pd.DataFrame, operation: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not self.include_answer_checks:
            return None
        sampled_dataframe = dataframe.head(200).copy()
        result = self.executor.execute(sampled_dataframe, operation)
        if not result.success:
            return None
        return {
            "data": sampled_dataframe.to_dict(orient="records"),
            "operation": operation,
            "expected": {"data": result.data, "table": result.table},
        }

    def _group_aggregation(self, metric: Optional[str]) -> str:
        normalized = normalize_column_name(metric or "")
        additive_terms = {"amount", "cost", "expense", "profit", "quantity", "revenue", "sales", "spend", "total"}
        if any(term in normalized for term in additive_terms):
            return "sum"
        return "mean"

    def _aggregation_case(self, dataframe: pd.DataFrame, profile: TableProfile, choices: ColumnChoices) -> Optional[Dict[str, Any]]:
        if not choices.metric:
            return None
        operation = {"operation": "aggregate", "column": choices.metric, "agg": "mean", "alias": "mean_{0}".format(choices.metric)}
        return self._base_case(
            profile,
            "aggregation",
            "Overall {0} ka scene kaisa hai? Average batao.".format(choices.metric),
            {
                "rewritten_contains": [choices.metric],
                "metrics_contains": [choices.metric],
                "aggregations_contains": ["mean"],
            },
            language="hi-en",
            csv_case=self._operation_case(dataframe, operation),
        )

    def _filtering_case(self, dataframe: pd.DataFrame, profile: TableProfile, choices: ColumnChoices) -> Optional[Dict[str, Any]]:
        values = _unique_values(dataframe, choices.category, limit=1)
        if not choices.category or not values:
            return None
        expected = {
            "rewritten_contains": [str(values[0])],
            "entities_contains": [str(values[0])],
            "aggregations_contains": ["count"],
        }
        operation = {
            "operation": "aggregate",
            "column": None,
            "agg": "count",
            "alias": "count_rows",
            "filters": [{"column": choices.category, "operator": "equals", "value": values[0]}],
        }
        return self._base_case(
            profile,
            "filtering",
            "If {0} is considered separately, records ka volume kitna hai?".format(values[0]),
            expected,
            language="hi-en",
            csv_case=self._operation_case(dataframe, operation),
        )

    def _grouping_case(self, dataframe: pd.DataFrame, profile: TableProfile, choices: ColumnChoices) -> Optional[Dict[str, Any]]:
        if not choices.category or not choices.second_category:
            return None
        operation = {
            "operation": "groupby",
            "group_by": [choices.category, choices.second_category],
            "aggregations": [{"column": None, "agg": "count", "alias": "count_rows"}],
        }
        return self._base_case(
            profile,
            "grouping",
            "Does {0} preference change across {1}?".format(choices.category, choices.second_category),
            {
                "grouping_contains": [choices.category, choices.second_category],
                "aggregations_contains": ["count"],
            },
            csv_case=self._operation_case(dataframe, operation),
        )

    def _trend_case(self, dataframe: pd.DataFrame, profile: TableProfile, choices: ColumnChoices) -> Optional[Dict[str, Any]]:
        group = choices.date or choices.category
        if not group or not choices.metric:
            return None
        aggregation = self._group_aggregation(choices.metric)
        alias = "{0}_{1}".format(aggregation, choices.metric)
        query = (
            "Which {0} performed best by total {1}?".format(group, choices.metric)
            if aggregation == "sum"
            else "Which {0} has the highest average {1}?".format(group, choices.metric)
        )
        operation = {
            "operation": "groupby",
            "group_by": [group],
            "aggregations": [{"column": choices.metric, "agg": aggregation, "alias": alias}],
            "sort_by": alias,
            "ascending": False,
            "limit": 10,
        }
        return self._base_case(
            profile,
            "trend_analysis",
            query,
            {
                "metrics_contains": [choices.metric],
                "grouping_contains": [group],
                "aggregations_contains": [aggregation],
                "sorting_direction": "desc",
            },
            csv_case=self._operation_case(dataframe, operation),
        )

    def _comparison_case(self, dataframe: pd.DataFrame, profile: TableProfile, choices: ColumnChoices) -> Optional[Dict[str, Any]]:
        values = _unique_values(dataframe, choices.category, limit=2)
        if not choices.category or len(values) < 2:
            return None
        operation = {
            "operation": "groupby",
            "group_by": [choices.category],
            "aggregations": [{"column": choices.metric, "agg": "mean", "alias": "mean_{0}".format(choices.metric)}]
            if choices.metric
            else [{"column": None, "agg": "count", "alias": "count_rows"}],
            "filters": [{"column": choices.category, "operator": "in", "value": values}],
        }
        return self._base_case(
            profile,
            "comparisons",
            "Compare {0} and {1} across {2}.".format(values[0], values[1], choices.metric or "overall volume"),
            {
                "entities_contains": [str(values[0]), str(values[1])],
                "grouping_contains": [choices.category],
            },
            csv_case=self._operation_case(dataframe, operation),
        )

    def _ranking_case(self, dataframe: pd.DataFrame, profile: TableProfile, choices: ColumnChoices) -> Optional[Dict[str, Any]]:
        if not choices.entity or not choices.metric:
            return None
        aggregation = self._group_aggregation(choices.metric)
        alias = "{0}_{1}".format(aggregation, choices.metric)
        query = (
            "Kaunse {0} sabse zyada contribute kar rahe hain total {1} mein?".format(choices.entity, choices.metric)
            if aggregation == "sum"
            else "Kaunse {0} highest average {1} dikha rahe hain?".format(choices.entity, choices.metric)
        )
        operation = {
            "operation": "groupby",
            "group_by": [choices.entity],
            "aggregations": [{"column": choices.metric, "agg": aggregation, "alias": alias}],
            "sort_by": alias,
            "ascending": False,
            "limit": 10,
        }
        return self._base_case(
            profile,
            "ranking",
            query,
            {
                "metrics_contains": [choices.metric],
                "grouping_contains": [choices.entity],
                "aggregations_contains": [aggregation],
                "sorting_direction": "desc",
            },
            language="hi-en",
            csv_case=self._operation_case(dataframe, operation),
        )

    def _top_k_case(self, dataframe: pd.DataFrame, profile: TableProfile, choices: ColumnChoices) -> Optional[Dict[str, Any]]:
        if not choices.entity or not choices.metric:
            return None
        aggregation = self._group_aggregation(choices.metric)
        alias = "{0}_{1}".format(aggregation, choices.metric)
        query = (
            "Show top 5 {0} by total {1}.".format(choices.entity, choices.metric)
            if aggregation == "sum"
            else "Show top 5 {0} by average {1}.".format(choices.entity, choices.metric)
        )
        operation = {
            "operation": "groupby",
            "group_by": [choices.entity],
            "aggregations": [{"column": choices.metric, "agg": aggregation, "alias": alias}],
            "sort_by": alias,
            "ascending": False,
            "limit": 5,
        }
        return self._base_case(
            profile,
            "top_k",
            query,
            {
                "metrics_contains": [choices.metric],
                "grouping_contains": [choices.entity],
                "aggregations_contains": [aggregation],
                "limit": 5,
                "sorting_direction": "desc",
            },
            csv_case=self._operation_case(dataframe, operation),
        )

    def _bottom_k_case(self, dataframe: pd.DataFrame, profile: TableProfile, choices: ColumnChoices) -> Optional[Dict[str, Any]]:
        if not choices.entity or not choices.metric:
            return None
        aggregation = self._group_aggregation(choices.metric)
        alias = "{0}_{1}".format(aggregation, choices.metric)
        query = (
            "Which 5 {0} are weakest by total {1}?".format(choices.entity, choices.metric)
            if aggregation == "sum"
            else "Which 5 {0} are weakest by average {1}?".format(choices.entity, choices.metric)
        )
        operation = {
            "operation": "groupby",
            "group_by": [choices.entity],
            "aggregations": [{"column": choices.metric, "agg": aggregation, "alias": alias}],
            "sort_by": alias,
            "ascending": True,
            "limit": 5,
        }
        return self._base_case(
            profile,
            "bottom_k",
            query,
            {
                "metrics_contains": [choices.metric],
                "grouping_contains": [choices.entity],
                "aggregations_contains": [aggregation],
                "limit": 5,
                "sorting_direction": "asc",
            },
            csv_case=self._operation_case(dataframe, operation),
        )

    def _anomaly_case(self, dataframe: pd.DataFrame, profile: TableProfile, choices: ColumnChoices) -> Optional[Dict[str, Any]]:
        if not choices.category or not choices.second_category:
            return None
        return self._base_case(
            profile,
            "anomaly_detection",
            "Identify unusual trends involving {0}, {1}, and {2}.".format(
                choices.category,
                choices.second_category,
                choices.metric or "record volume",
            ),
            {
                "grouping_contains": [choices.category, choices.second_category],
                "metrics_contains": [choices.metric] if choices.metric else [],
            },
        )

    def _correlation_case(self, dataframe: pd.DataFrame, profile: TableProfile, choices: ColumnChoices) -> Optional[Dict[str, Any]]:
        if not choices.metric or not choices.second_metric:
            return None
        operation = {"operation": "correlation", "columns": [choices.metric, choices.second_metric]}
        return self._base_case(
            profile,
            "correlation_exploration",
            "Are {0} and {1} related, or is there a visible relationship?".format(choices.metric, choices.second_metric),
            {
                "metrics_contains": [choices.metric, choices.second_metric],
                "analysis_type": "correlation",
            },
            csv_case=self._operation_case(dataframe, operation),
        )

    def _segmentation_case(self, dataframe: pd.DataFrame, profile: TableProfile, choices: ColumnChoices) -> Optional[Dict[str, Any]]:
        if not choices.metric or not choices.category:
            return None
        median_value = profile.numeric_stats.get(choices.metric, {}).get("median")
        if median_value is None:
            return None
        operation = {
            "operation": "groupby",
            "group_by": [choices.category],
            "aggregations": [{"column": None, "agg": "count", "alias": "count_rows"}],
            "filters": [{"column": choices.metric, "operator": "gt", "value": median_value}],
        }
        return self._base_case(
            profile,
            "segmentation",
            "If I only consider records above the median {0}, does {1} distribution change?".format(
                choices.metric,
                choices.category,
            ),
            {
                "metrics_contains": [choices.metric],
                "grouping_contains": [choices.category],
                "filters_contain_operator": ["gt"],
            },
            csv_case=self._operation_case(dataframe, operation),
        )

    def _executive_summary_case(self, dataframe: pd.DataFrame, profile: TableProfile, choices: ColumnChoices) -> Optional[Dict[str, Any]]:
        return self._base_case(
            profile,
            "executive_summaries",
            "Give me an executive summary of this dataset with key risks and opportunities.",
            {
                "metrics_contains": ["dataset_summary"],
            },
        )

    def _chart_case(self, dataframe: pd.DataFrame, profile: TableProfile, choices: ColumnChoices) -> Optional[Dict[str, Any]]:
        if not choices.category:
            return None
        operation = {
            "operation": "groupby",
            "group_by": [choices.category] + ([choices.second_category] if choices.second_category else []),
            "aggregations": [{"column": None, "agg": "count", "alias": "count_rows"}],
        }
        return self._base_case(
            profile,
            "chart_generation",
            "Show the relationship between {0} and {1} using a chart.".format(
                choices.category,
                choices.second_category or "record count",
            ),
            {
                "intent": "chart_request",
                "tools": ["table_analysis_tool", "chart_tool"],
                "grouping_contains": [choices.category] + ([choices.second_category] if choices.second_category else []),
                "chart_requested": True,
            },
            csv_case=self._operation_case(dataframe, operation),
        )

    def _business_insight_case(self, dataframe: pd.DataFrame, profile: TableProfile, choices: ColumnChoices) -> Optional[Dict[str, Any]]:
        return self._base_case(
            profile,
            "business_insights",
            "What hidden patterns and business insights exist in this data?",
            {
                "metrics_contains": ["dataset_summary"],
            },
        )

    def _follow_up_case(self, dataframe: pd.DataFrame, profile: TableProfile, choices: ColumnChoices) -> Optional[Dict[str, Any]]:
        if not choices.category:
            return None
        return self._base_case(
            profile,
            "follow_up_questions",
            "Now break that down by {0} and explain what changed.".format(choices.category),
            {
                "grouping_contains": [choices.category],
            },
        )

    def _multilingual_case(self, dataframe: pd.DataFrame, profile: TableProfile, choices: ColumnChoices) -> Optional[Dict[str, Any]]:
        if not choices.category:
            return None
        return self._base_case(
            profile,
            "multilingual_questions",
            "{0} ka overall trend kya hai? Chart bhi dikhao.".format(choices.category),
            {
                "intent": "chart_request",
                "tools": ["table_analysis_tool", "chart_tool"],
                "grouping_contains": [choices.category],
                "chart_requested": True,
            },
            language="hi-en",
        )


def generate_benchmark_file(
    paths: Iterable[Path],
    output_path: Path,
    include_answer_checks: bool = True,
) -> List[Dict[str, Any]]:
    generator = CsvBenchmarkGenerator()
    cases = generator.generate_for_paths(paths, include_answer_checks=include_answer_checks)
    generator.write_cases(cases, output_path)
    return cases
