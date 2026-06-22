from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.ingestion.table_loader import load_table
from src.models.query import QueryPlan
from src.models.table import TableProfile
from src.models.tool import ToolResult
from src.table_intelligence.pandas_executor import PandasExecutor
from src.table_intelligence.profiler import TableProfiler
from src.table_intelligence.semantic_column_mapper import SemanticColumnMapper
from src.table_intelligence.validator import TableResultValidator
from src.table_intelligence.value_matcher import ValueMatcher
from src.tools.base import BaseTool


def _payload_to_model(model_class, payload):
    if isinstance(payload, model_class):
        return payload
    if isinstance(payload, dict):
        return model_class(**payload)
    return model_class()


def _dump_model(model: Any) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    if hasattr(model, "dict"):
        return model.dict()
    return {}


def _metric_name(metric: Any) -> str:
    if isinstance(metric, dict):
        return str(metric.get("name") or metric.get("text") or metric.get("metric") or "")
    return str(metric or "")


def _aggregation_name(aggregation: Any) -> str:
    if isinstance(aggregation, dict):
        value = aggregation.get("operation") or aggregation.get("agg") or aggregation.get("aggregation") or aggregation.get("name")
    else:
        value = aggregation
    normalized = str(value or "").lower()
    if normalized in {"average", "avg"}:
        return "mean"
    if normalized in {"maximum", "highest"}:
        return "max"
    if normalized in {"minimum", "lowest"}:
        return "min"
    return normalized or "sum"


TABLE_SUMMARY_METRICS = {"dataset_summary", "table_summary", "key_insights", "insights", "summary"}
TABLE_SUMMARY_SIGNALS = {"insight", "insights", "key finding", "key findings", "overview", "summary"}


class TableAnalysisTool(BaseTool):
    name = "table_analysis_tool"
    description = "Deterministic table analysis over CSV/Excel sources using pandas."
    supported_intents = ["table_analysis", "chart_request", "compare_documents"]
    supported_source_types = ["table", "csv", "xlsx", "xls", "mixed"]
    input_types = ["QueryPlan", "SourceSelection", "DataFrame", "TableProfile"]
    output_types = ["table", "metrics", "metadata"]
    input_requirements = ["query_plan", "dataframe or table or source path"]
    capabilities = ["filter", "aggregate", "group", "rank", "compare_table_metrics"]
    positive_examples = ["average monthly profit batao", "top 5 products by sales", "Virat ke maximum runs batao"]
    negative_examples = ["outline this report"]
    can_chain_before = ["chart_tool", "compare_tool"]
    confidence = 0.88

    def __init__(self) -> None:
        super().__init__()
        self.profiler = TableProfiler()
        self.mapper = SemanticColumnMapper()
        self.value_matcher = ValueMatcher()
        self.executor = PandasExecutor()
        self.validator = TableResultValidator()

    def run(self, input_payload: Dict[str, Any]) -> ToolResult:
        try:
            query_plan = _payload_to_model(QueryPlan, (input_payload or {}).get("query_plan"))
            dataframes = (input_payload or {}).get("dataframes") or {}
            if query_plan.intent == "compare_documents" and isinstance(dataframes, dict) and len(dataframes) > 1:
                return self._run_multi_source(input_payload, query_plan, dataframes)
            dataframe = self._extract_dataframe(input_payload or {})
            if dataframe is None:
                return self._failed("No dataframe, table records, or structured source path was provided.")

            profile = self._extract_profile(input_payload or {}, dataframe)
            if self._is_profile_summary_request(query_plan):
                return self._profile_summary_result(query_plan, dataframe, profile)

            filters, filter_notes = self._build_filters(query_plan, dataframe, profile)
            analytical_result = self._execute_analytical_shortcut_if_needed(query_plan, dataframe, profile, filters)
            if analytical_result is not None:
                analytical_result.warnings = filter_notes + analytical_result.warnings
                return analytical_result

            multi_metric_result = self._execute_multi_metric_if_needed(query_plan, dataframe, profile, filters)
            if multi_metric_result is not None:
                multi_metric_result.warnings = filter_notes + multi_metric_result.warnings
                return multi_metric_result

            operation = self._build_operation(query_plan, profile, filters)
            pandas_result = self.executor.execute(dataframe, operation)

            if not pandas_result.success:
                return ToolResult(
                    success=False,
                    tool_name=self.name,
                    error_msg=pandas_result.error_msg,
                    table=pandas_result.table,
                    confidence=0.0,
                    warnings=pandas_result.warnings + filter_notes,
                    metadata={"operation": operation, "pandas_metadata": pandas_result.metadata},
                )

            validation = self.validator.validate(pandas_result, operation)
            return ToolResult(
                success=validation.is_valid or pandas_result.success,
                tool_name=self.name,
                data=pandas_result.data,
                answer=self._answer_for_result(query_plan, pandas_result, operation),
                table=pandas_result.table,
                chart=None,
                citations=[],
                confidence=min(pandas_result.confidence, validation.confidence if validation.confidence else pandas_result.confidence),
                warnings=filter_notes + pandas_result.warnings + validation.warnings,
                error_msg=None if pandas_result.success else pandas_result.error_msg,
                metadata={
                    "operation": operation,
                    "validation": _dump_model(validation),
                    "profile_summary": profile.semantic_summary,
                    "column_count": len(profile.columns),
                    "pandas_metadata": pandas_result.metadata,
                },
            )
        except Exception as exc:
            return self._failed("Table analysis failed safely: {0}".format(str(exc)))

    def _run_multi_source(
        self,
        payload: Dict[str, Any],
        query_plan: QueryPlan,
        dataframes: Dict[str, Any],
    ) -> ToolResult:
        profiles = payload.get("table_profiles") or {}
        descriptors = payload.get("source_descriptors") or {}
        source_results: List[Dict[str, Any]] = []
        rows: List[Dict[str, Any]] = []
        warnings: List[str] = []
        for source_id, dataframe in dataframes.items():
            if not isinstance(dataframe, pd.DataFrame):
                warnings.append("Structured source {0} was unavailable.".format(source_id))
                continue
            single_payload = dict(payload)
            single_payload.pop("dataframes", None)
            single_payload.pop("table_profiles", None)
            single_payload["dataframe"] = dataframe
            single_payload["table_profile"] = profiles.get(source_id)
            result = self.run(single_payload)
            descriptor = descriptors.get(source_id) or {}
            result_payload = _dump_model(result)
            result_payload["source_id"] = source_id
            result_payload["filename"] = descriptor.get("filename") or source_id
            source_results.append(result_payload)
            rows.append(
                {
                    "source_id": source_id,
                    "filename": descriptor.get("filename") or source_id,
                    "answer": result.answer,
                    "success": result.success,
                }
            )
            warnings.extend(result.warnings or [])
        successful_count = len([item for item in source_results if item.get("success")])
        return ToolResult(
            success=successful_count >= 2,
            tool_name=self.name,
            data={"source_results": source_results, "source_count": len(source_results)},
            answer="Compared deterministic table results across {0} structured sources.".format(successful_count),
            table=rows,
            confidence=self.confidence if successful_count >= 2 else 0.4,
            warnings=warnings,
            error_msg=None if successful_count >= 2 else "At least two structured sources are required.",
            metadata={"multi_source": True, "source_ids": list(dataframes.keys())},
        )

    def _extract_dataframe(self, payload: Dict[str, Any]) -> Optional[pd.DataFrame]:
        value = payload.get("dataframe")
        if value is None:
            value = payload.get("df")
        if isinstance(value, pd.DataFrame):
            return value
        table = payload.get("table")
        if isinstance(table, pd.DataFrame):
            return table
        if isinstance(table, list):
            return pd.DataFrame(table)
        path = payload.get("path") or payload.get("file_path")
        if not path:
            source = payload.get("source_selection") or {}
            if isinstance(source, dict):
                path = source.get("path") or source.get("source_path")
        if path:
            return load_table(Path(path))
        return None

    def _extract_profile(self, payload: Dict[str, Any], dataframe: pd.DataFrame) -> TableProfile:
        profile_payload = payload.get("table_profile")
        if isinstance(profile_payload, TableProfile):
            return profile_payload
        if isinstance(profile_payload, dict):
            return TableProfile(**profile_payload)
        return self.profiler.profile(dataframe)

    def _build_filters(self, query_plan: QueryPlan, dataframe: pd.DataFrame, profile: TableProfile) -> Tuple[List[Dict[str, Any]], List[str]]:
        filters: List[Dict[str, Any]] = []
        notes: List[str] = []
        grouping_columns = self._mapped_grouping_columns(query_plan, profile)

        for item in query_plan.filters:
            field_hint = str(item.get("field") or item.get("field_hint") or item.get("column") or "")
            value = item.get("value")
            if not field_hint or value is None:
                continue
            column_match = self.mapper.match_column(field_hint, profile)
            if not column_match.matched_column:
                notes.append("Could not map filter field: {0}".format(field_hint))
                continue
            filters.append(
                {
                    "column": column_match.matched_column,
                    "operator": item.get("operator", "contains"),
                    "value": value,
                }
            )

        entity_matches_by_column: Dict[str, List[Any]] = {}
        for entity in query_plan.entities:
            entity_value = self._entity_value(entity)
            if not entity_value:
                continue
            match = self._match_entity_value(entity_value, dataframe, profile, self._entity_field_hint(entity))
            if match:
                column, value = match
                entity_matches_by_column.setdefault(column, [])
                if value not in entity_matches_by_column[column]:
                    entity_matches_by_column[column].append(value)

        skipped_grouping_columns = self._skipped_grouping_entity_filter_columns(
            query_plan,
            grouping_columns,
            entity_matches_by_column,
        )
        for column, values in entity_matches_by_column.items():
            if column in skipped_grouping_columns:
                continue
            if len(values) == 1:
                filters.append({"column": column, "operator": "equals", "value": values[0]})
            else:
                filters.append({"column": column, "operator": "in", "value": values})
        return filters, notes

    def _entity_value(self, entity: Any) -> Any:
        if isinstance(entity, dict):
            return entity.get("normalized") or entity.get("value") or entity.get("text") or entity.get("name")
        return entity

    def _entity_field_hint(self, entity: Any) -> str:
        if not isinstance(entity, dict):
            return ""
        return str(entity.get("field") or entity.get("field_hint") or entity.get("column") or "")

    def _mapped_grouping_columns(self, query_plan: QueryPlan, profile: TableProfile) -> List[str]:
        grouping = [self._map_column(group, profile) for group in query_plan.grouping]
        return [group for group in grouping if group]

    def _skipped_grouping_entity_filter_columns(
        self,
        query_plan: QueryPlan,
        grouping_columns: List[str],
        entity_matches_by_column: Dict[str, List[Any]],
    ) -> set:
        if len(grouping_columns) < 2:
            return set()
        metrics = {_metric_name(metric).lower() for metric in query_plan.metrics if _metric_name(metric)}
        aggregations = {_aggregation_name(item) for item in query_plan.aggregations}
        if "count" not in metrics and "count" not in aggregations:
            return set()
        text = "{0} {1}".format(query_plan.original_query or "", query_plan.rewritten_query or "").lower()
        if "etc" in text:
            grouping_match_counts = [
                (column, len(entity_matches_by_column.get(column, [])))
                for column in grouping_columns
                if entity_matches_by_column.get(column)
            ]
            if not grouping_match_counts:
                return set()
            highest_count = max(count for _, count in grouping_match_counts)
            if highest_count > 1:
                return {column for column, count in grouping_match_counts if count == highest_count}
            return set()
        broad_breakdown_signals = {"all", "breakdown", "distribution"}
        if any(signal in text for signal in broad_breakdown_signals):
            return set(grouping_columns)
        return set()

    def _match_entity_value(
        self,
        entity_value: Any,
        dataframe: pd.DataFrame,
        profile: TableProfile,
        field_hint: str = "",
    ) -> Optional[Tuple[str, Any]]:
        if field_hint:
            column_match = self.mapper.match_column(field_hint, profile)
            if not column_match.matched_column:
                return None
            match = self.value_matcher.match_value(entity_value, column_match.matched_column, dataframe)
            if match.matched_value is None:
                return None
            return match.matched_column, match.matched_value

        candidates = list(dict.fromkeys(profile.entity_candidate_columns + profile.categorical_columns + profile.result_candidate_columns))
        best = None
        for column in candidates:
            match = self.value_matcher.match_value(entity_value, column, dataframe)
            if match.matched_value is not None and (best is None or match.confidence > best.confidence):
                best = match
        if best is None:
            return None
        return best.matched_column, best.matched_value

    def _build_operation(self, query_plan: QueryPlan, profile: TableProfile, filters: List[Dict[str, Any]]) -> Dict[str, Any]:
        metrics = [_metric_name(metric) for metric in query_plan.metrics if _metric_name(metric)]
        metric_name = metrics[0] if metrics else "count"
        metric_column = None if metric_name == "count" else self._map_column(metric_name, profile)
        aggregations = [_aggregation_name(item) for item in query_plan.aggregations]
        aggregation = aggregations[0] if aggregations else ("count" if metric_name == "count" else "sum")
        grouping = [self._map_column(group, profile) for group in query_plan.grouping]
        grouping = [group for group in grouping if group]
        limit = query_plan.limit

        if grouping:
            alias = "{0}_{1}".format(aggregation, metric_column or "rows")
            sort_by = query_plan.sorting.get("field") or query_plan.sorting.get("sort_by")
            if not sort_by and aggregation == "count":
                sort_by = alias
            sort_direction = str(query_plan.sorting.get("direction") or "").lower()
            return {
                "operation": "groupby",
                "group_by": grouping,
                "aggregations": [
                    {
                        "column": metric_column,
                        "agg": aggregation,
                        "alias": alias,
                    }
                ],
                "sort_by": sort_by or (alias if limit else None),
                "ascending": sort_direction == "asc",
                "limit": limit,
                "filters": filters,
            }

        if limit and metric_column:
            operation_name = "top_k"
            if query_plan.sorting.get("direction") == "asc":
                operation_name = "bottom_k"
            return {
                "operation": operation_name,
                "column": metric_column,
                "sort_by": metric_column,
                "k": limit,
                "filters": filters,
            }

        return {
            "operation": "aggregate",
            "column": metric_column,
            "agg": aggregation,
            "filters": filters,
        }

    def _is_profile_summary_request(self, query_plan: QueryPlan) -> bool:
        metrics = {_metric_name(metric).lower() for metric in query_plan.metrics if _metric_name(metric)}
        text = "{0} {1}".format(query_plan.original_query or "", query_plan.rewritten_query or "").lower()
        return bool(metrics.intersection(TABLE_SUMMARY_METRICS)) or (
            not metrics and any(signal in text for signal in TABLE_SUMMARY_SIGNALS)
        )

    def _profile_summary_result(self, query_plan: QueryPlan, dataframe: pd.DataFrame, profile: TableProfile) -> ToolResult:
        rows: List[Dict[str, Any]] = [
            {"insight": "Dataset size", "value": "{0} rows x {1} columns".format(profile.shape[0], profile.shape[1])},
            {"insight": "Numeric columns", "value": ", ".join(profile.numeric_columns[:10]) or "none"},
            {"insight": "Categorical columns", "value": ", ".join(profile.categorical_columns[:10]) or "none"},
            {"insight": "Metric candidates", "value": ", ".join(profile.metric_candidate_columns[:10]) or "none"},
            {"insight": "Entity candidates", "value": ", ".join(profile.entity_candidate_columns[:10]) or "none"},
        ]

        top_categories: Dict[str, Any] = {}
        for column in profile.categorical_columns[:8]:
            series = dataframe[column].dropna() if column in dataframe.columns else pd.Series(dtype=object)
            if series.empty:
                continue
            counts = series.value_counts(dropna=True)
            if counts.empty:
                continue
            top_value = counts.index[0]
            top_count = int(counts.iloc[0])
            top_categories[column] = {"value": top_value, "count": top_count}
            rows.append(
                {
                    "insight": "Most common {0}".format(column),
                    "value": "{0} ({1})".format(top_value, top_count),
                }
            )

        numeric_highlights: Dict[str, Any] = {}
        for column in profile.numeric_columns[:8]:
            stats = profile.numeric_stats.get(column) or {}
            if not stats:
                continue
            numeric_highlights[column] = {
                "mean": stats.get("mean"),
                "min": stats.get("min"),
                "max": stats.get("max"),
            }
            rows.append(
                {
                    "insight": "{0} range".format(column),
                    "value": "min {0}, mean {1:.2f}, max {2}".format(
                        stats.get("min"),
                        float(stats.get("mean") or 0.0),
                        stats.get("max"),
                    ),
                }
            )

        missing_columns = {
            column: count
            for column, count in profile.missing_values.items()
            if count
        }
        if missing_columns:
            rows.append(
                {
                    "insight": "Columns with missing values",
                    "value": ", ".join("{0}: {1}".format(column, count) for column, count in list(missing_columns.items())[:8]),
                }
            )

        answer_parts = [
            "I found {0} rows and {1} columns.".format(profile.shape[0], profile.shape[1]),
            "Main metric candidates: {0}.".format(", ".join(profile.metric_candidate_columns[:6]) or "none"),
            "Main categorical/entity fields: {0}.".format(", ".join((profile.entity_candidate_columns + profile.categorical_columns)[:6]) or "none"),
        ]
        if top_categories:
            category_bits = [
                "{0}: {1} ({2})".format(column, item["value"], item["count"])
                for column, item in list(top_categories.items())[:3]
            ]
            answer_parts.append("Top category signals: {0}.".format("; ".join(category_bits)))

        return ToolResult(
            success=True,
            tool_name=self.name,
            data={
                "row_count": profile.shape[0],
                "column_count": profile.shape[1],
                "numeric_columns": profile.numeric_columns,
                "categorical_columns": profile.categorical_columns,
                "metric_candidate_columns": profile.metric_candidate_columns,
                "entity_candidate_columns": profile.entity_candidate_columns,
                "top_categories": top_categories,
                "numeric_highlights": numeric_highlights,
                "missing_values": missing_columns,
            },
            answer=" ".join(answer_parts),
            table=rows,
            confidence=0.88 if profile.shape[0] else 0.55,
            warnings=[] if profile.shape[0] else ["The uploaded table is empty, so insights are limited."],
            metadata={
                "operation": {"operation": "table_profile_summary"},
                "profile_summary": profile.semantic_summary,
                "query_intent": query_plan.intent,
            },
        )

    def _execute_analytical_shortcut_if_needed(
        self,
        query_plan: QueryPlan,
        dataframe: pd.DataFrame,
        profile: TableProfile,
        filters: List[Dict[str, Any]],
    ) -> Optional[ToolResult]:
        for handler in [
            self._execute_percentage_if_needed,
            self._execute_correlation_if_needed,
            self._execute_median_segment_if_needed,
            self._execute_entity_comparison_if_needed,
        ]:
            result = handler(query_plan, dataframe, profile, filters)
            if result is not None:
                return result
        return None

    def _execute_percentage_if_needed(
        self,
        query_plan: QueryPlan,
        dataframe: pd.DataFrame,
        profile: TableProfile,
        filters: List[Dict[str, Any]],
    ) -> Optional[ToolResult]:
        text = self._plan_text(query_plan)
        if not any(signal in text for signal in {"percent", "percentage", "proportion", "share"}):
            return None
        matches = self._matched_entities(query_plan, dataframe, profile)
        if len(matches) < 2:
            return None

        ordered_matches = sorted(matches, key=lambda item: item["position"] if item["position"] >= 0 else 10**6)
        denominator_match = ordered_matches[0]
        numerator_match = ordered_matches[-1]
        if denominator_match["column"] == numerator_match["column"]:
            return None

        denominator_filters = [{"column": denominator_match["column"], "operator": "equals", "value": denominator_match["value"]}]
        numerator_filters = denominator_filters + [
            {"column": numerator_match["column"], "operator": "equals", "value": numerator_match["value"]}
        ]
        denominator_df = self.executor._apply_filters(dataframe, denominator_filters)
        numerator_df = self.executor._apply_filters(dataframe, numerator_filters)
        denominator_count = int(len(denominator_df))
        numerator_count = int(len(numerator_df))
        percentage = (numerator_count / float(denominator_count) * 100.0) if denominator_count else None
        row = {
            "base_column": denominator_match["column"],
            "base_value": denominator_match["value"],
            "target_column": numerator_match["column"],
            "target_value": numerator_match["value"],
            "numerator_count": numerator_count,
            "denominator_count": denominator_count,
            "percentage": round(percentage, 4) if percentage is not None else None,
        }
        return ToolResult(
            success=True,
            tool_name=self.name,
            data=row,
            answer="Percentage calculation completed from table counts.",
            table=[row],
            confidence=0.92 if denominator_count else 0.55,
            warnings=[] if denominator_count else ["Percentage denominator is zero."],
            metadata={
                "operation": "percentage_share",
                "numerator_filters": numerator_filters,
                "denominator_filters": denominator_filters,
                "row_count": len(dataframe),
            },
        )

    def _execute_correlation_if_needed(
        self,
        query_plan: QueryPlan,
        dataframe: pd.DataFrame,
        profile: TableProfile,
        filters: List[Dict[str, Any]],
    ) -> Optional[ToolResult]:
        text = self._plan_text(query_plan)
        comparison_type = str((query_plan.comparison or {}).get("type") or (query_plan.comparison or {}).get("analysis_type") or "")
        if comparison_type != "correlation" and not any(
            signal in text for signal in {"associated", "correlation", "correlated", "relationship", "related"}
        ):
            return None
        metric_columns = self._mapped_metric_columns(query_plan, profile)
        if len(metric_columns) < 2:
            return None
        operation = {
            "operation": "correlation",
            "columns": metric_columns[:2],
            "filters": filters,
        }
        result = self.executor.execute(dataframe, operation)
        if result.success:
            result.tool_name = self.name
            result.answer = "Correlation analysis completed."
            result.metadata["operation"] = operation
            return result
        return None

    def _execute_median_segment_if_needed(
        self,
        query_plan: QueryPlan,
        dataframe: pd.DataFrame,
        profile: TableProfile,
        filters: List[Dict[str, Any]],
    ) -> Optional[ToolResult]:
        text = self._plan_text(query_plan)
        if "above the median" not in text and "above median" not in text and "premium" not in text:
            return None
        metric_columns = self._mapped_metric_columns(query_plan, profile)
        if not metric_columns:
            return None
        metric_column = metric_columns[0]
        grouping = self._mapped_grouping_columns(query_plan, profile)
        grouping = [column for column in grouping if column != metric_column]
        if not grouping:
            return None
        median_value = profile.numeric_stats.get(metric_column, {}).get("median")
        if median_value is None:
            numeric = pd.to_numeric(dataframe[metric_column], errors="coerce").dropna()
            if numeric.empty:
                return None
            median_value = float(numeric.median())
        operation = {
            "operation": "groupby",
            "group_by": grouping[:1],
            "aggregations": [{"column": None, "agg": "count", "alias": "count_rows"}],
            "filters": filters + [{"column": metric_column, "operator": "gt", "value": median_value}],
            "sort_by": "count_rows",
            "ascending": False,
        }
        result = self.executor.execute(dataframe, operation)
        if result.success:
            result.tool_name = self.name
            result.answer = "Median-segment distribution completed."
            result.metadata["operation"] = operation
            result.data = {
                "metric_column": metric_column,
                "median": median_value,
                "segment": "above_median",
                "rows": result.table,
            }
            return result
        return None

    def _execute_entity_comparison_if_needed(
        self,
        query_plan: QueryPlan,
        dataframe: pd.DataFrame,
        profile: TableProfile,
        filters: List[Dict[str, Any]],
    ) -> Optional[ToolResult]:
        text = self._plan_text(query_plan)
        if not any(signal in text for signal in {"associated", "compare", "compared", "higher", "lower", "versus", "vs"}):
            return None
        metric_columns = self._mapped_metric_columns(query_plan, profile)
        if not metric_columns:
            return None
        matches = self._matched_entities(query_plan, dataframe, profile)
        matches_by_column: Dict[str, List[Any]] = {}
        for match in matches:
            matches_by_column.setdefault(match["column"], [])
            if match["value"] not in matches_by_column[match["column"]]:
                matches_by_column[match["column"]].append(match["value"])
        comparison_column = next((column for column, values in matches_by_column.items() if len(values) >= 2), None)
        if not comparison_column:
            return None
        metric_column = metric_columns[0]
        operation = {
            "operation": "groupby",
            "group_by": [comparison_column],
            "aggregations": [{"column": metric_column, "agg": "mean", "alias": "mean_{0}".format(metric_column)}],
            "filters": [{"column": comparison_column, "operator": "in", "value": matches_by_column[comparison_column]}],
            "sort_by": "mean_{0}".format(metric_column),
            "ascending": False,
        }
        result = self.executor.execute(dataframe, operation)
        if result.success:
            result.tool_name = self.name
            result.answer = "Entity comparison completed with deterministic grouped means."
            result.metadata["operation"] = operation
            return result
        return None

    def _mapped_metric_columns(self, query_plan: QueryPlan, profile: TableProfile) -> List[str]:
        columns: List[str] = []
        for metric in [_metric_name(metric) for metric in query_plan.metrics if _metric_name(metric)]:
            if metric == "count" or metric in TABLE_SUMMARY_METRICS:
                continue
            column = self._map_column(metric, profile)
            if column and column not in columns:
                columns.append(column)
        return columns

    def _matched_entities(self, query_plan: QueryPlan, dataframe: pd.DataFrame, profile: TableProfile) -> List[Dict[str, Any]]:
        text = self._plan_text(query_plan)
        matches: List[Dict[str, Any]] = []
        for entity in query_plan.entities:
            entity_value = self._entity_value(entity)
            if not entity_value:
                continue
            match = self._match_entity_value(entity_value, dataframe, profile, self._entity_field_hint(entity))
            if not match:
                continue
            column, value = match
            entity_text = str(entity.get("text") or entity_value) if isinstance(entity, dict) else str(entity_value)
            normalized = str(value).lower()
            position_candidates = [text.find(entity_text.lower()), text.find(normalized)]
            position = min([item for item in position_candidates if item >= 0], default=-1)
            matches.append({"column": column, "value": value, "text": entity_text, "position": position})
        return matches

    def _plan_text(self, query_plan: QueryPlan) -> str:
        return "{0} {1}".format(query_plan.original_query or "", query_plan.rewritten_query or "").lower()

    def _execute_multi_metric_if_needed(
        self,
        query_plan: QueryPlan,
        dataframe: pd.DataFrame,
        profile: TableProfile,
        filters: List[Dict[str, Any]],
    ) -> Optional[ToolResult]:
        metrics = [_metric_name(metric) for metric in query_plan.metrics if _metric_name(metric)]
        if len(metrics) <= 1 or query_plan.grouping or query_plan.limit:
            return None

        aggregations = [_aggregation_name(item) for item in query_plan.aggregations]
        aggregation = aggregations[0] if aggregations else "sum"
        row: Dict[str, Any] = {}
        metadata_results = []
        warnings: List[str] = []

        for metric in metrics:
            column = self._map_column(metric, profile)
            if not column:
                warnings.append("Could not map metric: {0}".format(metric))
                continue
            operation = {
                "operation": "aggregate",
                "column": column,
                "agg": aggregation,
                "filters": filters,
                "alias": "{0}_{1}".format(aggregation, column),
            }
            result = self.executor.execute(dataframe, operation)
            metadata_results.append({"metric": metric, "operation": operation, "success": result.success})
            if result.success:
                row[operation["alias"]] = result.data.get("value")
            else:
                warnings.append(result.error_msg or "Metric aggregation failed: {0}".format(metric))

        if not row:
            return None

        return ToolResult(
            success=True,
            tool_name=self.name,
            data={"metrics": row, "row_count": 1},
            answer="Table analysis completed for multiple metrics.",
            table=[row],
            confidence=0.9,
            warnings=warnings,
            metadata={"multi_metric": True, "metric_results": metadata_results},
        )

    def _map_column(self, term: str, profile: TableProfile) -> Optional[str]:
        if not term:
            return None
        match = self.mapper.match_column(term, profile)
        return match.matched_column

    def _answer_for_result(self, query_plan: QueryPlan, pandas_result: ToolResult, operation: Dict[str, Any]) -> str:
        if operation.get("operation") == "groupby":
            return "Table analysis completed with grouped results."
        if operation.get("operation") in {"top_k", "bottom_k"}:
            return "Table analysis completed with ranked results."
        return pandas_result.answer or "Table analysis completed."

    def _failed(self, message: str) -> ToolResult:
        return ToolResult(
            success=False,
            tool_name=self.name,
            data={},
            answer=None,
            table=[],
            confidence=0.0,
            warnings=[],
            error_msg=message,
            metadata={"tool_name": self.name},
        )
