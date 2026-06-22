from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.evaluation.benchmark_generator import BENCHMARK_CATEGORIES, CsvBenchmarkGenerator
from src.evaluation.evaluator import EvaluationReport, Evaluator
from src.evaluation.metrics import citation_presence_score, hallucination_risk_score, sequence_match_score, table_value_score
from src.table_intelligence.profiler import TableProfiler


def test_metric_helpers_score_expected_values() -> None:
    assert sequence_match_score(["a", "b"], ["a", "b"]) == 1.0
    assert citation_presence_score(True, 1) == 1.0
    assert hallucination_risk_score("Profit is 200.", {"value": 200}) == 1.0
    assert hallucination_risk_score("Profit is 999.", {"value": 200}) == 0.0
    assert table_value_score([{"profit": 200.0}], [{"profit": 200.0}]) == 1.0


def test_evaluator_returns_report_for_builtin_benchmark_file() -> None:
    evaluator = Evaluator()
    report = evaluator.evaluate_files([Path("evaluation/benchmark_queries.json")])

    assert isinstance(report, EvaluationReport)
    assert report.total_cases == 6
    assert report.passed_cases >= 5
    assert report.accuracy_percentage >= 80.0
    assert "intent_accuracy" in report.metric_accuracy


def test_evaluator_scores_csv_answer_accuracy() -> None:
    evaluator = Evaluator()
    report = evaluator.evaluate_files([Path("evaluation/sample_csv_cases.json")])

    assert report.metric_accuracy["csv_answer_accuracy"] == 100.0
    assert report.passed_cases == report.total_cases


def test_evaluator_scores_rag_citation_presence() -> None:
    evaluator = Evaluator()
    report = evaluator.evaluate_files([Path("evaluation/sample_rag_cases.json")])

    assert report.metric_accuracy["rag_citation_presence"] == 100.0
    assert report.metric_accuracy["hallucination_safety"] == 100.0


def test_evaluator_handles_missing_source_safely() -> None:
    evaluator = Evaluator()
    cases = evaluator.load_cases([Path("evaluation/sample_tool_selection_cases.json")])
    missing_source_case = [case for case in cases if case["id"] == "tool_missing_source_safe"][0]

    result = evaluator.evaluate_case(missing_source_case)

    assert result.scores["error_handling"] == 1.0
    assert result.actual["tools"] == []
    assert result.passed is True


def test_report_markdown_contains_summary() -> None:
    evaluator = Evaluator()
    report = evaluator.evaluate_files([Path("evaluation/sample_tool_selection_cases.json")])
    markdown = report.to_markdown()

    assert "# Evaluation Report" in markdown
    assert "Total cases" in markdown
    assert "Metric Accuracy" in markdown


def test_csv_benchmark_generator_creates_difficult_category_coverage() -> None:
    dataframe = pd.DataFrame(
        {
            "fuel_type": ["Diesel", "Petrol", "Diesel", "CNG"],
            "transmission": ["Manual", "Automatic", "Automatic", "Manual"],
            "brand": ["A", "B", "A", "C"],
            "price": [100, 200, 300, 150],
            "overall_cost": [80, 180, 250, 120],
        }
    )
    profile = TableProfiler().profile(dataframe, source_id="cars", filename="cars.csv")

    cases = CsvBenchmarkGenerator().generate_for_dataframe(dataframe, profile)

    categories = {case["category"] for case in cases}
    assert set(BENCHMARK_CATEGORIES).issuperset(categories)
    assert {"aggregation", "grouping", "comparisons", "correlation_exploration", "chart_generation"}.issubset(categories)
    assert all(case["expected"]["source_id"] == "cars" for case in cases)
    assert any(case.get("csv_case") for case in cases)


def test_csv_benchmark_generator_questions_only_mode_omits_embedded_answer_data() -> None:
    dataframe = pd.DataFrame({"fuel_type": ["Diesel", "Petrol"], "price": [100, 200]})
    profile = TableProfiler().profile(dataframe, source_id="cars", filename="cars.csv")

    cases = CsvBenchmarkGenerator().generate_for_dataframe(dataframe, profile, include_answer_checks=False)

    assert cases
    assert all("csv_case" not in case for case in cases)


def test_generated_csv_benchmarks_score_semantic_plans() -> None:
    dataframe = pd.DataFrame(
        {
            "fuel_type": ["Diesel", "Petrol", "Diesel", "CNG"],
            "transmission": ["Manual", "Automatic", "Automatic", "Manual"],
            "brand": ["A", "B", "A", "C"],
            "price": [100, 200, 300, 150],
            "overall_cost": [80, 180, 250, 120],
        }
    )
    profile = TableProfiler().profile(dataframe, source_id="cars", filename="cars.csv")
    cases = CsvBenchmarkGenerator().generate_for_dataframe(dataframe, profile)

    report = Evaluator().evaluate_cases(cases)

    assert report.total_cases >= 10
    assert report.metric_accuracy["semantic_plan_accuracy"] >= 80.0
