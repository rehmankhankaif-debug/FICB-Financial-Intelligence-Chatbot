from __future__ import annotations

from src.agents.planning_validator import PlanningValidator
from src.models.query import QueryPlan
from src.models.source import SourceSelection
from src.models.validation import ValidationResult


def test_valid_table_plan_passes() -> None:
    plan = QueryPlan(
        intent="table_analysis",
        required_source_type="table",
        metrics=[{"name": "profit"}],
        aggregations=[{"operation": "mean"}],
        confidence=0.88,
    )
    source = SourceSelection(selected_source_id="sales", source_type="table", confidence=0.9)

    validation = PlanningValidator().validate_plan(plan, source)

    assert isinstance(validation, ValidationResult)
    assert validation.is_valid is True
    assert validation.issues == []


def test_invalid_intent_fails() -> None:
    plan = QueryPlan(intent="bad_intent", confidence=0.9)

    validation = PlanningValidator().validate_plan(plan)

    assert validation.is_valid is False
    assert "Invalid intent" in validation.issues[0]


def test_low_confidence_triggers_clarification() -> None:
    plan = QueryPlan(intent="general_finance", confidence=0.2)

    validation = PlanningValidator().validate_plan(plan)

    assert validation.clarification_needed is True
    assert validation.clarification_question
    assert validation.warnings


def test_chart_request_missing_chart_type_warns() -> None:
    plan = QueryPlan(
        intent="chart_request",
        required_source_type="table",
        metrics=[{"name": "count"}],
        chart_requested=True,
        chart_type=None,
        confidence=0.82,
    )
    source = SourceSelection(selected_source_id="cars", source_type="table", confidence=0.85)

    validation = PlanningValidator().validate_plan(plan, source)

    assert validation.is_valid is True
    assert any("chart_type" in warning for warning in validation.warnings)


def test_table_plan_without_analysis_fields_fails() -> None:
    plan = QueryPlan(intent="table_analysis", required_source_type="table", confidence=0.82)
    source = SourceSelection(selected_source_id="sales", source_type="table", confidence=0.9)

    validation = PlanningValidator().validate_plan(plan, source)

    assert validation.is_valid is False
    assert any("Table query needs" in issue for issue in validation.issues)


def test_document_plan_with_table_source_fails() -> None:
    plan = QueryPlan(intent="summarize_document", required_source_type="document", confidence=0.85)
    source = SourceSelection(selected_source_id="sales", source_type="table", confidence=0.9)

    validation = PlanningValidator().validate_plan(plan, source)

    assert validation.is_valid is False
    assert any("Document" in issue for issue in validation.issues)
