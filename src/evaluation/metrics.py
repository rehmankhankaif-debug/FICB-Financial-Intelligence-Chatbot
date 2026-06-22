from __future__ import annotations

import math
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence


def normalize_text(value: Any) -> str:
    text = str(value or "").lower().strip()
    text = re.sub(r"[^a-z0-9.%+-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def exact_match(expected: Any, actual: Any) -> float:
    return 1.0 if normalize_text(expected) == normalize_text(actual) else 0.0


def contains_all_score(text: str, expected_terms: Iterable[str]) -> float:
    terms = [normalize_text(term) for term in expected_terms or [] if normalize_text(term)]
    if not terms:
        return 1.0
    normalized = normalize_text(text)
    matched = [term for term in terms if term in normalized]
    return len(matched) / float(len(terms))


def sequence_match_score(expected: Sequence[Any], actual: Sequence[Any], ordered: bool = True) -> float:
    expected_values = [normalize_text(item) for item in expected or []]
    actual_values = [normalize_text(item) for item in actual or []]
    if not expected_values:
        return 1.0 if not actual_values else 0.0
    if ordered:
        return 1.0 if expected_values == actual_values else 0.0
    matched = len(set(expected_values).intersection(set(actual_values)))
    return matched / float(len(set(expected_values)))


def source_selection_score(expected_source_id: Optional[str], actual_source_id: Optional[str], expected_source_type: Optional[str] = None, actual_source_type: Optional[str] = None) -> float:
    if expected_source_id:
        return 1.0 if str(expected_source_id) == str(actual_source_id) else 0.0
    if expected_source_type:
        expected_types = {normalize_text(item) for item in str(expected_source_type).split("|") if item}
        return 1.0 if normalize_text(actual_source_type) in expected_types else 0.0
    return 1.0


def citation_presence_score(requires_citations: bool, citation_count: int) -> float:
    if not requires_citations:
        return 1.0
    return 1.0 if int(citation_count or 0) > 0 else 0.0


def numeric_tokens(text: Any) -> List[str]:
    tokens = re.findall(r"(?<![A-Za-z])[-+]?\d+(?:,\d{3})*(?:\.\d+)?%?", str(text or ""))
    return [token.replace(",", "") for token in tokens]


def hallucination_risk_score(answer: str, grounded_payload: Any) -> float:
    answer_numbers = set(numeric_tokens(answer))
    if not answer_numbers:
        return 1.0
    grounded_numbers = set(numeric_tokens(grounded_payload))
    ungrounded = answer_numbers.difference(grounded_numbers)
    if not ungrounded:
        return 1.0
    return max(0.0, 1.0 - (len(ungrounded) / float(len(answer_numbers))))


def table_value_score(expected: Any, actual: Any, tolerance: float = 1e-6) -> float:
    if expected is None:
        return 1.0
    if isinstance(expected, dict):
        return _dict_score(expected, actual, tolerance)
    if isinstance(expected, list):
        return _list_score(expected, actual, tolerance)
    return _scalar_score(expected, actual, tolerance)


def _dict_score(expected: Dict[str, Any], actual: Any, tolerance: float) -> float:
    actual_dict = actual if isinstance(actual, dict) else {}
    if not expected:
        return 1.0
    scores = [_scalar_score(value, actual_dict.get(key), tolerance) for key, value in expected.items()]
    return sum(scores) / float(len(scores))


def _list_score(expected: List[Any], actual: Any, tolerance: float) -> float:
    actual_list = actual if isinstance(actual, list) else []
    if not expected:
        return 1.0 if not actual_list else 0.0
    if len(expected) != len(actual_list):
        return 0.0
    scores = []
    for expected_item, actual_item in zip(expected, actual_list):
        if isinstance(expected_item, dict):
            scores.append(_dict_score(expected_item, actual_item, tolerance))
        else:
            scores.append(_scalar_score(expected_item, actual_item, tolerance))
    return sum(scores) / float(len(scores))


def _scalar_score(expected: Any, actual: Any, tolerance: float) -> float:
    expected_number = _to_float(expected)
    actual_number = _to_float(actual)
    if expected_number is not None and actual_number is not None:
        return 1.0 if math.isclose(expected_number, actual_number, rel_tol=tolerance, abs_tol=tolerance) else 0.0
    return exact_match(expected, actual)


def _to_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def pass_fail(score: float, threshold: float = 1.0) -> bool:
    return float(score or 0.0) >= threshold
