from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from src.models.citation import Citation


FINANCIAL_METRIC_ALIASES: Dict[str, Sequence[str]] = {
    "expenses": ("total expenses", "expenses", "expense", "costs", "cost"),
    "revenue": ("total revenue", "revenue", "sales", "turnover"),
    "profit": ("net profit", "gross profit", "operating profit", "profit", "earnings"),
    "income": ("net income", "operating income", "income"),
    "assets": ("total assets", "assets"),
    "liabilities": ("total liabilities", "liabilities", "debt"),
    "cash flow": ("cash flow", "cashflow"),
    "margin": ("profit margin", "operating margin", "gross margin", "margin"),
}

_CURRENCY_CODES = {"USD", "EUR", "GBP", "INR", "JPY", "CAD", "AUD"}
_CURRENCY_SYMBOLS = {"$": "USD", "€": "EUR", "£": "GBP", "₹": "INR", "¥": "JPY"}
_SCALE_FACTORS = {
    "thousand": 1_000.0,
    "thousands": 1_000.0,
    "k": 1_000.0,
    "million": 1_000_000.0,
    "millions": 1_000_000.0,
    "mn": 1_000_000.0,
    "m": 1_000_000.0,
    "billion": 1_000_000_000.0,
    "billions": 1_000_000_000.0,
    "bn": 1_000_000_000.0,
    "b": 1_000_000_000.0,
    "crore": 10_000_000.0,
    "crores": 10_000_000.0,
    "cr": 10_000_000.0,
    "lakh": 100_000.0,
    "lakhs": 100_000.0,
}
_SCALE_LABELS = {
    1_000.0: "thousand",
    100_000.0: "lakh",
    1_000_000.0: "million",
    10_000_000.0: "crore",
    1_000_000_000.0: "billion",
}
_NUMBER_RE = re.compile(
    r"(?P<prefix>USD|EUR|GBP|INR|JPY|CAD|AUD|[$€£₹¥])?\s*"
    r"(?P<number>[-+]?\d+(?:,\d{3})*(?:\.\d+)?)\s*"
    r"(?P<scale>thousands?|millions?|billions?|crores?|lakhs?|mn|bn|cr|[kmb])?"
    r"(?P<percent>%)?",
    re.IGNORECASE,
)
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")


@dataclass(frozen=True)
class FinancialEvidence:
    source_id: str
    filename: str
    metric: str
    value: float
    normalized_value: float
    currency: str
    scale: float
    scale_label: str
    period: str
    page: Optional[int]
    chunk_id: str
    snippet: str
    score: float

    def table_row(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "filename": self.filename,
            "metric": self.metric,
            "period": self.period,
            "value": self.value,
            "currency": self.currency or None,
            "scale": self.scale_label or None,
            "normalized_value": self.normalized_value,
            "page": self.page,
            "chunk_id": self.chunk_id or None,
        }

    def citation(self) -> Citation:
        return Citation(
            source_id=self.source_id,
            filename=self.filename,
            page=self.page,
            chunk_id=self.chunk_id,
            text_snippet=self.snippet,
        )


def compare_document_financial_metrics(
    chunks: Iterable[Dict[str, Any]],
    query_plan: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    query_plan = dict(query_plan or {})
    query = " ".join(
        str(query_plan.get(key) or "")
        for key in ("original_query", "rewritten_query")
    ).strip()
    metric, aliases = _requested_metric(query, query_plan.get("metrics") or [])
    if not metric:
        return None

    evidence = _extract_evidence(list(chunks or []), metric, aliases, query)
    best_by_source = _best_per_source(evidence)
    if len(best_by_source) < 2:
        return None

    ordered = _order_evidence(best_by_source, query)
    left, right = ordered[0], ordered[1]
    compatible, compatibility_warning = _compatible_units(left, right)
    difference = left.normalized_value - right.normalized_value if compatible else None
    percentage_change = None
    if compatible and not math.isclose(right.normalized_value, 0.0, abs_tol=1e-12):
        percentage_change = (difference / abs(right.normalized_value)) * 100.0

    answer = _comparison_answer(left, right, difference, percentage_change, compatible)
    warnings = [compatibility_warning] if compatibility_warning else []
    return {
        "answer": answer,
        "table": [item.table_row() for item in ordered],
        "citations": [item.citation() for item in ordered],
        "warnings": warnings,
        "data": {
            "comparison_type": "numeric_financial",
            "metric": metric,
            "left_source_id": left.source_id,
            "right_source_id": right.source_id,
            "left_value": left.normalized_value,
            "right_value": right.normalized_value,
            "absolute_difference": difference,
            "percentage_change": percentage_change,
            "direction": _direction(difference),
            "unit_compatible": compatible,
            "source_count": len(ordered),
        },
    }


def _requested_metric(query: str, plan_metrics: Iterable[Any]) -> Tuple[str, Sequence[str]]:
    lowered = str(query or "").lower()
    candidates: List[str] = []
    for item in plan_metrics or []:
        if isinstance(item, dict):
            candidates.extend(str(item.get(key) or "") for key in ("name", "metric", "text"))
        else:
            candidates.append(str(item))
    search_text = " ".join(candidates + [lowered]).lower()
    for canonical, aliases in FINANCIAL_METRIC_ALIASES.items():
        if any(_phrase_present(alias, search_text) for alias in aliases):
            return canonical, aliases
    return "", ()


def _extract_evidence(
    chunks: List[Dict[str, Any]],
    metric: str,
    aliases: Sequence[str],
    query: str,
) -> List[FinancialEvidence]:
    evidence: List[FinancialEvidence] = []
    query_requests_percentage = any(term in query.lower() for term in ("percent", "percentage", "margin", "rate"))
    for chunk in chunks:
        content = str(chunk.get("content") or "")
        for sentence in _sentences(content):
            lowered = sentence.lower()
            matched_aliases = [alias for alias in aliases if _phrase_present(alias, lowered)]
            if not matched_aliases:
                continue
            alias = max(matched_aliases, key=len)
            alias_position = lowered.find(alias.lower())
            for match in _NUMBER_RE.finditer(sentence):
                parsed = _parse_number(match, sentence, query_requests_percentage)
                if parsed is None:
                    continue
                value, normalized_value, currency, scale, scale_label = parsed
                period = _period(sentence, str(chunk.get("filename") or ""))
                distance = abs(match.start() - max(0, alias_position))
                score = 10.0 - min(6.0, distance / 40.0)
                score += 2.0 if "total" in lowered else 0.0
                score += 1.5 if currency else 0.0
                score += 1.0 if scale > 1.0 else 0.0
                score += 1.0 if period else 0.0
                evidence.append(
                    FinancialEvidence(
                        source_id=str(chunk.get("source_id") or ""),
                        filename=str(chunk.get("filename") or chunk.get("source_id") or "source"),
                        metric=metric,
                        value=value,
                        normalized_value=normalized_value,
                        currency=currency,
                        scale=scale,
                        scale_label=scale_label,
                        period=period,
                        page=_optional_int(chunk.get("page")),
                        chunk_id=str(chunk.get("chunk_id") or ""),
                        snippet=" ".join(sentence.split())[:300],
                        score=score,
                    )
                )
    return evidence


def _parse_number(
    match: re.Match[str],
    sentence: str,
    query_requests_percentage: bool,
) -> Optional[Tuple[float, float, str, float, str]]:
    raw_number = str(match.group("number") or "").replace(",", "")
    try:
        value = float(raw_number)
    except ValueError:
        return None
    prefix = str(match.group("prefix") or "").upper()
    scale_token = str(match.group("scale") or "").lower()
    is_percent = bool(match.group("percent"))
    if is_percent and not query_requests_percentage:
        return None
    if not prefix and not scale_token and not is_percent and value.is_integer() and 1900 <= value <= 2100:
        return None
    currency = _CURRENCY_SYMBOLS.get(prefix, prefix if prefix in _CURRENCY_CODES else "")
    scale = _SCALE_FACTORS.get(scale_token, 1.0)
    scale_label = _SCALE_LABELS.get(scale, "")
    if is_percent:
        scale_label = "percent"
    return value, value * scale, currency, scale, scale_label


def _best_per_source(evidence: Iterable[FinancialEvidence]) -> List[FinancialEvidence]:
    selected: Dict[str, FinancialEvidence] = {}
    for item in evidence:
        if not item.source_id:
            continue
        current = selected.get(item.source_id)
        if current is None or item.score > current.score:
            selected[item.source_id] = item
    return list(selected.values())


def _order_evidence(evidence: List[FinancialEvidence], query: str) -> List[FinancialEvidence]:
    requested_periods = _YEAR_RE.findall(query or "")
    period_order = {period: index for index, period in enumerate(requested_periods)}
    original_order = {item.source_id: index for index, item in enumerate(evidence)}
    return sorted(
        evidence,
        key=lambda item: (
            period_order.get(item.period, len(period_order)),
            original_order.get(item.source_id, 999),
        ),
    )


def _compatible_units(left: FinancialEvidence, right: FinancialEvidence) -> Tuple[bool, str]:
    if left.currency and right.currency and left.currency != right.currency:
        return False, "The document values use different currencies; no arithmetic difference was calculated."
    if left.scale_label == "percent" or right.scale_label == "percent":
        if left.scale_label != right.scale_label:
            return False, "A percentage value cannot be directly compared with an absolute financial value."
    return True, ""


def _comparison_answer(
    left: FinancialEvidence,
    right: FinancialEvidence,
    difference: Optional[float],
    percentage_change: Optional[float],
    compatible: bool,
) -> str:
    metric_label = left.metric.replace("_", " ").title()
    left_label = left.period or left.filename
    right_label = right.period or right.filename
    left_value = _display_value(left.normalized_value, left.currency, left.scale)
    right_value = _display_value(right.normalized_value, right.currency, right.scale)
    if not compatible or difference is None:
        return "{0}: {1} reports {2}; {3} reports {4}.".format(
            metric_label, left_label, left_value, right_label, right_value
        )
    display_scale = left.scale if math.isclose(left.scale, right.scale) else 1.0
    currency = left.currency or right.currency
    difference_label = _display_value(difference, currency, display_scale)
    direction = _direction(difference)
    percent_text = ""
    if percentage_change is not None:
        percent_text = ", a {0:.2f}% {1}".format(abs(percentage_change), direction)
    return (
        "{0}: {1} reports {2}; {3} reports {4}. "
        "Difference ({1} - {3}) is {5}{6}."
    ).format(metric_label, left_label, left_value, right_label, right_value, difference_label, percent_text)


def _display_value(normalized_value: float, currency: str, preferred_scale: float) -> str:
    scale = preferred_scale if preferred_scale > 0 else 1.0
    value = normalized_value / scale
    number = "{0:,.2f}".format(value).rstrip("0").rstrip(".")
    scale_label = _SCALE_LABELS.get(scale, "")
    parts = [part for part in (currency, number, scale_label) if part]
    return " ".join(parts)


def _direction(difference: Optional[float]) -> str:
    if difference is None:
        return "not_calculated"
    if math.isclose(difference, 0.0, abs_tol=1e-12):
        return "no change"
    return "increase" if difference > 0 else "decrease"


def _period(sentence: str, filename: str) -> str:
    match = _YEAR_RE.search(sentence or "") or _YEAR_RE.search(filename or "")
    return match.group(0) if match else ""


def _sentences(content: str) -> List[str]:
    return [item.strip() for item in re.split(r"(?<=[.!?])\s+|[\r\n]+", content or "") if item.strip()]


def _phrase_present(phrase: str, text: str) -> bool:
    return bool(re.search(r"(?<![A-Za-z0-9]){0}(?![A-Za-z0-9])".format(re.escape(phrase.lower())), text.lower()))


def _optional_int(value: Any) -> Optional[int]:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None
