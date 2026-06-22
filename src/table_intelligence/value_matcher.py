from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any, Dict, List

import pandas as pd

from src.models.table import ValueMatch

try:
    from rapidfuzz import fuzz
except Exception:
    fuzz = None


VALUE_ALIASES: Dict[str, List[str]] = {
    "rcb": ["royal challengers bangalore", "royal challenger bangalore", "bangalore"],
    "mi": ["mumbai indians"],
    "csk": ["chennai super kings"],
    "kkr": ["kolkata knight riders"],
    "srh": ["sunrisers hyderabad"],
    "dc": ["delhi capitals", "delhi daredevils"],
    "pbks": ["punjab kings", "kings xi punjab"],
}


def _normalize_value(value: Any) -> str:
    return str(value).strip().lower()


def _fuzzy_score(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    if fuzz is not None:
        return float(fuzz.token_set_ratio(left, right)) / 100.0
    return SequenceMatcher(None, left, right).ratio()


class ValueMatcher:
    def match_value(self, requested_value: Any, matched_column: str, dataframe: pd.DataFrame) -> ValueMatch:
        if dataframe is None or matched_column not in dataframe.columns:
            return ValueMatch(
                requested_value=str(requested_value),
                matched_column=matched_column,
                matched_value=None,
                confidence=0.0,
                strategy="none",
                reason="Column is not available for value matching.",
            )

        unique_values = list(dataframe[matched_column].dropna().unique())
        if not unique_values:
            return ValueMatch(
                requested_value=str(requested_value),
                matched_column=matched_column,
                matched_value=None,
                confidence=0.0,
                strategy="none",
                reason="Column has no non-empty values.",
            )

        requested_text = _normalize_value(requested_value)
        alias_terms = VALUE_ALIASES.get(requested_text, [])
        candidates: List[Dict[str, Any]] = []

        for value in unique_values:
            value_text = _normalize_value(value)
            score, strategy, reason = self._score_value(requested_text, alias_terms, value_text)
            candidates.append(
                {
                    "value": value,
                    "confidence": round(score, 4),
                    "strategy": strategy,
                    "reason": reason,
                }
            )

        candidates = sorted(candidates, key=lambda item: item["confidence"], reverse=True)
        best = candidates[0]
        alternatives = candidates[1:6]

        if best["confidence"] < 0.6:
            return ValueMatch(
                requested_value=str(requested_value),
                matched_column=matched_column,
                matched_value=None,
                confidence=best["confidence"],
                strategy="none",
                reason="No value met the minimum matching confidence threshold.",
                alternatives=candidates[:5],
            )

        if alternatives and abs(best["confidence"] - alternatives[0]["confidence"]) <= 0.03 and best["confidence"] < 0.99:
            return ValueMatch(
                requested_value=str(requested_value),
                matched_column=matched_column,
                matched_value=None,
                confidence=best["confidence"],
                strategy="ambiguous",
                reason="Multiple values matched too closely to choose safely.",
                alternatives=candidates[:5],
            )

        return ValueMatch(
            requested_value=str(requested_value),
            matched_column=matched_column,
            matched_value=best["value"],
            confidence=best["confidence"],
            strategy=best["strategy"],
            reason=best["reason"],
            alternatives=alternatives,
        )

    def _score_value(self, requested_text: str, alias_terms: List[str], value_text: str):
        if requested_text == value_text:
            return 1.0, "exact", "Requested value exactly matches a dataframe value."

        if value_text in alias_terms or any(alias == value_text for alias in alias_terms):
            return 0.96, "alias", "Requested value maps to a known alias for this dataframe value."

        if requested_text and requested_text in value_text:
            return 0.88, "contains", "Requested value is contained in a dataframe value."

        if alias_terms and any(alias in value_text or value_text in alias for alias in alias_terms):
            return 0.86, "alias", "Requested value partially matches a known alias."

        fuzzy_score = _fuzzy_score(requested_text, value_text)
        if fuzzy_score >= 0.8:
            return fuzzy_score, "fuzzy", "Requested value fuzzily matches a dataframe value."

        return fuzzy_score, "fuzzy", "Weak fuzzy match."
