from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from src.models.table import TableProfile
from src.table_intelligence.semantic_column_mapper import SemanticColumnMapper


COUNT_SIGNALS = {
    "count",
    "counts",
    "dominant",
    "dominates",
    "frequency",
    "frequencies",
    "frequent",
    "how many",
    "kitne",
    "kitni",
    "number",
    "quantity",
    "qty",
    "volume",
    "total",
}
MEAN_SIGNALS = {"average", "avg", "mean"}
MAX_SIGNALS = {"best", "highest", "max", "maximum", "most", "top", "zyada"}
MIN_SIGNALS = {"bottom", "least", "lowest", "min", "minimum", "weakest"}
SUM_SIGNALS = {"contribute", "contributes", "contribution", "sum", "total"}
CHART_SIGNALS = {"bar", "bar graph", "chart", "graph", "pie", "pie chart", "plot", "visual"}
CORRELATION_SIGNALS = {"associated", "correlation", "correlated", "relationship", "related"}
TABLE_SIGNALS = COUNT_SIGNALS | MEAN_SIGNALS | MAX_SIGNALS | MIN_SIGNALS | SUM_SIGNALS | {
    "across",
    "anomaly",
    "anomalies",
    "breakdown",
    "break down",
    "break that down",
    "business insights",
    "change",
    "changed",
    "compare",
    "data",
    "distribution",
    "executive summary",
    "hidden patterns",
    "insight",
    "insights",
    "pattern",
    "patterns",
    "performance",
    "preference",
    "relationship",
    "scene",
    "segment",
    "segmentation",
    "trend",
    "trends",
    "unusual",
}
STOPWORDS = {
    "above",
    "across",
    "and",
    "aur",
    "bar",
    "bhai",
    "btao",
    "batao",
    "chart",
    "data",
    "de",
    "dikhao",
    "draw",
    "for",
    "give",
    "graph",
    "hai",
    "hain",
    "in",
    "ka",
    "ke",
    "ki",
    "kya",
    "me",
    "of",
    "overall",
    "please",
    "show",
    "the",
    "type",
    "waale",
    "what",
    "which",
}


def _tokens(text: str) -> List[str]:
    return [token for token in re.split(r"[^a-zA-Z0-9_]+", text.lower()) if token]


def _signal_present(signal: str, text: str, token_set: set) -> bool:
    normalized = signal.lower().strip()
    if not normalized:
        return False
    if " " in normalized:
        return re.search(r"\b{0}\b".format(re.escape(normalized)), text) is not None
    return normalized in token_set


def _contains_signal(text: str, signals: set) -> bool:
    token_set = set(_tokens(text))
    return any(_signal_present(signal, text, token_set) for signal in signals)


def _phrase_in_text(phrase: str, normalized_text: str) -> bool:
    return " {0} ".format(phrase) in normalized_text


def _profile_payload(profile: Any) -> Optional[TableProfile]:
    if isinstance(profile, TableProfile):
        return profile
    if isinstance(profile, dict):
        try:
            return TableProfile(**profile)
        except Exception:
            return None
    return None


class SemanticMeaningExtractor:
    """Extract analytical meaning from messy language using table schema evidence."""

    def __init__(self, mapper: Optional[SemanticColumnMapper] = None) -> None:
        self.mapper = mapper or SemanticColumnMapper()

    def extract(
        self,
        query: str,
        rewritten_query: str = "",
        table_profiles: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        profiles = [profile for profile in (_profile_payload(item) for item in (table_profiles or [])) if profile is not None]
        text = "{0} {1}".format(query or "", rewritten_query or "").lower()
        if not profiles or not text.strip():
            return {}

        count_requested = _contains_signal(text, COUNT_SIGNALS)
        chart_requested = _contains_signal(text, CHART_SIGNALS)
        rank_requested = _contains_signal(text, MAX_SIGNALS | MIN_SIGNALS) or re.search(r"\btop\s+\d+\b", text) is not None
        relationship_requested = _contains_signal(text, CORRELATION_SIGNALS)
        if relationship_requested:
            count_requested = False
        distribution_requested = any(signal in text for signal in {"across", "break down", "break that down", "distribution", "preference", "trend"})
        aggregation = self._aggregation(text)
        if relationship_requested and aggregation == "count":
            aggregation = ""
        phrases = self._candidate_phrases(text)
        column_matches = self._column_matches(phrases, profiles)
        value_entities = self._value_entities(text, profiles)

        grouping = self._grouping_columns(column_matches, chart_requested or rank_requested or distribution_requested)
        metrics = self._metric_columns(column_matches, aggregation, grouping)

        if count_requested and not metrics:
            metrics = [{"name": "count", "confidence": 0.82, "source": "semantic_meaning"}]
            aggregation = "count"
        if distribution_requested and grouping and not metrics and not aggregation:
            aggregation = "count"
            metrics = [{"name": "count", "confidence": 0.8, "source": "semantic_meaning"}]
        if (rank_requested or _contains_signal(text, SUM_SIGNALS)) and grouping and metrics and aggregation != "mean":
            aggregation = "sum"
        aggregations = [{"operation": aggregation, "confidence": 0.82, "source": "semantic_meaning"}] if aggregation else []

        table_requested = bool(grouping or metrics or value_entities or aggregations) and (
            _contains_signal(text, TABLE_SIGNALS) or chart_requested
        )
        intent = "chart_request" if chart_requested and table_requested else ("table_analysis" if table_requested else None)
        chart_types = []
        for chart_type, signals in [
            ("bar", {"bar", "bar graph", "bar chart"}),
            ("pie", {"pie", "pie graph", "pie chart"}),
            ("line", {"line", "line graph", "line chart"}),
            ("scatter", {"scatter", "scatter plot"}),
            ("histogram", {"histogram"}),
        ]:
            if any(signal in text for signal in signals):
                chart_types.append(chart_type)
        if chart_requested and not chart_types:
            chart_types.append("bar")

        return {
            "intent": intent,
            "required_source_type": "table" if intent in {"table_analysis", "chart_request"} else None,
            "metrics": metrics,
            "aggregations": aggregations,
            "grouping": grouping,
            "entities": value_entities,
            "chart_requested": chart_requested,
            "chart_type": chart_types[0] if chart_types else None,
            "chart_types": chart_types,
            "sorting": self._sorting(text),
            "comparison": {"type": "correlation", "analysis_type": "correlation"} if relationship_requested and len(metrics) >= 2 else {},
            "confidence": 0.84 if table_requested else 0.0,
            "reasoning_short": "Semantic schema extraction from query phrases, columns, and sample values.",
        }

    def _aggregation(self, text: str) -> str:
        if _contains_signal(text, MEAN_SIGNALS):
            return "mean"
        if _contains_signal(text, MAX_SIGNALS):
            return "max"
        if _contains_signal(text, MIN_SIGNALS):
            return "min"
        if _contains_signal(text, SUM_SIGNALS):
            return "sum"
        if _contains_signal(text, COUNT_SIGNALS):
            return "count"
        return ""

    def _sorting(self, text: str) -> Dict[str, Any]:
        if _contains_signal(text, MIN_SIGNALS):
            return {"direction": "asc"}
        if _contains_signal(text, MAX_SIGNALS) or _contains_signal(text, COUNT_SIGNALS):
            return {"direction": "desc"}
        return {}

    def _candidate_phrases(self, text: str) -> List[str]:
        tokens = _tokens(text)
        phrases: List[str] = []
        for n in range(4, 0, -1):
            for index in range(0, max(0, len(tokens) - n + 1)):
                window = tokens[index : index + n]
                if not window:
                    continue
                if all(token in STOPWORDS or token in COUNT_SIGNALS for token in window):
                    continue
                phrase = " ".join(window)
                if phrase not in phrases:
                    phrases.append(phrase)
        return phrases

    def _column_matches(self, phrases: List[str], profiles: List[TableProfile]) -> List[Dict[str, Any]]:
        matches: List[Dict[str, Any]] = []
        seen = set()
        for profile in profiles:
            for phrase in phrases:
                match = self.mapper.match_column(phrase, profile)
                if not match.matched_column or match.confidence < 0.74:
                    continue
                if match.strategy == "semantic" and match.matched_column not in profile.unique_values:
                    continue
                key = (profile.source_id, match.matched_column, phrase)
                if key in seen:
                    continue
                seen.add(key)
                matches.append(
                    {
                        "source_id": profile.source_id,
                        "column": match.matched_column,
                        "term": phrase,
                        "confidence": match.confidence,
                        "is_numeric": match.matched_column in profile.numeric_columns,
                        "is_categorical": match.matched_column in profile.categorical_columns,
                        "is_boolean": match.matched_column in profile.boolean_columns,
                        "is_datetime": match.matched_column in profile.datetime_columns,
                    }
                )
        return sorted(matches, key=lambda item: (item["confidence"], len(item["term"])), reverse=True)

    def _grouping_columns(self, matches: List[Dict[str, Any]], prefer_categories: bool) -> List[str]:
        if not prefer_categories:
            return []
        grouping: List[str] = []
        for match in matches:
            if not ((match["is_categorical"] and not match["is_numeric"]) or match["is_boolean"] or match["is_datetime"]):
                continue
            if match["column"] not in grouping:
                grouping.append(match["column"])
            if len(grouping) >= 2:
                break
        return grouping

    def _metric_columns(self, matches: List[Dict[str, Any]], aggregation: str, grouping: List[str]) -> List[Dict[str, Any]]:
        metrics: List[Dict[str, Any]] = []
        seen = set()
        for match in matches:
            if match["column"] in grouping:
                continue
            if not match["is_numeric"]:
                continue
            if match["column"] in seen:
                continue
            seen.add(match["column"])
            metrics.append({"name": match["column"], "confidence": match["confidence"], "source": "semantic_meaning"})
            if len(metrics) >= 2:
                break
        if aggregation == "count" and not metrics:
            return [{"name": "count", "confidence": 0.82, "source": "semantic_meaning"}]
        return metrics

    def _value_entities(self, text: str, profiles: List[TableProfile]) -> List[Dict[str, Any]]:
        entities: List[Dict[str, Any]] = []
        seen = set()
        normalized_text = " {0} ".format(text.lower())
        query_tokens = set(_tokens(text))
        for profile in profiles:
            for column, values in {**profile.unique_values, **profile.sample_values}.items():
                for value in values:
                    value_text = str(value).strip()
                    if not value_text:
                        continue
                    value_tokens = _tokens(value_text)
                    if not value_tokens:
                        continue
                    if all(token.isdigit() for token in value_tokens):
                        continue
                    value_phrase = " ".join(value_tokens)
                    if not self._value_matches_query(value_phrase, value_tokens, normalized_text, query_tokens):
                        continue
                    key = (column, value_text.lower())
                    if key in seen:
                        continue
                    seen.add(key)
                    entities.append(
                        {
                            "text": value_text,
                            "normalized": value,
                            "type": "category_value",
                            "field": column,
                            "confidence": 0.84,
                            "source": "semantic_meaning",
                        }
                    )
        return entities

    def _value_matches_query(
        self,
        value_phrase: str,
        value_tokens: List[str],
        normalized_text: str,
        query_tokens: set,
    ) -> bool:
        if _phrase_in_text(value_phrase, normalized_text):
            return True
        if len(value_tokens) == 1:
            return value_tokens[0] in query_tokens
        if len(value_tokens) <= 3:
            return all(token in query_tokens for token in value_tokens if token)
        return False
