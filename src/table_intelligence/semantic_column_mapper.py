from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Tuple

from src.models.table import ColumnMatch, TableProfile
from src.table_intelligence.profiler import normalize_column_name

try:
    from rapidfuzz import fuzz
except Exception:
    fuzz = None


SYNONYM_REGISTRY: Dict[str, List[str]] = {
    "age": ["age", "respondent_age"],
    "avenue": ["avenue", "investment_avenue", "investment_avenues", "preferred_avenue", "preferred_investment"],
    "duration": ["duration", "investment_duration", "time_horizon", "horizon"],
    "equity_market": ["equity", "equity_market", "equity market"],
    "expected_return": ["expect", "expected", "expected_return", "return_expectation", "returns"],
    "fixed_deposits": ["fd", "fixed_deposit", "fixed_deposits", "fixed deposits", "reason_fd"],
    "gender": ["gender", "sex"],
    "gold": ["gold"],
    "government_bonds": ["bond", "bonds", "government_bond", "government_bonds", "reason_bonds"],
    "investment_monitor": ["invest_monitor", "investment_monitor", "monitor", "monitoring"],
    "mutual_funds": ["mutual", "mutual_fund", "mutual_funds", "reason_mutual"],
    "objective": ["objective", "objectives", "savings_objective", "savings_objectives"],
    "ppf": ["ppf", "public_provident_fund"],
    "purpose": ["purpose", "investment_purpose"],
    "source": ["source", "information_source", "financial_information_source"],
    "stock_market": ["stock", "stock_market", "stock market", "stock_marktet", "stocks"],
    "sales": ["sales", "revenue", "income", "earnings", "turnover", "net_sales"],
    "revenue": ["revenue", "sales", "income", "earnings", "turnover"],
    "profit": ["profit", "margin", "net_profit", "gain", "earnings"],
    "expense": ["expense", "expenses", "cost", "spend", "expenditure"],
    "price": ["price", "priced", "premium", "premium_price", "premium priced", "listing_price", "sale_price"],
    "mileage": ["mileage", "km", "km_driven", "kilometers", "kilometres", "odometer"],
    "fuel_type": ["fuel", "fuel_type", "fuel type", "fuel category", "fuel_category", "diesel", "petrol", "cng", "electric"],
    "brand": ["brand", "make", "manufacturer"],
    "product": ["product", "item", "sku"],
    "runs": ["runs", "batsman_runs", "score", "scored_runs", "run"],
    "strike_rate": ["strike_rate", "strike rate", "sr", "strikerate"],
    "batter": ["batter", "batsman", "player", "striker"],
    "winner": ["winner", "result", "status", "outcome", "winning_team"],
    "match": ["match", "match_id", "game", "fixture"],
    "transmission": ["transmission", "gear", "manual", "automatic"],
    "count": ["count", "total_count", "number", "quantity", "volume"],
    "date": ["date", "month", "year", "quarter", "period", "time"],
}


def _term_tokens(text: str) -> List[str]:
    return [token for token in re.split(r"[^a-zA-Z0-9]+", text.lower()) if token]


def _expanded_terms(term: str) -> List[str]:
    normalized = normalize_column_name(term)
    terms = {normalized, term.lower().strip(), normalized.replace("_", " ")}
    for key, aliases in SYNONYM_REGISTRY.items():
        normalized_aliases = {normalize_column_name(alias) for alias in aliases}
        if normalized in normalized_aliases or normalized == key:
            terms.update(normalized_aliases)
            terms.update(alias.lower() for alias in aliases)
    return sorted(terms)


def _fuzzy_score(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    if fuzz is not None:
        return float(fuzz.token_set_ratio(left, right)) / 100.0
    return SequenceMatcher(None, left, right).ratio()


class SemanticColumnMapper:
    def match_column(self, requested_term: str, table_profile: TableProfile) -> ColumnMatch:
        candidates = self._score_candidates(requested_term, table_profile)
        if not candidates:
            return ColumnMatch(
                requested_term=requested_term,
                matched_column=None,
                confidence=0.0,
                strategy="none",
                reason="No columns are available to match.",
                alternatives=[],
            )

        candidates = sorted(candidates, key=lambda item: item["confidence"], reverse=True)
        best = candidates[0]
        alternatives = candidates[1:6]

        if best["confidence"] < 0.55:
            return ColumnMatch(
                requested_term=requested_term,
                matched_column=None,
                confidence=best["confidence"],
                strategy="none",
                reason="No column met the minimum semantic confidence threshold.",
                alternatives=candidates[:5],
            )

        return ColumnMatch(
            requested_term=requested_term,
            matched_column=best["column"],
            confidence=best["confidence"],
            strategy=best["strategy"],
            reason=best["reason"],
            alternatives=alternatives,
        )

    def match_columns(self, requested_terms: List[str], table_profile: TableProfile) -> List[ColumnMatch]:
        return [self.match_column(term, table_profile) for term in requested_terms]

    def _score_candidates(self, requested_term: str, table_profile: TableProfile) -> List[Dict[str, Any]]:
        requested_norm = normalize_column_name(requested_term)
        requested_words = set(_term_tokens(requested_term))
        expanded_terms = _expanded_terms(requested_term)
        candidates: List[Dict[str, Any]] = []

        for column in table_profile.columns:
            normalized_column = table_profile.normalized_columns.get(column, normalize_column_name(column))
            score, strategy, reason = self._score_column(
                requested_term=requested_term,
                requested_norm=requested_norm,
                requested_words=requested_words,
                expanded_terms=expanded_terms,
                column=column,
                normalized_column=normalized_column,
                table_profile=table_profile,
            )
            candidates.append(
                {
                    "column": column,
                    "confidence": round(score, 4),
                    "strategy": strategy,
                    "reason": reason,
                }
            )
        return candidates

    def _score_column(
        self,
        requested_term: str,
        requested_norm: str,
        requested_words: set,
        expanded_terms: List[str],
        column: str,
        normalized_column: str,
        table_profile: TableProfile,
    ) -> Tuple[float, str, str]:
        if requested_norm == normalized_column or requested_term.strip().lower() == column.lower():
            return 1.0, "exact", "Requested term exactly matches the column."

        normalized_expanded_terms = {normalize_column_name(term) for term in expanded_terms}
        if normalized_column in normalized_expanded_terms:
            return 0.92, "synonym", "Requested term and column belong to the same synonym group."
        column_tokens = set(normalized_column.split("_"))
        synonym_token_matches = [
            term
            for term in normalized_expanded_terms
            if term in column_tokens or (len(term) >= 3 and term in normalized_column)
        ]
        if synonym_token_matches:
            return 0.86, "synonym", "Requested term matches a synonym token in the column name."

        column_words = set(_term_tokens(normalized_column.replace("_", " ")))
        overlap = requested_words.intersection(column_words)
        overlap_score = len(overlap) / float(max(1, len(requested_words))) if requested_words else 0.0

        fuzzy_score = max(
            _fuzzy_score(requested_norm, normalized_column),
            _fuzzy_score(requested_term.lower(), column.lower()),
        )

        sample_score = self._sample_value_score(requested_words, column, table_profile)

        score = max(
            fuzzy_score * 0.88,
            overlap_score * 0.82,
            sample_score,
        )

        if sample_score >= 0.75:
            return sample_score, "semantic", "Requested term matches representative values in this column."
        if fuzzy_score >= 0.72:
            return fuzzy_score * 0.88, "fuzzy", "Requested term fuzzily matches the column name."
        if overlap_score >= 0.5:
            return overlap_score * 0.82, "normalized", "Requested term shares normalized tokens with the column."

        return score, "fuzzy", "Weak semantic or fuzzy similarity."

    def _sample_value_score(self, requested_words: set, column: str, table_profile: TableProfile) -> float:
        if not requested_words:
            return 0.0
        values = table_profile.unique_values.get(column) or table_profile.sample_values.get(column) or []
        normalized_values = " ".join(str(value).lower() for value in values)
        matched_words = [word for word in requested_words if word in normalized_values]
        if not matched_words:
            return 0.0
        coverage = len(matched_words) / float(len(requested_words))
        return min(0.9, 0.55 + coverage * 0.35)
