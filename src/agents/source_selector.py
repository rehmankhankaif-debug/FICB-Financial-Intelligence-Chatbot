from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from src.agents.confidence import normalize_confidence
from src.models.query import QueryPlan
from src.models.source import SourceSelection
from src.models.table import TableProfile


TABLE_FILE_TYPES = {"csv", "xlsx", "xls", "table"}
DOCUMENT_FILE_TYPES = {"pdf", "docx", "txt", "html", "url", "document"}
TABLE_INTENTS = {"table_analysis", "chart_request"}
DOCUMENT_INTENTS = {"summarize_document", "rag_question", "url_lookup"}


def _payload(item: Any) -> Dict[str, Any]:
    if item is None:
        return {}
    if isinstance(item, dict):
        return dict(item)
    if hasattr(item, "model_dump"):
        return item.model_dump()
    if hasattr(item, "dict"):
        return item.dict()
    return {"value": str(item)}


def _source_id(source: Dict[str, Any]) -> str:
    return str(source.get("source_id") or source.get("id") or source.get("path") or source.get("filename") or "")


def _source_file_type(source: Dict[str, Any]) -> str:
    value = source.get("file_type") or source.get("source_type") or ""
    metadata = source.get("metadata") if isinstance(source.get("metadata"), dict) else {}
    if not value:
        value = metadata.get("file_type") or metadata.get("source_type") or metadata.get("source_category") or ""
    if not value and source.get("filename"):
        filename = str(source.get("filename"))
        if "." in filename:
            value = filename.rsplit(".", 1)[-1]
    return str(value).lower()


def _category(file_type: str) -> str:
    if file_type in TABLE_FILE_TYPES:
        return "table"
    if file_type in DOCUMENT_FILE_TYPES:
        return "document"
    return file_type or "unknown"


def _tokens_from_text(text: Any) -> set:
    return {token for token in re.split(r"[^a-zA-Z0-9_]+", str(text).lower()) if token}


class SourceSelector:
    def select_source(
        self,
        query_plan: QueryPlan,
        available_sources: List[Any],
        table_profiles: Optional[List[TableProfile]] = None,
        document_metadata: Optional[List[Any]] = None,
    ) -> SourceSelection:
        try:
            sources = [_payload(source) for source in (available_sources or [])]
            if not sources:
                return SourceSelection(
                    selected_source_id=None,
                    source_type="",
                    confidence=0.0,
                    reason="No uploaded sources are available.",
                    alternatives=[],
                )

            table_profile_map = {
                profile.source_id: profile for profile in (table_profiles or []) if getattr(profile, "source_id", None)
            }
            document_metadata_map = {
                str((_payload(item).get("source_id") or _payload(item).get("id") or "")): _payload(item)
                for item in (document_metadata or [])
            }

            scored = [
                self._score_source(query_plan, source, table_profile_map, document_metadata_map)
                for source in sources
            ]
            scored = sorted(scored, key=lambda item: item["confidence"], reverse=True)
            best = scored[0]

            if self._required_category(query_plan) == "mixed":
                comparable = [
                    item for item in scored
                    if item["source_type"] in {"table", "document"} and item["source_id"]
                ]
                if len(comparable) >= 2:
                    source_types = {item["source_id"]: item["source_type"] for item in comparable}
                    return SourceSelection(
                        selected_source_id=comparable[0]["source_id"],
                        selected_source_ids=[item["source_id"] for item in comparable],
                        selected_source_types=source_types,
                        source_type="mixed",
                        confidence=min(1.0, sum(item["confidence"] for item in comparable) / len(comparable)),
                        reason="Selected {0} comparable uploaded sources across: {1}.".format(
                            len(comparable), ", ".join(sorted(set(source_types.values())))
                        ),
                        alternatives=[],
                    )

            if best["confidence"] < 0.55:
                required_type = self._required_category(query_plan)
                matching_required = [item for item in scored if item["source_type"] == required_type]
                if required_type in {"table", "document"} and len(matching_required) == 1:
                    selected = dict(matching_required[0])
                    selected["confidence"] = max(selected["confidence"], 0.58)
                    selected["reason"] = "{0}; selected as the only uploaded {1} source for this query".format(
                        selected["reason"],
                        required_type,
                    )
                    return SourceSelection(
                        selected_source_id=selected["source_id"],
                        source_type=selected["source_type"],
                        confidence=selected["confidence"],
                        reason=selected["reason"],
                        alternatives=[item for item in scored if item["source_id"] != selected["source_id"]],
                    )
                return SourceSelection(
                    selected_source_id=None,
                    source_type=best["source_type"],
                    confidence=best["confidence"],
                    reason="No source met the relevance threshold. {0}".format(best["reason"]),
                    alternatives=scored,
                )

            return SourceSelection(
                selected_source_id=best["source_id"],
                selected_source_ids=[best["source_id"]],
                selected_source_types={best["source_id"]: best["source_type"]},
                source_type=best["source_type"],
                confidence=best["confidence"],
                reason=best["reason"],
                alternatives=scored[1:],
            )
        except Exception as exc:
            return SourceSelection(
                selected_source_id=None,
                source_type="",
                confidence=0.0,
                reason="Source selection failed safely: {0}".format(str(exc)),
                alternatives=[],
            )

    def _score_source(
        self,
        query_plan: QueryPlan,
        source: Dict[str, Any],
        table_profile_map: Dict[str, TableProfile],
        document_metadata_map: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        source_id = _source_id(source)
        file_type = _source_file_type(source)
        source_type = _category(file_type)
        metadata = source.get("metadata") if isinstance(source.get("metadata"), dict) else {}
        if source_type == "unknown" and str(metadata.get("source_category") or "").lower() in {"table", "document"}:
            source_type = str(metadata.get("source_category")).lower()
        required_type = self._required_category(query_plan)

        type_score = 0.0
        if required_type == "mixed":
            type_score = 0.55 if source_type in {"table", "document"} else 0.0
        elif required_type is None:
            type_score = 0.35
        elif source_type == required_type:
            type_score = 0.60

        query_tokens = self._query_tokens(query_plan)
        filename_score = self._filename_score(query_tokens, source)
        metadata_score = 0.0

        if source_type == "table":
            metadata_score = self._table_profile_score(query_tokens, table_profile_map.get(source_id))
        elif source_type == "document":
            metadata_score = self._document_metadata_score(query_tokens, document_metadata_map.get(source_id), source)

        confidence = normalize_confidence(type_score + metadata_score + filename_score)
        reason = "source_type={0}, required_type={1}, metadata_match={2:.2f}, filename_match={3:.2f}".format(
            source_type,
            required_type or "any",
            metadata_score,
            filename_score,
        )
        return {
            "source_id": source_id,
            "filename": source.get("filename", ""),
            "file_type": file_type,
            "source_type": source_type,
            "confidence": confidence,
            "reason": reason,
        }

    def _required_category(self, query_plan: QueryPlan) -> Optional[str]:
        if query_plan.required_source_type in {"table", "document", "mixed"}:
            return query_plan.required_source_type
        if query_plan.intent in TABLE_INTENTS:
            return "table"
        if query_plan.intent in DOCUMENT_INTENTS:
            return "document"
        if query_plan.intent == "compare_documents":
            return "mixed"
        return None

    def _query_tokens(self, query_plan: QueryPlan) -> set:
        pieces = [
            query_plan.original_query,
            query_plan.rewritten_query,
            query_plan.intent,
            str(query_plan.comparison),
            " ".join(query_plan.grouping),
        ]
        for item in query_plan.metrics + query_plan.entities + query_plan.filters + query_plan.aggregations:
            pieces.append(str(item))
        return _tokens_from_text(" ".join(pieces))

    def _filename_score(self, query_tokens: set, source: Dict[str, Any]) -> float:
        filename_tokens = _tokens_from_text(source.get("filename", ""))
        if not query_tokens or not filename_tokens:
            return 0.0
        overlap = query_tokens.intersection(filename_tokens)
        return min(0.12, 0.04 * len(overlap))

    def _table_profile_score(self, query_tokens: set, profile: Optional[TableProfile]) -> float:
        if profile is None:
            return 0.08
        profile_tokens = set()
        profile_tokens.update(_tokens_from_text(" ".join(profile.columns)))
        profile_tokens.update(_tokens_from_text(" ".join(profile.normalized_columns.values())))
        profile_tokens.update(_tokens_from_text(profile.semantic_summary))
        profile_tokens.update(_tokens_from_text(" ".join(profile.metric_candidate_columns)))
        profile_tokens.update(_tokens_from_text(" ".join(profile.entity_candidate_columns)))
        profile_tokens.update(_tokens_from_text(" ".join(profile.result_candidate_columns)))
        for values in profile.sample_values.values():
            profile_tokens.update(_tokens_from_text(" ".join(str(value) for value in values)))

        overlap = query_tokens.intersection(profile_tokens)
        return min(0.38, 0.08 * len(overlap))

    def _document_metadata_score(self, query_tokens: set, metadata: Optional[Dict[str, Any]], source: Dict[str, Any]) -> float:
        merged = {}
        merged.update(source)
        if metadata:
            merged.update(metadata)
        metadata_tokens = _tokens_from_text(" ".join(str(value) for value in merged.values()))
        overlap = query_tokens.intersection(metadata_tokens)
        return min(0.34, 0.06 * len(overlap)) if overlap else 0.08
