from __future__ import annotations

from typing import Any, Dict, List

from src.models.document import RetrievedChunk
from src.models.query import QueryPlan
from src.models.tool import ToolResult
from src.rag.citations import CitationBuilder
from src.rag.retriever import Retriever
from src.rag.validator import RetrievalValidator
from src.security.prompt_guard import PromptInjectionGuard
from src.tools.base import BaseTool
from src.utils.text_summary import build_extractive_answer, clean_document_text


DEFAULT_MINIMUM_RETRIEVAL_SCORE = 0.05


def _dump_model(model: Any) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    if hasattr(model, "dict"):
        return model.dict()
    return {}


class RagQATool(BaseTool):
    name = "rag_qa_tool"
    description = "Retrieve document evidence and produce a grounded extractive response."
    supported_intents = ["rag_question", "compare_documents"]
    supported_source_types = ["document", "pdf", "docx", "url", "txt", "html", "mixed"]
    input_types = ["QueryPlan", "Retriever", "RetrievedChunk"]
    output_types = ["retrieved_context", "citations", "metadata"]
    input_requirements = ["query_plan and retriever or retrieved_chunks"]
    capabilities = ["retrieve_context", "document_qa_context"]
    positive_examples = ["what risks are mentioned in the report?"]
    negative_examples = ["calculate average profit"]
    can_chain_before = ["compare_tool"]
    confidence = 0.85

    def run(self, input_payload: Dict[str, Any]) -> ToolResult:
        try:
            query_plan = QueryPlan(**(input_payload.get("query_plan") or {}))
            chunks = self._get_chunks(input_payload or {}, query_plan)
            if not chunks:
                return ToolResult(
                    success=True,
                    tool_name=self.name,
                    data={"retrieved_chunks": [], "answer_found": False},
                    answer="Information not found in the retrieved document context.",
                    table=[],
                    citations=[],
                    confidence=0.0,
                    warnings=["No relevant document chunks were retrieved."],
                    metadata={"retrieval_empty": True},
                )

            document_risk = PromptInjectionGuard().assess_document_text("\n".join(chunk.content for chunk in chunks[:5]))
            security_warnings = [document_risk.warning()] if document_risk.is_suspicious else []
            validator = RetrievalValidator()
            chunks = validator.deduplicate(chunks)
            minimum_score = float((input_payload or {}).get("minimum_score", DEFAULT_MINIMUM_RETRIEVAL_SCORE))
            validation = validator.validate(chunks, minimum_score=minimum_score)
            if not validation.is_valid:
                return ToolResult(
                    success=True,
                    tool_name=self.name,
                    data={"retrieved_chunks": [self._chunk_payload(chunk) for chunk in chunks], "answer_found": False},
                    answer="I could not find enough reliable evidence in the retrieved document context.",
                    table=[],
                    citations=[],
                    confidence=validation.confidence,
                    warnings=validation.warnings + validation.issues + security_warnings,
                    metadata={
                        "retrieved_count": len(chunks),
                        "validation": _dump_model(validation),
                        "minimum_score": minimum_score,
                        "prompt_injection_risk": _dump_model(document_risk),
                    },
                )
            citations = CitationBuilder().build(chunks)
            answer = self._extractive_answer(chunks)
            return ToolResult(
                success=True,
                tool_name=self.name,
                data={"retrieved_chunks": [self._chunk_payload(chunk) for chunk in chunks], "answer_found": True},
                answer=answer,
                table=[],
                citations=citations,
                confidence=validation.confidence or self.confidence,
                warnings=validation.warnings + security_warnings,
                metadata={
                    "retrieved_count": len(chunks),
                    "validation": _dump_model(validation),
                    "prompt_injection_risk": _dump_model(document_risk),
                },
            )
        except Exception as exc:
            return ToolResult(success=False, tool_name=self.name, confidence=0.0, error_msg="RAG QA failed safely: {0}".format(str(exc)), metadata={})

    def _get_chunks(self, payload: Dict[str, Any], query_plan: QueryPlan) -> List[RetrievedChunk]:
        raw_chunks = payload.get("retrieved_chunks")
        if raw_chunks:
            return [chunk if isinstance(chunk, RetrievedChunk) else RetrievedChunk(**chunk) for chunk in raw_chunks]
        retriever = payload.get("retriever")
        if isinstance(retriever, Retriever):
            query = query_plan.rewritten_query or query_plan.original_query
            top_k = int(payload.get("top_k", 5))
            source_ids = [str(item) for item in (payload.get("source_ids") or []) if item]
            if source_ids:
                balanced: List[RetrievedChunk] = []
                per_source = max(2, top_k)
                for source_id in source_ids:
                    balanced.extend(
                        retriever.retrieve(
                            query,
                            top_k=per_source,
                            metadata_filter={"source_id": source_id},
                            minimum_score=float(payload.get("minimum_score", 0.0)),
                        )
                    )
                return balanced
            metadata_filter = payload.get("metadata_filter")
            minimum_score = float(payload.get("minimum_score", 0.0))
            return retriever.retrieve(query, top_k=top_k, metadata_filter=metadata_filter, minimum_score=minimum_score)
        return []

    def _extractive_answer(self, chunks: List[RetrievedChunk]) -> str:
        answer = build_extractive_answer([clean_document_text(chunk.content) for chunk in chunks[:4]])
        return answer or "Information not found in the retrieved document context."

    def _chunk_payload(self, chunk: RetrievedChunk) -> Dict[str, Any]:
        payload = _dump_model(chunk)
        payload["content"] = clean_document_text(payload.get("content", ""))
        return payload
