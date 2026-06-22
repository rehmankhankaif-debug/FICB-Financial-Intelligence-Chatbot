from __future__ import annotations

import re
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
            answer = self._extractive_answer_for_query(query_plan, chunks)
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
        return self._extractive_answer_for_query(QueryPlan(), chunks)

    def _extractive_answer_for_query(self, query_plan: QueryPlan, chunks: List[RetrievedChunk]) -> str:
        profile_answer = self._profile_answer(query_plan, chunks)
        if profile_answer:
            return profile_answer
        answer = build_extractive_answer([clean_document_text(chunk.content) for chunk in chunks[:4]])
        return answer or "Information not found in the retrieved document context."

    def _profile_answer(self, query_plan: QueryPlan, chunks: List[RetrievedChunk]) -> str:
        query_text = "{0} {1}".format(query_plan.original_query or "", query_plan.rewritten_query or "").lower()
        if not any(term in query_text for term in {"cv", "resume", "profile", "owner", "name", "who", "where", "projects", "skills", "experience"}):
            return ""

        combined = "\n".join(clean_document_text(chunk.content) for chunk in chunks if chunk.content).strip()
        upper = combined.upper()
        if not combined or not any(term in upper for term in {"PROFESSIONAL SUMMARY", "TECHNICAL SKILLS", "PROFESSIONAL EXPERIENCE"}):
            return ""

        ask_name = any(term in query_text for term in {"owner", "name", "who", "whose"})
        ask_role = any(term in query_text for term in {"what does", "what he does", "what she does", "profession", "role", "work", "does he do", "does she do"})
        ask_location = any(term in query_text for term in {"where", "from", "location"})
        ask_projects = "project" in query_text
        ask_skills = "skill" in query_text

        name = self._extract_profile_name(combined)
        role = self._extract_profile_role(combined)
        location = self._extract_profile_location(combined)
        projects = self._extract_profile_bullets(combined, "KEY PROJECTS & ACHIEVEMENTS", 3)
        skills = self._extract_profile_skills(combined)

        parts: List[str] = []
        if ask_name and name:
            parts.append("{0} is the owner of this CV.".format(name))
        if ask_role and role:
            subject = name or "The candidate"
            parts.append("{0} is {1}.".format(subject, role))
        if ask_location and location:
            subject = name or "The candidate"
            parts.append("{0} is from {1}.".format(subject, location))
        if ask_projects and projects:
            parts.append("Key projects include {0}.".format("; ".join(projects)))
        if ask_skills and skills:
            parts.append("Core skills include {0}.".format(", ".join(skills)))
        return " ".join(parts).strip()

    def _extract_profile_name(self, text: str) -> str:
        match = re.search(r"^\s*([A-Z][A-Z ]{3,}?)(?=\s+\+?\d|\s+[\w.+-]+@|\s+\|)", text)
        if not match:
            return ""
        return match.group(1).title().strip()

    def _extract_profile_role(self, text: str) -> str:
        match = re.search(r"PROFESSIONAL SUMMARY\s+(.*?)(?:\n[A-Z][A-Z &/]{3,}|\Z)", text, flags=re.IGNORECASE | re.DOTALL)
        if not match:
            return ""
        summary = " ".join(match.group(1).split())
        role_match = re.search(r"([A-Z][A-Za-z/& -]{2,80}?) with ", summary)
        if role_match:
            return role_match.group(1).strip()
        sentence_match = re.search(r"([A-Z][^.]{8,140}\.)", summary)
        if sentence_match:
            return sentence_match.group(1).rstrip(".")
        return summary[:120].rsplit(" ", 1)[0].strip() if summary else ""

    def _extract_profile_location(self, text: str) -> str:
        match = re.search(r"\|\s*([A-Za-z][A-Za-z .'-]+,\s*[A-Za-z][A-Za-z .'-]+,\s*[A-Za-z][A-Za-z .'-]+)\s+PROFESSIONAL SUMMARY", text)
        if match:
            return match.group(1).strip()
        for line in text.splitlines()[:4]:
            compact = " ".join(line.split())
            if compact.count(",") >= 2 and "@" not in compact and "linkedin" not in compact.lower():
                return compact.strip(" |")
        return ""

    def _extract_profile_bullets(self, text: str, heading: str, limit: int) -> List[str]:
        match = re.search(
            r"{0}\s+(.*?)(?:\n[A-Z][A-Z &/]{{3,}}|\Z)".format(re.escape(heading)),
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if not match:
            return []
        section = match.group(1)
        bullets = []
        normalized = re.sub(r"\s*[·•]\s*", "\n- ", section)
        for raw_line in normalized.splitlines():
            line = raw_line.strip().lstrip("- ").strip()
            if line:
                bullets.append(line[:160].rsplit(" ", 1)[0].rstrip() if len(line) > 160 else line)
            if len(bullets) >= limit:
                break
        return bullets

    def _extract_profile_skills(self, text: str) -> List[str]:
        match = re.search(r"TECHNICAL SKILLS\s+(.*?)(?:\n[A-Z][A-Z &/]{3,}|\Z)", text, flags=re.IGNORECASE | re.DOTALL)
        if not match:
            return []
        section = " ".join(match.group(1).split())
        pieces = [item.strip(" .,-") for item in re.split(r"[,|/]", section) if item.strip()]
        return pieces[:8]

    def _chunk_payload(self, chunk: RetrievedChunk) -> Dict[str, Any]:
        payload = _dump_model(chunk)
        payload["content"] = clean_document_text(payload.get("content", ""))
        return payload
