from __future__ import annotations

import re
from typing import Any, Dict, List

from src.models.document import RetrievedChunk
from src.models.query import QueryPlan
from src.models.tool import ToolResult
from src.rag.citations import CitationBuilder
from src.tools.base import BaseTool
from src.utils.text_summary import build_extractive_summary, clean_document_text, extract_candidate_sentences


MAX_SUMMARY_CONTEXT_CHARS = 24000
MAX_SUMMARY_CONTEXT_CHUNKS = 64
MAX_FALLBACK_POINTS = 7
RESUME_HEADINGS = [
    "PROFESSIONAL SUMMARY",
    "EDUCATION",
    "TECHNICAL SKILLS",
    "PROFESSIONAL EXPERIENCE",
    "KEY PROJECTS & ACHIEVEMENTS",
    "CERTIFICATIONS",
    "ACHIEVEMENTS",
    "PERSONAL DETAILS",
]


class SummarizeTool(BaseTool):
    name = "summarize_tool"
    description = "Create extractive summaries, outlines, key points, TLDRs, and executive summaries from retrieved chunks."
    supported_intents = ["summarize_document"]
    supported_source_types = ["document", "pdf", "docx", "url", "txt", "html"]
    input_types = ["RetrievedChunk", "document chunks"]
    output_types = ["summary", "citations", "metadata"]
    input_requirements = ["retrieved_chunks or document_chunks"]
    capabilities = ["outline", "summarize", "extract_key_points", "tldr", "executive_summary"]
    positive_examples = ["outline this report"]
    negative_examples = ["average monthly profit"]
    confidence = 0.87

    def run(self, input_payload: Dict[str, Any]) -> ToolResult:
        try:
            query_plan = QueryPlan(**(input_payload.get("query_plan") or {}))
            chunks = self._chunks(input_payload or {})
            if not chunks:
                return ToolResult(
                    success=False,
                    tool_name=self.name,
                    answer=None,
                    citations=[],
                    confidence=0.0,
                    warnings=[],
                    error_msg="No document chunks were provided for summarization.",
                    metadata={},
                )

            citations = CitationBuilder().build(chunks)
            mode = self._mode(query_plan)
            answer = self._summarize(chunks, mode)
            summary_context, context_truncated = self._narration_context(chunks)
            return ToolResult(
                success=True,
                tool_name=self.name,
                data={
                    "mode": mode,
                    "chunk_count": len(chunks),
                    "summary_context": summary_context,
                    "context_truncated": context_truncated,
                },
                answer=answer,
                citations=citations,
                confidence=self.confidence,
                warnings=[],
                metadata={"mode": mode, "citation_count": len(citations)},
            )
        except Exception as exc:
            return ToolResult(success=False, tool_name=self.name, confidence=0.0, error_msg="Summarization failed safely: {0}".format(str(exc)), metadata={})

    def _chunks(self, payload: Dict[str, Any]) -> List[RetrievedChunk]:
        raw_chunks = payload.get("retrieved_chunks") or payload.get("document_chunks") or []
        chunks = []
        for index, item in enumerate(raw_chunks):
            if isinstance(item, RetrievedChunk):
                chunks.append(item)
            elif isinstance(item, dict):
                chunks.append(RetrievedChunk(**item))
            else:
                chunks.append(RetrievedChunk(chunk_id="summary_chunk_{0}".format(index), content=str(item)))
        return chunks

    def _mode(self, query_plan: QueryPlan) -> str:
        text = "{0} {1}".format(query_plan.original_query, query_plan.rewritten_query).lower()
        if "outline" in text:
            return "outline"
        if "key point" in text:
            return "key_points"
        if "tldr" in text or "tl;dr" in text:
            return "tldr"
        if "executive" in text:
            return "executive_summary"
        return "summary"

    def _summarize(self, chunks: List[RetrievedChunk], mode: str) -> str:
        full_text = self._join_chunk_text(chunks)
        resume_answer = self._resume_summary(full_text, mode)
        if resume_answer:
            return resume_answer
        if len(chunks) > 2:
            coverage_answer = self._coverage_summary(chunks, mode)
            if coverage_answer:
                return coverage_answer

        answer = build_extractive_summary([clean_document_text(chunk.content) for chunk in chunks], mode=mode)
        if answer:
            return answer

        points = []
        for chunk in chunks[:6]:
            text = " ".join(clean_document_text(chunk.content).split())
            if text:
                points.append(text[:220].rsplit(" ", 1)[0].rstrip())
        return " ".join(point for point in points if point)

    def _join_chunk_text(self, chunks: List[RetrievedChunk]) -> str:
        combined = ""
        for chunk in chunks:
            content = clean_document_text(chunk.content).strip()
            if not content:
                continue
            overlap = 0
            max_overlap = min(500, len(combined), len(content))
            for size in range(max_overlap, 19, -1):
                if combined[-size:].casefold() == content[:size].casefold():
                    overlap = size
                    break
            combined = "{0}\n{1}".format(combined.rstrip(), content[overlap:].lstrip()).strip()
        return combined

    def _resume_summary(self, text: str, mode: str) -> str:
        upper = text.upper()
        if "PROFESSIONAL EXPERIENCE" not in upper or "TECHNICAL SKILLS" not in upper:
            return ""
        sections = self._resume_sections(text)
        summary = self._compact_section(sections.get("PROFESSIONAL SUMMARY", ""), 520)
        education = self._compact_section(sections.get("EDUCATION", ""), 280)
        skills = self._bullet_lines(sections.get("TECHNICAL SKILLS", ""), 5)
        experience = self._bullet_lines(sections.get("PROFESSIONAL EXPERIENCE", ""), 4)
        projects = self._bullet_lines(sections.get("KEY PROJECTS & ACHIEVEMENTS", ""), 4)
        achievements = self._bullet_lines(sections.get("ACHIEVEMENTS", ""), 3)

        name_match = re.match(r"\s*([A-Z][A-Z ]{3,}?)(?=\s+\+?\d|\s+[\w.+-]+@)", text)
        candidate_name = name_match.group(1).strip() if name_match else "Candidate"
        parts = ["## Resume summary", "**{0}**".format(candidate_name)]
        if summary:
            parts.extend(["", summary])
        if education:
            parts.extend(["", "### Education", education])
        for heading, values in [
            ("Core skills", skills),
            ("Experience highlights", experience),
            ("Projects", projects),
            ("Selected achievements", achievements),
        ]:
            if values:
                parts.extend(["", "### {0}".format(heading)])
                parts.extend("- {0}".format(value) for value in values)
        return "\n".join(parts)

    def _resume_sections(self, text: str) -> Dict[str, str]:
        headings_pattern = "|".join(re.escape(item) for item in RESUME_HEADINGS)
        matches = list(
            re.finditer(
                r"(?i)\b({0})\b\s*(?:[─━—-]{{3,}}\s*)?".format(headings_pattern),
                text,
            )
        )
        sections: Dict[str, str] = {}
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            sections[match.group(1).upper()] = text[match.end():end].strip()
        return sections

    def _compact_section(self, text: str, limit: int) -> str:
        compact = " ".join(line.strip(" ·\t") for line in text.splitlines() if line.strip())
        if len(compact) <= limit:
            return compact
        return compact[:limit].rsplit(" ", 1)[0].rstrip() + "."

    def _bullet_lines(self, text: str, limit: int) -> List[str]:
        lines: List[str] = []
        current = ""
        bullet_normalized = re.sub(r"\s*[·•]\s*", "\n· ", text)
        for raw_line in bullet_normalized.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            starts_bullet = line.startswith(("·", "•", "-"))
            cleaned = line.lstrip("·•- ").strip()
            if starts_bullet:
                if current:
                    lines.append(current)
                current = cleaned
            elif current:
                current = "{0} {1}".format(current, cleaned)
            else:
                current = cleaned
            if len(lines) >= limit:
                break
        if current and len(lines) < limit:
            lines.append(current)
        return [self._compact_section(line, 360) for line in lines[:limit] if line]

    def _coverage_summary(self, chunks: List[RetrievedChunk], mode: str) -> str:
        points: List[str] = []
        seen = set()
        for chunk in self._evenly_sample(chunks, MAX_FALLBACK_POINTS):
            candidates = extract_candidate_sentences([clean_document_text(chunk.content)], limit=1)
            if not candidates:
                continue
            point = candidates[0]
            key = point.lower()
            if key in seen:
                continue
            seen.add(key)
            points.append(point)

        if not points:
            return ""
        if mode == "outline":
            return "\n".join("{0}. {1}".format(index + 1, point) for index, point in enumerate(points))
        if mode == "key_points":
            return "\n".join("- {0}".format(point) for point in points)
        if mode == "tldr":
            return " ".join(points[:2])

        heading = "Executive summary" if mode == "executive_summary" else "Summary"
        lead = points[0]
        if len(points) == 1:
            return "{0}: {1}".format(heading, lead)
        return "{0}: {1}\n\nKey points:\n{2}".format(
            heading,
            lead,
            "\n".join("- {0}".format(point) for point in points[1:]),
        )

    def _narration_context(self, chunks: List[RetrievedChunk]) -> tuple[List[Dict[str, Any]], bool]:
        sampled = self._evenly_sample(chunks, MAX_SUMMARY_CONTEXT_CHUNKS)
        if not sampled:
            return [], False
        per_chunk_budget = max(240, MAX_SUMMARY_CONTEXT_CHARS // len(sampled))
        context: List[Dict[str, Any]] = []
        truncated = len(sampled) < len(chunks)
        used_chars = 0
        for chunk in sampled:
            content = clean_document_text(chunk.content)
            remaining = MAX_SUMMARY_CONTEXT_CHARS - used_chars
            if remaining <= 0:
                truncated = True
                break
            allowed = min(per_chunk_budget, remaining)
            excerpt = content[:allowed].strip()
            if len(excerpt) < len(content):
                truncated = True
            if excerpt:
                context.append({"page": chunk.page, "content": excerpt})
                used_chars += len(excerpt)
        return context, truncated

    def _evenly_sample(self, chunks: List[RetrievedChunk], limit: int) -> List[RetrievedChunk]:
        if len(chunks) <= limit:
            return list(chunks)
        if limit <= 1:
            return [chunks[0]]
        indexes = [round(index * (len(chunks) - 1) / float(limit - 1)) for index in range(limit)]
        return [chunks[index] for index in indexes]
