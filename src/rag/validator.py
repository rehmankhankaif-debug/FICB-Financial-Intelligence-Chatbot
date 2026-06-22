from __future__ import annotations

from typing import List, Set

from src.models.document import RetrievedChunk
from src.models.validation import ValidationResult


class RetrievalValidator:
    def deduplicate(self, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        seen: Set[str] = set()
        unique_chunks: List[RetrievedChunk] = []
        for chunk in chunks:
            key = chunk.chunk_id or "{0}:{1}".format(chunk.source_id, chunk.content[:80])
            if key in seen:
                continue
            seen.add(key)
            unique_chunks.append(chunk)
        return unique_chunks

    def validate(
        self,
        chunks: List[RetrievedChunk],
        minimum_score: float = 0.1,
        minimum_content_length: int = 20,
    ) -> ValidationResult:
        issues: List[str] = []
        warnings: List[str] = []

        if not chunks:
            return ValidationResult(
                is_valid=False,
                confidence=0.0,
                issues=["Retrieval returned no chunks."],
                requires_retry=True,
            )

        unique_chunks = self.deduplicate(chunks)
        if len(unique_chunks) < len(chunks):
            warnings.append("Duplicate retrieved chunks were detected and removed for validation.")

        low_score_count = sum(1 for chunk in unique_chunks if chunk.score < minimum_score)
        if low_score_count == len(unique_chunks):
            issues.append("All retrieved chunks are below the minimum score threshold.")
        elif low_score_count:
            warnings.append("{0} retrieved chunk(s) are below the minimum score threshold.".format(low_score_count))

        weak_content_count = sum(1 for chunk in unique_chunks if len((chunk.content or "").strip()) < minimum_content_length)
        if weak_content_count:
            warnings.append("{0} retrieved chunk(s) have weak or very short content.".format(weak_content_count))

        average_score = sum(chunk.score for chunk in unique_chunks) / float(len(unique_chunks))
        confidence = max(0.0, min(1.0, average_score))

        return ValidationResult(
            is_valid=not issues,
            confidence=confidence,
            issues=issues,
            warnings=warnings,
            requires_retry=bool(issues),
            clarification_needed=False,
            clarification_question=None,
        )
