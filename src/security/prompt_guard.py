from __future__ import annotations

import re
from typing import List

from pydantic import BaseModel, Field


HIGH_RISK_PATTERNS = {
    "ignore_instructions": re.compile(r"\b(ignore|disregard|forget|bypass)\b.{0,40}\b(previous|system|developer|above|instructions?)\b", re.IGNORECASE),
    "reveal_secrets": re.compile(r"\b(reveal|show|print|dump|expose)\b.{0,40}\b(secret|api[_ -]?key|token|password|system prompt|developer message)\b", re.IGNORECASE),
    "role_override": re.compile(r"\b(you are now|act as|pretend to be)\b.{0,40}\b(system|developer|admin|root)\b", re.IGNORECASE),
    "jailbreak": re.compile(r"\b(jailbreak|dan mode|developer mode|no safety|unfiltered)\b", re.IGNORECASE),
}

DOCUMENT_RISK_PATTERNS = {
    "embedded_instruction": re.compile(r"\b(ignore|disregard|forget)\b.{0,50}\b(user|system|assistant|instructions?)\b", re.IGNORECASE),
    "instruction_to_model": re.compile(r"\b(model|assistant|chatbot|llm)\b.{0,40}\b(must|should|will)\b.{0,40}\b(answer|respond|say)\b", re.IGNORECASE),
    "secret_request": HIGH_RISK_PATTERNS["reveal_secrets"],
}


class PromptInjectionAssessment(BaseModel):
    is_suspicious: bool = False
    should_block: bool = False
    risk_score: float = 0.0
    reasons: List[str] = Field(default_factory=list)

    def warning(self) -> str:
        if not self.is_suspicious:
            return ""
        return "Prompt-injection risk detected: {0}".format(", ".join(self.reasons))


class PromptInjectionGuard:
    def assess_user_query(self, text: str) -> PromptInjectionAssessment:
        return self._assess(text, HIGH_RISK_PATTERNS, block_threshold=0.5)

    def assess_document_text(self, text: str) -> PromptInjectionAssessment:
        return self._assess(text, DOCUMENT_RISK_PATTERNS, block_threshold=1.1)

    def _assess(self, text: str, patterns: dict, block_threshold: float) -> PromptInjectionAssessment:
        value = str(text or "")
        reasons = [name for name, pattern in patterns.items() if pattern.search(value)]
        risk_score = min(1.0, len(reasons) / 2.0)
        return PromptInjectionAssessment(
            is_suspicious=bool(reasons),
            should_block=risk_score >= block_threshold,
            risk_score=risk_score,
            reasons=reasons,
        )
