from __future__ import annotations

from typing import Any, Dict, Optional

from src.config import settings
from src.llm.json_utils import parse_json_safely
from src.reliability.policies import CircuitBreaker, CircuitOpenError, RetryPolicy, execute_with_retries
from src.utils.logging import log_error, log_event


def _is_quota_error(exc: BaseException) -> bool:
    message = str(exc or "").lower()
    return "resource_exhausted" in message or "quota exceeded" in message or "code': 429" in message or " 429 " in message


class GeminiClient:
    _circuit_breaker = CircuitBreaker(
        failure_threshold=settings.circuit_breaker_failure_threshold,
        recovery_seconds=settings.circuit_breaker_recovery_seconds,
    )

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        client: Optional[Any] = None,
    ) -> None:
        self.api_key = api_key if api_key is not None else settings.gemini_api_key
        self.model_name = model_name or settings.default_gemini_model
        self.client = client
        self.last_error: Optional[str] = None
        self.last_error_type: Optional[str] = None

    def is_available(self) -> bool:
        return bool(self.api_key or self.client)

    def generate(self, prompt: str) -> str:
        self.last_error = None
        self.last_error_type = None
        if not self.is_available():
            return ""

        try:
            retry_policy = RetryPolicy(
                max_attempts=settings.retry_max_attempts,
                backoff_seconds=settings.retry_backoff_seconds,
            )

            def _operation() -> str:
                client = self._client()
                if client is None:
                    self.last_error = self.last_error or "Gemini client could not be initialized."
                    self.last_error_type = self.last_error_type or "ClientInitializationError"
                    return ""

                if hasattr(client, "models") and hasattr(client.models, "generate_content"):
                    response = client.models.generate_content(model=self.model_name, contents=prompt)
                elif hasattr(client, "generate_content"):
                    response = client.generate_content(prompt)
                else:
                    self.last_error = "Configured Gemini client does not support content generation."
                    self.last_error_type = "UnsupportedClientError"
                    return ""

                text = getattr(response, "text", None)
                if text is not None:
                    return str(text)
                if isinstance(response, str):
                    return response
                return ""

            return execute_with_retries(
                _operation,
                retry_policy=retry_policy,
                circuit_breaker=self._circuit_breaker,
                on_retry=lambda attempt, exc, delay: log_event(
                    "llm_retry",
                    {
                        "attempt": attempt,
                        "delay_seconds": delay,
                        "error_type": exc.__class__.__name__,
                        "model": self.model_name,
                    },
                    level="warning",
                ),
                should_retry=lambda exc: not _is_quota_error(exc),
            )
        except CircuitOpenError as exc:
            self.last_error = str(exc)
            self.last_error_type = exc.__class__.__name__
            log_error(exc, {"stage": "gemini_generate", "model": self.model_name, "circuit_open": True})
            return ""
        except Exception as exc:
            if _is_quota_error(exc):
                self.last_error = "Gemini request quota is exhausted. Wait for the provider reset window or use a key/model with available quota."
                self.last_error_type = "QuotaExceededError"
            else:
                self.last_error = str(exc)
                self.last_error_type = exc.__class__.__name__
            log_error(exc, {"stage": "gemini_generate", "model": self.model_name})
            return ""

    def generate_json(self, prompt: str, fallback: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        safe_fallback = dict(fallback or {})
        if not self.is_available():
            return safe_fallback
        text = self.generate(prompt)
        return parse_json_safely(text, fallback=safe_fallback)

    def _client(self) -> Optional[Any]:
        if self.client is not None:
            return self.client
        if not self.api_key:
            return None
        try:
            from google import genai

            self.client = genai.Client(api_key=self.api_key)
            return self.client
        except Exception as exc:
            self.last_error = str(exc)
            self.last_error_type = exc.__class__.__name__
            log_error(exc, {"stage": "gemini_client_initialization", "model": self.model_name})
            return None
