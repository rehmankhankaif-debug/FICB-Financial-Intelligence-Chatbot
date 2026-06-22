from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, TypeVar


T = TypeVar("T")


class CircuitOpenError(RuntimeError):
    """Raised when a protected downstream dependency is temporarily disabled."""


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3
    backoff_seconds: float = 0.25
    backoff_multiplier: float = 2.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1.")
        if self.backoff_seconds < 0:
            raise ValueError("backoff_seconds cannot be negative.")
        if self.backoff_multiplier < 1:
            raise ValueError("backoff_multiplier must be at least 1.")

    def delay_for_attempt(self, attempt_index: int) -> float:
        return self.backoff_seconds * (self.backoff_multiplier ** max(0, attempt_index - 1))


class CircuitBreaker:
    def __init__(self, failure_threshold: int = 3, recovery_seconds: float = 30.0) -> None:
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be at least 1.")
        if recovery_seconds < 0:
            raise ValueError("recovery_seconds cannot be negative.")
        self.failure_threshold = failure_threshold
        self.recovery_seconds = recovery_seconds
        self.failure_count = 0
        self.opened_at: Optional[float] = None

    @property
    def is_open(self) -> bool:
        if self.opened_at is None:
            return False
        if time.monotonic() - self.opened_at >= self.recovery_seconds:
            return False
        return True

    def before_call(self) -> None:
        if self.is_open:
            raise CircuitOpenError("Circuit breaker is open.")

    def record_success(self) -> None:
        self.failure_count = 0
        self.opened_at = None

    def record_failure(self) -> None:
        self.failure_count += 1
        if self.failure_count >= self.failure_threshold:
            self.opened_at = time.monotonic()


def execute_with_retries(
    operation: Callable[[], T],
    retry_policy: RetryPolicy,
    retry_exceptions: Tuple[type[BaseException], ...] = (Exception,),
    circuit_breaker: Optional[CircuitBreaker] = None,
    on_retry: Optional[Callable[[int, BaseException, float], None]] = None,
    should_retry: Optional[Callable[[BaseException], bool]] = None,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    last_error: Optional[BaseException] = None
    for attempt in range(1, retry_policy.max_attempts + 1):
        if circuit_breaker is not None:
            circuit_breaker.before_call()
        try:
            result = operation()
            if circuit_breaker is not None:
                circuit_breaker.record_success()
            return result
        except retry_exceptions as exc:
            last_error = exc
            if should_retry is not None and not should_retry(exc):
                if circuit_breaker is not None:
                    circuit_breaker.record_failure()
                raise
            if attempt >= retry_policy.max_attempts:
                if circuit_breaker is not None:
                    circuit_breaker.record_failure()
                raise
            delay = retry_policy.delay_for_attempt(attempt)
            if on_retry is not None:
                on_retry(attempt, exc, delay)
            if delay > 0:
                sleep(delay)

    if last_error is not None:
        raise last_error
    raise RuntimeError("Retry operation failed without an exception.")
