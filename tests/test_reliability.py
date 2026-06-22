from __future__ import annotations

import pytest

from src.reliability import CircuitBreaker, CircuitOpenError, RetryPolicy, execute_with_retries


def test_execute_with_retries_recovers_from_transient_error() -> None:
    attempts = {"count": 0}

    def flaky_operation() -> str:
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RuntimeError("temporary")
        return "ok"

    result = execute_with_retries(
        flaky_operation,
        RetryPolicy(max_attempts=2, backoff_seconds=0),
    )

    assert result == "ok"
    assert attempts["count"] == 2


def test_circuit_breaker_opens_after_failures() -> None:
    breaker = CircuitBreaker(failure_threshold=1, recovery_seconds=60)

    with pytest.raises(RuntimeError):
        execute_with_retries(
            lambda: (_ for _ in ()).throw(RuntimeError("downstream failed")),
            RetryPolicy(max_attempts=1, backoff_seconds=0),
            circuit_breaker=breaker,
        )

    with pytest.raises(CircuitOpenError):
        execute_with_retries(
            lambda: "ok",
            RetryPolicy(max_attempts=1, backoff_seconds=0),
            circuit_breaker=breaker,
        )
