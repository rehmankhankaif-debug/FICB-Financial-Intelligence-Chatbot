from __future__ import annotations

import time
from collections import defaultdict, deque
from threading import RLock
from typing import Deque, Dict, Tuple


class RateLimitExceeded(RuntimeError):
    pass


class SlidingWindowRateLimiter:
    def __init__(self) -> None:
        self._events: Dict[Tuple[str, str], Deque[float]] = defaultdict(deque)
        self._lock = RLock()

    def check(self, subject: str, action: str, limit: int, window_seconds: float = 60.0) -> None:
        now = time.monotonic()
        key = (str(subject or "anonymous"), str(action or "request"))
        with self._lock:
            events = self._events[key]
            cutoff = now - max(1.0, float(window_seconds))
            while events and events[0] <= cutoff:
                events.popleft()
            if len(events) >= max(1, int(limit)):
                raise RateLimitExceeded("Too many {0} requests. Please wait and try again.".format(action))
            events.append(now)


def can_upload(role: str) -> bool:
    return str(role or "").lower() in {"admin", "user", "analyst"}
