from __future__ import annotations

import math
from typing import Iterable


HIGH_CONFIDENCE_THRESHOLD = 0.80
MEDIUM_CONFIDENCE_THRESHOLD = 0.55


def normalize_confidence(score) -> float:
    try:
        value = float(score)
    except Exception:
        return 0.0
    if math.isnan(value) or math.isinf(value):
        return 0.0
    return max(0.0, min(1.0, value))


def clamp_confidence(score) -> float:
    return normalize_confidence(score)


def combine_confidence(scores: Iterable[float]) -> float:
    normalized = []
    for score in scores or []:
        try:
            value = float(score)
        except Exception:
            continue
        if math.isnan(value) or math.isinf(value):
            continue
        normalized.append(normalize_confidence(value))
    if not normalized:
        return 0.0
    return round(sum(normalized) / float(len(normalized)), 4)


def is_high_confidence(score: float) -> bool:
    return normalize_confidence(score) >= HIGH_CONFIDENCE_THRESHOLD


def is_medium_confidence(score: float) -> bool:
    normalized = normalize_confidence(score)
    return MEDIUM_CONFIDENCE_THRESHOLD <= normalized < HIGH_CONFIDENCE_THRESHOLD


def is_low_confidence(score: float) -> bool:
    return normalize_confidence(score) < MEDIUM_CONFIDENCE_THRESHOLD
