from __future__ import annotations

from src.agents.confidence import (
    combine_confidence,
    is_high_confidence,
    is_low_confidence,
    is_medium_confidence,
    normalize_confidence,
)


def test_normalize_confidence_clamps_values() -> None:
    assert normalize_confidence(1.5) == 1.0
    assert normalize_confidence(-0.2) == 0.0
    assert normalize_confidence("bad") == 0.0


def test_combine_confidence_averages_safe_scores() -> None:
    assert combine_confidence([0.91, 0.87, 0.84]) == 0.8733
    assert combine_confidence([]) == 0.0
    assert combine_confidence([0.5, "bad", 2.0]) == 0.75


def test_confidence_threshold_helpers() -> None:
    assert is_high_confidence(0.8) is True
    assert is_medium_confidence(0.7) is True
    assert is_low_confidence(0.54) is True
