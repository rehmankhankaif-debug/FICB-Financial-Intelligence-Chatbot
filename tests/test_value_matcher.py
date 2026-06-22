from __future__ import annotations

import pandas as pd

from src.table_intelligence.value_matcher import ValueMatcher


def test_matches_partial_player_name() -> None:
    dataframe = pd.DataFrame({"batter": ["Virat Kohli", "Rohit Sharma", "MS Dhoni"]})

    match = ValueMatcher().match_value("Virat", "batter", dataframe)

    assert match.matched_value == "Virat Kohli"
    assert match.confidence >= 0.8
    assert match.strategy == "contains"


def test_matches_team_alias_rcb() -> None:
    dataframe = pd.DataFrame(
        {"winner": ["Royal Challengers Bangalore", "Mumbai Indians", "Chennai Super Kings"]}
    )

    match = ValueMatcher().match_value("RCB", "winner", dataframe)

    assert match.matched_value == "Royal Challengers Bangalore"
    assert match.confidence >= 0.85
    assert match.strategy == "alias"


def test_exact_value_match() -> None:
    dataframe = pd.DataFrame({"transmission": ["Manual", "Automatic"]})

    match = ValueMatcher().match_value("Manual", "transmission", dataframe)

    assert match.matched_value == "Manual"
    assert match.confidence == 1.0


def test_unknown_value_returns_no_arbitrary_match() -> None:
    dataframe = pd.DataFrame({"batter": ["Virat Kohli", "Rohit Sharma"]})

    match = ValueMatcher().match_value("Completely Unknown", "batter", dataframe)

    assert match.matched_value is None
    assert match.confidence < 0.6


def test_missing_column_returns_safe_no_match() -> None:
    dataframe = pd.DataFrame({"batter": ["Virat Kohli"]})

    match = ValueMatcher().match_value("Virat", "player", dataframe)

    assert match.matched_value is None
    assert match.confidence == 0.0
