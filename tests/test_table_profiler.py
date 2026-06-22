from __future__ import annotations

import pandas as pd

from src.table_intelligence.profiler import TableProfiler, normalize_column_name


def test_profile_detects_core_table_shape_and_columns() -> None:
    dataframe = pd.DataFrame(
        {
            "batter": ["Virat Kohli", "Rohit Sharma", "MS Dhoni"],
            "batsman_runs": [80, 70, 60],
            "sr": [145.5, 138.2, 155.0],
            "winner": ["Royal Challengers Bangalore", "Mumbai Indians", "Chennai Super Kings"],
            "match_date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        }
    )

    profile = TableProfiler().profile(dataframe, source_id="ipl", filename="ipl.csv")

    assert profile.shape == (3, 5)
    assert profile.columns == ["batter", "batsman_runs", "sr", "winner", "match_date"]
    assert profile.normalized_columns["batsman_runs"] == "batsman_runs"


def test_profile_detects_entity_metric_datetime_and_result_candidates() -> None:
    dataframe = pd.DataFrame(
        {
            "batter": ["Virat Kohli", "Rohit Sharma", "MS Dhoni"],
            "batsman_runs": [80, 70, 60],
            "winner": ["RCB", "MI", "CSK"],
            "match_date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        }
    )

    profile = TableProfiler().profile(dataframe)

    assert "batter" in profile.entity_candidate_columns
    assert "batsman_runs" in profile.metric_candidate_columns
    assert "winner" in profile.result_candidate_columns
    assert "match_date" in profile.datetime_columns


def test_profile_generates_samples_missing_values_unique_values_and_stats() -> None:
    dataframe = pd.DataFrame(
        {
            "region": ["North", "South", "North", None],
            "profit": [100.0, 200.0, None, 300.0],
        }
    )

    profile = TableProfiler().profile(dataframe)

    assert profile.sample_values["region"] == ["North", "South"]
    assert profile.missing_values["region"] == 1
    assert profile.unique_values["region"] == ["North", "South"]
    assert profile.numeric_stats["profit"]["sum"] == 600.0
    assert "profit" in profile.metric_candidate_columns


def test_empty_dataframe_profile_is_safe() -> None:
    dataframe = pd.DataFrame(columns=["sales", "status"])

    profile = TableProfiler().profile(dataframe)

    assert profile.shape == (0, 2)
    assert profile.sample_values["sales"] == []
    assert profile.missing_values["status"] == 0
    assert "sales" in profile.metric_candidate_columns
    assert "status" in profile.result_candidate_columns


def test_normalize_column_name() -> None:
    assert normalize_column_name("Strike Rate %") == "strike_rate"
