from __future__ import annotations

import pandas as pd

from src.table_intelligence.profiler import TableProfiler
from src.table_intelligence.semantic_column_mapper import SemanticColumnMapper


def _profile_for_columns():
    dataframe = pd.DataFrame(
        {
            "batter": ["Virat Kohli", "MS Dhoni"],
            "batsman_runs": [100, 90],
            "sr": [150.0, 140.0],
            "winner": ["Royal Challengers Bangalore", "Chennai Super Kings"],
            "transmission": ["Manual", "Automatic"],
            "Revenue Amount": [1000, 2000],
        }
    )
    return TableProfiler().profile(dataframe)


def test_maps_runs_to_batsman_runs_using_synonyms() -> None:
    match = SemanticColumnMapper().match_column("runs", _profile_for_columns())

    assert match.matched_column == "batsman_runs"
    assert match.confidence >= 0.9
    assert match.strategy in {"synonym", "exact", "fuzzy"}


def test_maps_strike_rate_to_sr() -> None:
    match = SemanticColumnMapper().match_column("strike rate", _profile_for_columns())

    assert match.matched_column == "sr"
    assert match.confidence >= 0.9


def test_maps_sales_to_revenue_amount() -> None:
    match = SemanticColumnMapper().match_column("sales", _profile_for_columns())

    assert match.matched_column == "Revenue Amount"
    assert match.confidence >= 0.55


def test_maps_manual_automatic_to_transmission_from_sample_values() -> None:
    match = SemanticColumnMapper().match_column("manual automatic", _profile_for_columns())

    assert match.matched_column == "transmission"
    assert match.strategy == "semantic"


def test_unknown_column_does_not_blindly_assume() -> None:
    match = SemanticColumnMapper().match_column("warehouse humidity", _profile_for_columns())

    assert match.matched_column is None
    assert match.confidence < 0.55
    assert match.alternatives


def test_maps_finance_survey_terms_to_finance_data_columns() -> None:
    dataframe = pd.DataFrame(
        {
            "Equity_Market": [2, 1],
            "Stock_Marktet": ["Yes", "No"],
            "Avenue": ["Mutual Fund", "Equity"],
            "Expect": ["20%-30%", "10%-20%"],
            "Source": ["Internet", "Television"],
        }
    )
    profile = TableProfiler().profile(dataframe)
    mapper = SemanticColumnMapper()

    assert mapper.match_column("stock market", profile).matched_column == "Stock_Marktet"
    assert mapper.match_column("equity market", profile).matched_column == "Equity_Market"
    assert mapper.match_column("investment avenue", profile).matched_column == "Avenue"
    assert mapper.match_column("expected return", profile).matched_column == "Expect"
    assert mapper.match_column("financial information source", profile).matched_column == "Source"
