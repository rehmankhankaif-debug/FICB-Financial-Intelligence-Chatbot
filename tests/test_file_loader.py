from __future__ import annotations

import pandas as pd
import pytest

from src.ingestion.file_loader import load_csv, load_excel
from src.ingestion.table_loader import load_table
from src.utils.errors import IngestionError, UnsupportedFileError


def test_load_csv_returns_dataframe(tmp_path) -> None:
    path = tmp_path / "sales.csv"
    pd.DataFrame({"month": ["Jan"], "profit": [100]}).to_csv(path, index=False)

    dataframe = load_csv(path)

    assert isinstance(dataframe, pd.DataFrame)
    assert dataframe.loc[0, "profit"] == 100


def test_load_excel_returns_dataframe(tmp_path) -> None:
    path = tmp_path / "sales.xlsx"
    pd.DataFrame({"month": ["Jan"], "profit": [100]}).to_excel(path, index=False)

    dataframe = load_excel(path)

    assert isinstance(dataframe, pd.DataFrame)
    assert dataframe.loc[0, "month"] == "Jan"


def test_load_table_dispatches_csv(tmp_path) -> None:
    path = tmp_path / "sales.csv"
    pd.DataFrame({"sales": [10, 20]}).to_csv(path, index=False)

    dataframe = load_table(path)

    assert list(dataframe.columns) == ["sales"]


def test_missing_file_raises_ingestion_error(tmp_path) -> None:
    with pytest.raises(IngestionError):
        load_csv(tmp_path / "missing.csv")


def test_unsupported_loader_type_raises_custom_error(tmp_path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("hello", encoding="utf-8")

    with pytest.raises(UnsupportedFileError):
        load_csv(path)


def test_load_table_rejects_pdf_for_phase_2(tmp_path) -> None:
    path = tmp_path / "report.pdf"
    path.write_bytes(b"%PDF")

    with pytest.raises(UnsupportedFileError):
        load_table(path)
