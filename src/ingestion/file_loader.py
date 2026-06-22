from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.errors import IngestionError, UnsupportedFileError
from src.utils.security import get_file_type


CSV_EXTENSIONS = {"csv"}
EXCEL_EXTENSIONS = {"xlsx", "xls"}


def _validate_existing_file(path: Any, allowed_extensions: set, loader_name: str) -> Path:
    file_path = Path(path)
    if not file_path.exists():
        raise IngestionError(
            "File does not exist.",
            metadata={"path": str(file_path), "loader": loader_name},
        )
    if not file_path.is_file():
        raise IngestionError(
            "Path is not a file.",
            metadata={"path": str(file_path), "loader": loader_name},
        )

    file_type = get_file_type(str(file_path))
    if file_type not in allowed_extensions:
        raise UnsupportedFileError(
            "Unsupported file type for loader.",
            metadata={
                "path": str(file_path),
                "file_type": file_type,
                "allowed_extensions": sorted(allowed_extensions),
                "loader": loader_name,
            },
        )
    return file_path


def _ensure_dataframe(dataframe: pd.DataFrame, path: Path, loader_name: str) -> pd.DataFrame:
    if dataframe is None:
        raise IngestionError(
            "Loader returned no dataframe.",
            metadata={"path": str(path), "loader": loader_name},
        )
    if not isinstance(dataframe, pd.DataFrame):
        raise IngestionError(
            "Loader returned an invalid object.",
            metadata={
                "path": str(path),
                "loader": loader_name,
                "returned_type": type(dataframe).__name__,
            },
        )
    return dataframe


def load_csv(path: Any) -> pd.DataFrame:
    file_path = _validate_existing_file(path, CSV_EXTENSIONS, "load_csv")
    try:
        dataframe = pd.read_csv(file_path)
    except Exception as exc:
        raise IngestionError(
            "Failed to load CSV file.",
            metadata={"path": str(file_path), "error": str(exc)},
        )
    return _ensure_dataframe(dataframe, file_path, "load_csv")


def load_excel(path: Any) -> pd.DataFrame:
    file_path = _validate_existing_file(path, EXCEL_EXTENSIONS, "load_excel")
    try:
        dataframe = pd.read_excel(file_path)
    except Exception as exc:
        raise IngestionError(
            "Failed to load Excel file.",
            metadata={"path": str(file_path), "error": str(exc)},
        )
    return _ensure_dataframe(dataframe, file_path, "load_excel")
