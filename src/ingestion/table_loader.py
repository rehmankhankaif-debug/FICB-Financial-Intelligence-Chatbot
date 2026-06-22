from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.ingestion.file_loader import load_csv, load_excel
from src.utils.errors import UnsupportedFileError
from src.utils.security import get_file_type


def load_table(path: Any) -> pd.DataFrame:
    file_type = get_file_type(str(Path(path)))
    if file_type == "csv":
        return load_csv(path)
    if file_type in {"xlsx", "xls"}:
        return load_excel(path)
    raise UnsupportedFileError(
        "Only CSV and Excel files are supported by the structured table loader.",
        metadata={"path": str(path), "file_type": file_type},
    )
