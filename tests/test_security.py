from __future__ import annotations

from src.config import MAX_FILE_SIZE_BYTES
from src.utils.security import (
    get_file_type,
    is_supported_file,
    sanitize_filename,
    validate_file_extension,
    validate_file_size,
)


def test_supported_file_extensions_pass() -> None:
    assert validate_file_extension("report.csv")
    assert validate_file_extension("report.xlsx")
    assert validate_file_extension("report.xls")
    assert validate_file_extension("report.pdf")
    assert validate_file_extension("report.docx")
    assert validate_file_extension("report.txt")
    assert validate_file_extension("report.html")


def test_unsupported_file_extensions_fail() -> None:
    assert not validate_file_extension("script.exe")
    assert not is_supported_file("archive.zip")


def test_file_size_under_limit_passes() -> None:
    assert validate_file_size(MAX_FILE_SIZE_BYTES)
    assert validate_file_size(1024)


def test_file_size_over_limit_fails() -> None:
    assert not validate_file_size(MAX_FILE_SIZE_BYTES + 1)
    assert not validate_file_size(-1)


def test_filename_sanitization_removes_unsafe_characters() -> None:
    sanitized = sanitize_filename("../my unsafe report @#$%.csv")
    assert "/" not in sanitized
    assert "\\" not in sanitized
    assert " " not in sanitized
    assert sanitized.endswith(".csv")


def test_get_file_type_returns_expected_type() -> None:
    assert get_file_type("REPORT.CSV") == "csv"
    assert get_file_type("financials.xlsx") == "xlsx"
    assert get_file_type("report") == ""
