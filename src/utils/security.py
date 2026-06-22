from __future__ import annotations

import re
from pathlib import Path

from src.config import ALLOWED_FILE_EXTENSIONS, MAX_FILE_SIZE_BYTES

ZIP_BASED_EXTENSIONS = {"docx", "xlsx"}
OLE_EXTENSIONS = {"xls"}


def sanitize_filename(filename: str) -> str:
    base_name = Path(filename or "").name
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", base_name).strip("._")
    return sanitized or "uploaded_file"


def get_file_type(filename: str) -> str:
    suffix = Path(filename or "").suffix.lower().lstrip(".")
    return suffix


def is_supported_file(filename: str) -> bool:
    return get_file_type(filename) in ALLOWED_FILE_EXTENSIONS


def validate_file_extension(filename: str) -> bool:
    return is_supported_file(filename)


def validate_file_size(file_size: int) -> bool:
    if file_size < 0:
        return False
    return file_size <= MAX_FILE_SIZE_BYTES


def validate_file_signature(filename: str, content: bytes) -> bool:
    file_type = get_file_type(filename)
    header = bytes(content or b"")[:16]
    stripped = bytes(content or b"")[:128].lstrip()

    if file_type == "pdf":
        return header.startswith(b"%PDF")
    if file_type in ZIP_BASED_EXTENSIONS:
        return header.startswith(b"PK\x03\x04") or header.startswith(b"PK\x05\x06") or header.startswith(b"PK\x07\x08")
    if file_type in OLE_EXTENSIONS:
        return header.startswith(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1")
    if file_type == "html":
        lowered = stripped.lower()
        return lowered.startswith((b"<!doctype html", b"<html", b"<head", b"<body"))
    return True
