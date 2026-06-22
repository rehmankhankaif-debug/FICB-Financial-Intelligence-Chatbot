from __future__ import annotations

from pathlib import Path
from hashlib import sha256
from typing import Any, Optional
from uuid import uuid4

from src.config import settings
from src.models.document import DocumentSource
from src.utils.security import (
    get_file_type,
    sanitize_filename,
    validate_file_extension,
    validate_file_signature,
    validate_file_size,
)
from src.security.upload_scanner import scan_upload_content


def create_failed_upload_source(filename: str, error_msg: str) -> DocumentSource:
    return DocumentSource(
        source_id="",
        filename=sanitize_filename(filename),
        file_type=get_file_type(filename),
        path="",
        status="failed",
        error_msg=error_msg,
    )


def save_uploaded_file(uploaded_file: Any, upload_dir: Optional[Path] = None) -> DocumentSource:
    target_dir = upload_dir or settings.upload_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    original_name = getattr(uploaded_file, "name", "")
    file_size = int(getattr(uploaded_file, "size", 0) or 0)
    safe_name = sanitize_filename(original_name)
    file_type = get_file_type(safe_name)

    if not validate_file_extension(safe_name):
        return create_failed_upload_source(original_name, f"Unsupported file type: {file_type or 'unknown'}")

    if not validate_file_size(file_size):
        return create_failed_upload_source(original_name, "File size exceeds the configured upload limit.")

    source_id = uuid4().hex
    stored_filename = f"{source_id}_{safe_name}"
    destination = target_dir / stored_filename

    if hasattr(uploaded_file, "seek"):
        uploaded_file.seek(0)

    content = bytes(uploaded_file.getbuffer())
    if not validate_file_signature(safe_name, content):
        return create_failed_upload_source(original_name, "File content does not match the declared file type.")

    scan_result = scan_upload_content(safe_name, content)
    if not scan_result.safe:
        return create_failed_upload_source(original_name, "Security scan rejected the file: {0}".format(scan_result.reason))

    content_hash = sha256(content).hexdigest()

    destination.write_bytes(content)

    return DocumentSource(
        source_id=source_id,
        filename=safe_name,
        file_type=file_type,
        path=str(destination),
        metadata={
            "original_filename": original_name,
            "stored_filename": stored_filename,
            "size_bytes": file_size,
            "content_sha256": content_hash,
            "security_scanner": scan_result.scanner,
        },
        status="uploaded",
    )
