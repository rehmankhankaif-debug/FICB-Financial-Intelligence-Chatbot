from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import PurePosixPath
from zipfile import BadZipFile, ZipFile


EICAR_MARKER = b"X5O!P%@AP[4\\PZX54(P^)7CC)7}$EICAR-STANDARD-ANTIVIRUS-TEST-FILE!$H+H*"
EXECUTABLE_HEADERS = (b"MZ", b"\x7fELF", b"\xcf\xfa\xed\xfe", b"\xca\xfe\xba\xbe")
MAX_ARCHIVE_ENTRIES = 5000
MAX_ARCHIVE_UNCOMPRESSED_BYTES = 250 * 1024 * 1024


@dataclass(frozen=True)
class UploadScanResult:
    safe: bool
    reason: str = ""
    scanner: str = "built_in"


def scan_upload_content(filename: str, content: bytes) -> UploadScanResult:
    payload = bytes(content or b"")
    lowered_name = str(filename or "").lower()
    if EICAR_MARKER in payload:
        return UploadScanResult(False, "Known antivirus test signature detected.")
    if payload.startswith(EXECUTABLE_HEADERS):
        return UploadScanResult(False, "Executable content is not allowed.")
    if lowered_name.endswith((".docx", ".xlsx")):
        return _scan_office_archive(payload)
    return UploadScanResult(True)


def _scan_office_archive(content: bytes) -> UploadScanResult:
    try:
        with ZipFile(BytesIO(content)) as archive:
            entries = archive.infolist()
            if len(entries) > MAX_ARCHIVE_ENTRIES:
                return UploadScanResult(False, "Archive contains too many entries.")
            total_size = sum(max(0, item.file_size) for item in entries)
            if total_size > MAX_ARCHIVE_UNCOMPRESSED_BYTES:
                return UploadScanResult(False, "Archive expands beyond the safe processing limit.")
            for item in entries:
                path = PurePosixPath(item.filename)
                if path.is_absolute() or ".." in path.parts:
                    return UploadScanResult(False, "Archive contains an unsafe path.")
                if item.filename.lower().endswith(("vbaproject.bin", ".exe", ".dll", ".js", ".vbs")):
                    return UploadScanResult(False, "Archive contains active or executable content.")
    except BadZipFile:
        return UploadScanResult(False, "Office document archive is corrupted.")
    return UploadScanResult(True)
