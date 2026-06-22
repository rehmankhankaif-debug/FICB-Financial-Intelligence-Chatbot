from __future__ import annotations

from src.security.rate_limit import RateLimitExceeded, SlidingWindowRateLimiter, can_upload
from src.security.upload_scanner import EICAR_MARKER, scan_upload_content
from src.utils.upload import save_uploaded_file


class UploadedBytes:
    def __init__(self, name: str, content: bytes) -> None:
        self.name = name
        self.content = content
        self.size = len(content)

    def seek(self, position: int) -> None:
        return None

    def getbuffer(self):
        return memoryview(self.content)


def test_upload_scanner_rejects_antivirus_test_signature() -> None:
    result = scan_upload_content("transactions.csv", b"header\n" + EICAR_MARKER)

    assert result.safe is False
    assert "antivirus" in result.reason.lower()


def test_saved_upload_contains_stable_content_hash(tmp_path) -> None:
    first = save_uploaded_file(UploadedBytes("data.csv", b"Gender\nF\nM\n"), upload_dir=tmp_path)
    second = save_uploaded_file(UploadedBytes("copy.csv", b"Gender\nF\nM\n"), upload_dir=tmp_path)

    assert first.status == "uploaded"
    assert first.metadata["content_sha256"] == second.metadata["content_sha256"]


def test_save_uploaded_file_rejects_detected_malware(tmp_path) -> None:
    result = save_uploaded_file(UploadedBytes("data.csv", EICAR_MARKER), upload_dir=tmp_path)

    assert result.status == "failed"
    assert "Security scan rejected" in result.error_msg


def test_sliding_window_rate_limiter_enforces_budget() -> None:
    limiter = SlidingWindowRateLimiter()
    limiter.check("user-1", "query", 2)
    limiter.check("user-1", "query", 2)

    try:
        limiter.check("user-1", "query", 2)
    except RateLimitExceeded as exc:
        assert "Too many query requests" in str(exc)
    else:
        raise AssertionError("Expected the third request to be rate limited")


def test_upload_permission_is_role_based() -> None:
    assert can_upload("admin") is True
    assert can_upload("analyst") is True
    assert can_upload("viewer") is False
