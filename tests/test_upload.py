from __future__ import annotations

from io import BytesIO

from src.utils.upload import save_uploaded_file


class DummyUploadedFile(BytesIO):
    def __init__(self, name: str, content: bytes) -> None:
        super().__init__(content)
        self.name = name
        self.size = len(content)

    def getbuffer(self):
        return super().getbuffer()


def test_save_uploaded_file_persists_supported_file(tmp_path) -> None:
    uploaded = DummyUploadedFile("sales.csv", b"month,profit\nJan,100\n")
    source = save_uploaded_file(uploaded, upload_dir=tmp_path)

    assert source.status == "uploaded"
    assert source.filename == "sales.csv"
    assert source.file_type == "csv"
    assert source.path
    assert source.metadata["size_bytes"] == len(b"month,profit\nJan,100\n")


def test_save_uploaded_file_rejects_unsupported_file(tmp_path) -> None:
    uploaded = DummyUploadedFile("script.exe", b"bad")
    source = save_uploaded_file(uploaded, upload_dir=tmp_path)

    assert source.status == "failed"
    assert source.error_msg
    assert "Unsupported" in source.error_msg


def test_save_uploaded_file_rejects_spoofed_pdf(tmp_path) -> None:
    uploaded = DummyUploadedFile("report.pdf", b"not really a pdf")
    source = save_uploaded_file(uploaded, upload_dir=tmp_path)

    assert source.status == "failed"
    assert source.error_msg
    assert "does not match" in source.error_msg
    assert not list(tmp_path.iterdir())
