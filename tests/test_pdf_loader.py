from __future__ import annotations

from pathlib import Path

import pytest

from src.ingestion.pdf_loader import PageExtractionTimeout, _extract_page_tables, load_pdf
from src.utils.errors import IngestionError, UnsupportedFileError


def write_minimal_pdf(path: Path, text: str) -> None:
    escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    objects = [
        "1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n",
        "2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n",
        "3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>\nendobj\n",
        "4 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n",
    ]
    stream = "BT\n/F1 12 Tf\n72 720 Td\n({0}) Tj\nET\n".format(escaped)
    objects.append("5 0 obj\n<< /Length {0} >>\nstream\n{1}endstream\nendobj\n".format(len(stream.encode()), stream))

    body = "%PDF-1.4\n"
    offsets = []
    for item in objects:
        offsets.append(len(body.encode()))
        body += item
    xref_offset = len(body.encode())
    xref = "xref\n0 6\n0000000000 65535 f \n"
    for offset in offsets:
        xref += "{0:010d} 00000 n \n".format(offset)
    trailer = "trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n{0}\n%%EOF\n".format(xref_offset)
    path.write_bytes((body + xref + trailer).encode("latin-1"))


def test_load_pdf_extracts_text_and_page_metadata(tmp_path) -> None:
    path = tmp_path / "report.pdf"
    write_minimal_pdf(path, "Financial report revenue profit growth")

    sources = load_pdf(path, source_id="pdf_source")

    assert len(sources) == 1
    assert sources[0].source_id == "pdf_source"
    assert sources[0].source_type == "pdf"
    assert sources[0].filename == "report.pdf"
    assert sources[0].metadata["page"] == 1
    assert "table_count" in sources[0].metadata
    assert "possible_chart_or_visual" in sources[0].metadata
    assert "ocr_attempted" in sources[0].metadata
    assert "revenue profit" in sources[0].content


def test_pdf_table_extraction_serializes_rows_for_rag() -> None:
    class FakeTable:
        def extract(self):
            return [["Metric", "2023", "2022"], ["Revenue", "120", "100"]]

    class FakeFinder:
        tables = [FakeTable()]

    class FakePage:
        def find_tables(self):
            return FakeFinder()

    payload = _extract_page_tables(FakePage())

    assert payload["table_count"] == 1
    assert "Revenue\t120\t100" in payload["table_text"]


def test_load_pdf_rejects_wrong_extension(tmp_path) -> None:
    path = tmp_path / "report.txt"
    path.write_text("not a pdf", encoding="utf-8")

    with pytest.raises(UnsupportedFileError):
        load_pdf(path)


def test_load_pdf_handles_corrupted_pdf(tmp_path) -> None:
    path = tmp_path / "broken.pdf"
    path.write_bytes(b"not a real pdf")

    with pytest.raises(IngestionError):
        load_pdf(path)


def test_load_pdf_records_page_error_when_pypdf_page_times_out(tmp_path, monkeypatch) -> None:
    path = tmp_path / "slow.pdf"
    write_minimal_pdf(path, "Fallback path")

    class FakePage:
        def extract_text(self):
            raise PageExtractionTimeout("timed out")

    class FakeReader:
        metadata = {}
        pages = [FakePage()]

    monkeypatch.setattr("src.ingestion.pdf_loader._load_pdf_with_pymupdf", lambda file_path, source_id: None)
    monkeypatch.setattr("src.ingestion.pdf_loader.PdfReader", lambda file_path: FakeReader())

    sources = load_pdf(path, source_id="slow_pdf")

    assert len(sources) == 1
    assert sources[0].content == ""
    assert sources[0].metadata["extraction_backend"] == "pypdf"
    assert "timed out" in sources[0].metadata["page_error"]
