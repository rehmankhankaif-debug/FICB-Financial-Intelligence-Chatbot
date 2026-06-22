from __future__ import annotations

import os
import signal
import threading
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pypdf import PdfReader

from src.models.document import DocumentChunkSource
from src.utils.errors import IngestionError, UnsupportedFileError
from src.utils.security import get_file_type

DEFAULT_PAGE_EXTRACTION_TIMEOUT_SECONDS = 1.0
DEFAULT_PDF_OCR_ENABLED = True
DEFAULT_PDF_OCR_MIN_TEXT_CHARACTERS = 40


class PageExtractionTimeout(TimeoutError):
    pass


def _validate_pdf_path(path: Any) -> Path:
    file_path = Path(path)
    if not file_path.exists():
        raise IngestionError("PDF file does not exist.", metadata={"path": str(file_path)})
    if not file_path.is_file():
        raise IngestionError("PDF path is not a file.", metadata={"path": str(file_path)})
    if get_file_type(str(file_path)) != "pdf":
        raise UnsupportedFileError(
            "PDF loader only supports .pdf files.",
            metadata={"path": str(file_path), "file_type": get_file_type(str(file_path))},
        )
    return file_path


def _page_timeout_seconds() -> float:
    raw_value = os.getenv("PDF_PAGE_EXTRACTION_TIMEOUT_SECONDS", str(DEFAULT_PAGE_EXTRACTION_TIMEOUT_SECONDS))
    try:
        return max(0.0, float(raw_value))
    except ValueError:
        return DEFAULT_PAGE_EXTRACTION_TIMEOUT_SECONDS


def _ocr_enabled() -> bool:
    return os.getenv("PDF_OCR_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}


def _ocr_minimum_text_characters() -> int:
    try:
        return max(0, int(os.getenv("PDF_OCR_MIN_TEXT_CHARACTERS", str(DEFAULT_PDF_OCR_MIN_TEXT_CHARACTERS))))
    except ValueError:
        return DEFAULT_PDF_OCR_MIN_TEXT_CHARACTERS


def _extract_page_tables(page: Any) -> Dict[str, Any]:
    tables_text: List[str] = []
    try:
        finder = page.find_tables()
        tables = list(getattr(finder, "tables", []) or [])
        for table_index, table in enumerate(tables, start=1):
            rows = table.extract() or []
            normalized_rows = [
                [str(cell or "").replace("\n", " ").strip() for cell in row]
                for row in rows if row
            ]
            if normalized_rows:
                rendered = "\n".join("\t".join(row) for row in normalized_rows)
                tables_text.append("[Extracted table {0}]\n{1}".format(table_index, rendered))
        return {"table_count": len(tables_text), "table_text": "\n\n".join(tables_text)}
    except Exception as exc:
        return {"table_count": 0, "table_text": "", "table_extraction_error": str(exc)}


def _page_visual_metadata(page: Any) -> Dict[str, Any]:
    try:
        image_count = len(page.get_images(full=True) or [])
    except Exception:
        image_count = 0
    try:
        drawing_count = len(page.get_drawings() or [])
    except Exception:
        drawing_count = 0
    return {
        "image_count": image_count,
        "drawing_count": drawing_count,
        "possible_chart_or_visual": bool(image_count or drawing_count >= 4),
    }


def _ocr_page(page: Any) -> Dict[str, Any]:
    if not _ocr_enabled():
        return {"ocr_attempted": False, "ocr_used": False, "ocr_text": ""}
    try:
        import pytesseract
        from PIL import Image

        pixmap = page.get_pixmap(dpi=200, alpha=False)
        image = Image.open(BytesIO(pixmap.tobytes("png")))
        text = pytesseract.image_to_string(image) or ""
        return {"ocr_attempted": True, "ocr_used": bool(text.strip()), "ocr_text": text.strip()}
    except Exception as exc:
        return {"ocr_attempted": True, "ocr_used": False, "ocr_text": "", "ocr_error": str(exc)}


def _build_chunk_source(
    file_path: Path,
    source_id: str,
    page_number: int,
    page_count: int,
    text: str,
    metadata: Dict[str, Any],
    page_error: Optional[str] = None,
) -> DocumentChunkSource:
    page_metadata: Dict[str, Any] = {
        "page": page_number,
        "page_count": page_count,
        "source_path": str(file_path),
        **metadata,
    }
    if page_error:
        page_metadata["page_error"] = page_error

    return DocumentChunkSource(
        source_id=source_id,
        filename=file_path.name,
        source_type="pdf",
        content=(text or "").strip(),
        metadata=page_metadata,
    )


def _reader_metadata(reader: PdfReader) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}
    try:
        raw_metadata = reader.metadata or {}
        for key, value in raw_metadata.items():
            metadata[str(key).lstrip("/")] = str(value)
    except Exception:
        metadata["metadata_error"] = "Unable to read PDF metadata."
    return metadata


def _load_pdf_with_pymupdf(file_path: Path, source_id: str) -> Optional[List[DocumentChunkSource]]:
    try:
        import fitz  # PyMuPDF
    except Exception:
        return None

    try:
        document = fitz.open(str(file_path))
    except Exception as exc:
        raise IngestionError(
            "Failed to load PDF document.",
            metadata={"path": str(file_path), "error": str(exc), "backend": "pymupdf"},
        )

    try:
        pdf_metadata = {
            key: str(value)
            for key, value in (document.metadata or {}).items()
            if value is not None
        }
        page_count = int(document.page_count)
        sources: List[DocumentChunkSource] = []

        for index, page in enumerate(document, start=1):
            try:
                text = page.get_text("text") or ""
                page_error = None
            except Exception as exc:
                text = ""
                page_error = str(exc)

            table_payload = _extract_page_tables(page)
            visual_payload = _page_visual_metadata(page)
            ocr_payload: Dict[str, Any] = {"ocr_attempted": False, "ocr_used": False, "ocr_text": ""}
            if len(text.strip()) < _ocr_minimum_text_characters():
                ocr_payload = _ocr_page(page)
                if ocr_payload.get("ocr_text"):
                    text = "{0}\n{1}".format(text.strip(), ocr_payload["ocr_text"]).strip()
            if table_payload.get("table_text"):
                text = "{0}\n\n{1}".format(text.strip(), table_payload["table_text"]).strip()

            extraction_metadata = {
                "pdf_metadata": pdf_metadata,
                "extraction_backend": "pymupdf",
                "table_count": table_payload.get("table_count", 0),
                **visual_payload,
                "ocr_attempted": ocr_payload.get("ocr_attempted", False),
                "ocr_used": ocr_payload.get("ocr_used", False),
            }
            if table_payload.get("table_extraction_error"):
                extraction_metadata["table_extraction_error"] = table_payload["table_extraction_error"]
            if ocr_payload.get("ocr_error"):
                extraction_metadata["ocr_error"] = ocr_payload["ocr_error"]

            sources.append(
                _build_chunk_source(
                    file_path=file_path,
                    source_id=source_id,
                    page_number=index,
                    page_count=page_count,
                    text=text,
                    metadata=extraction_metadata,
                    page_error=page_error,
                )
            )

        if not sources:
            raise IngestionError("PDF contains no pages.", metadata={"path": str(file_path), "backend": "pymupdf"})
        return sources
    finally:
        document.close()


def _extract_text_with_timeout(page: Any, timeout_seconds: float) -> str:
    if (
        timeout_seconds <= 0
        or not hasattr(signal, "SIGALRM")
        or threading.current_thread() is not threading.main_thread()
    ):
        return page.extract_text() or ""

    previous_handler = signal.getsignal(signal.SIGALRM)

    def _handle_timeout(signum: int, frame: Any) -> None:
        raise PageExtractionTimeout("PDF page text extraction timed out after {0:g}s.".format(timeout_seconds))

    try:
        signal.signal(signal.SIGALRM, _handle_timeout)
        signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
        return page.extract_text() or ""
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def _load_pdf_with_pypdf(file_path: Path, source_id: str) -> List[DocumentChunkSource]:
    try:
        reader = PdfReader(str(file_path))
        pdf_metadata = _reader_metadata(reader)
        sources: List[DocumentChunkSource] = []
        page_count = len(reader.pages)
        timeout_seconds = _page_timeout_seconds()

        for index, page in enumerate(reader.pages, start=1):
            try:
                text = _extract_text_with_timeout(page, timeout_seconds)
            except Exception as exc:
                text = ""
                page_error = str(exc)
            else:
                page_error = None

            sources.append(
                _build_chunk_source(
                    file_path=file_path,
                    source_id=source_id,
                    page_number=index,
                    page_count=page_count,
                    text=text,
                    metadata={"pdf_metadata": pdf_metadata, "extraction_backend": "pypdf"},
                    page_error=page_error,
                )
            )

        if not sources:
            raise IngestionError("PDF contains no pages.", metadata={"path": str(file_path)})
        return sources
    except IngestionError:
        raise
    except Exception as exc:
        raise IngestionError(
            "Failed to load PDF document.",
            metadata={"path": str(file_path), "error": str(exc)},
        )


def load_pdf(path: Any, source_id: Optional[str] = None) -> List[DocumentChunkSource]:
    file_path = _validate_pdf_path(path)
    source_id = source_id or uuid4().hex

    pymupdf_sources = _load_pdf_with_pymupdf(file_path, source_id)
    if pymupdf_sources is not None:
        return pymupdf_sources

    return _load_pdf_with_pypdf(file_path, source_id)
