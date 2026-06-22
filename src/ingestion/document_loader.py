from __future__ import annotations

from pathlib import Path
from typing import Any, List

from src.ingestion.docx_loader import load_docx
from src.ingestion.pdf_loader import load_pdf
from src.ingestion.url_loader import load_url
from src.models.document import DocumentChunkSource
from src.utils.errors import UnsupportedFileError
from src.utils.security import get_file_type


def load_document(source: Any) -> List[DocumentChunkSource]:
    source_text = str(source)
    if source_text.startswith("http://") or source_text.startswith("https://"):
        return load_url(source_text)

    file_type = get_file_type(str(Path(source_text)))
    if file_type == "pdf":
        return load_pdf(source)
    if file_type == "docx":
        return load_docx(source)

    raise UnsupportedFileError(
        "Document loader supports only PDF, DOCX, and URLs in Phase 3.",
        metadata={"source": source_text, "file_type": file_type},
    )
