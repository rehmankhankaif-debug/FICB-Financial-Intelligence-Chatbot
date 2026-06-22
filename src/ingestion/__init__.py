"""Structured file and document loading utilities."""

from src.ingestion.document_loader import load_document
from src.ingestion.docx_loader import load_docx
from src.ingestion.file_loader import load_csv, load_excel
from src.ingestion.pdf_loader import load_pdf
from src.ingestion.table_loader import load_table
from src.ingestion.url_loader import load_url

__all__ = ["load_csv", "load_docx", "load_document", "load_excel", "load_pdf", "load_table", "load_url"]
