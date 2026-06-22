"""Application services that keep Streamlit thin and backend logic reusable."""

from src.services.ingestion_service import IngestionResult, IngestionService

__all__ = ["IngestionResult", "IngestionService"]
