from __future__ import annotations

from src.models.document import DocumentSource
import src.storage.sqlite_store as sqlite_store_module
from src.storage.sqlite_store import SQLiteStore


def test_sqlite_store_persists_user_documents_jobs_and_audit_events(tmp_path) -> None:
    store = SQLiteStore(tmp_path / "app.sqlite3")
    user = store.create_user(
        email="kaif@example.com",
        display_name="Kaif",
        password_hash="hash",
    )
    source = DocumentSource(
        source_id="source_1",
        filename="report.pdf",
        file_type="pdf",
        path="/tmp/report.pdf",
        status="uploaded",
        metadata={"chunk_count": 3},
    )

    store.upsert_document_source(user.user_id, source)
    job_id = store.create_job(user_id=user.user_id, job_type="ingest_document", source_id=source.source_id)
    store.update_job(job_id, status="completed", progress=100, metadata={"indexed_chunks": 3})
    store.record_audit_event("test.event", {"ok": True}, user_id=user.user_id)

    sources = store.list_document_sources(user.user_id)

    assert store.count_users() == 1
    assert sources[0].source_id == "source_1"
    assert sources[0].metadata["chunk_count"] == 3
    assert store.get_user_by_email("kaif@example.com").user_id == user.user_id


def test_sqlite_store_handles_stale_settings_without_database_path(monkeypatch, tmp_path) -> None:
    fallback_path = tmp_path / "fallback.sqlite3"
    monkeypatch.setattr(sqlite_store_module, "DATABASE_PATH", fallback_path)
    monkeypatch.setattr(sqlite_store_module, "settings", object())

    store = SQLiteStore()

    assert store.database_path == fallback_path
    assert fallback_path.exists()
