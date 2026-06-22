from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from src.config import DATABASE_PATH, settings
from src.models.document import DocumentSource
from src.models.user import User


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_dumps(value: Any) -> str:
    return json.dumps(value or {}, ensure_ascii=True, default=str)


def _json_loads(value: Any) -> Dict[str, Any]:
    try:
        loaded = json.loads(value or "{}")
        return loaded if isinstance(loaded, dict) else {}
    except Exception:
        return {}


class SQLiteStore:
    def __init__(self, database_path: Optional[Path] = None) -> None:
        configured_path = database_path or getattr(settings, "database_path", DATABASE_PATH)
        self.database_path = Path(configured_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                PRAGMA foreign_keys = ON;

                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    email TEXT NOT NULL UNIQUE,
                    display_name TEXT NOT NULL,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'user',
                    is_active INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_login_at TEXT,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                );

                CREATE TABLE IF NOT EXISTS documents (
                    source_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    path TEXT NOT NULL,
                    status TEXT NOT NULL,
                    error_msg TEXT,
                    uploaded_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(user_id) REFERENCES users(user_id)
                );

                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    job_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    progress INTEGER NOT NULL DEFAULT 0,
                    source_id TEXT,
                    error_msg TEXT,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(user_id) REFERENCES users(user_id)
                );

                CREATE TABLE IF NOT EXISTS audit_events (
                    event_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    event_type TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );
                """
            )

    def create_user(
        self,
        *,
        email: str,
        display_name: str,
        password_hash: str,
        role: str = "user",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> User:
        now = _utc_now()
        user = User(
            user_id=uuid4().hex,
            email=email,
            display_name=display_name,
            password_hash=password_hash,
            role=role,
            created_at=now,
            updated_at=now,
            metadata=metadata or {},
        )
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO users (
                    user_id, email, display_name, password_hash, role, is_active,
                    created_at, updated_at, last_login_at, metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user.user_id,
                    user.email,
                    user.display_name,
                    user.password_hash,
                    user.role,
                    1 if user.is_active else 0,
                    user.created_at.isoformat(),
                    user.updated_at.isoformat(),
                    None,
                    _json_dumps(user.metadata),
                ),
            )
        return user

    def get_user_by_email(self, email: str) -> Optional[User]:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        return self._row_to_user(row)

    def get_user_by_id(self, user_id: str) -> Optional[User]:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone()
        return self._row_to_user(row)

    def count_users(self) -> int:
        with self._connect() as connection:
            row = connection.execute("SELECT COUNT(*) AS user_count FROM users").fetchone()
        return int(row["user_count"] if row else 0)

    def update_last_login(self, user_id: str) -> None:
        now = _utc_now()
        with self._connect() as connection:
            connection.execute(
                "UPDATE users SET last_login_at = ?, updated_at = ? WHERE user_id = ?",
                (now, now, user_id),
            )

    def upsert_document_source(self, user_id: str, source: DocumentSource) -> None:
        now = _utc_now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO documents (
                    source_id, user_id, filename, file_type, path, status, error_msg,
                    uploaded_at, metadata_json, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_id) DO UPDATE SET
                    filename = excluded.filename,
                    file_type = excluded.file_type,
                    path = excluded.path,
                    status = excluded.status,
                    error_msg = excluded.error_msg,
                    uploaded_at = excluded.uploaded_at,
                    metadata_json = excluded.metadata_json,
                    updated_at = excluded.updated_at
                """,
                (
                    source.source_id,
                    user_id,
                    source.filename,
                    source.file_type,
                    source.path,
                    source.status,
                    source.error_msg,
                    source.uploaded_at.isoformat(),
                    _json_dumps(source.metadata),
                    now,
                    now,
                ),
            )

    def list_document_sources(self, user_id: str) -> List[DocumentSource]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM documents WHERE user_id = ? ORDER BY created_at ASC",
                (user_id,),
            ).fetchall()
        return [self._row_to_document(row) for row in rows]

    def find_document_by_content_hash(self, user_id: str, content_sha256: str) -> Optional[DocumentSource]:
        expected = str(content_sha256 or "").strip().lower()
        if not expected:
            return None
        for source in self.list_document_sources(user_id):
            if str(source.metadata.get("content_sha256") or "").lower() == expected and source.status != "failed":
                return source
        return None

    def fail_interrupted_jobs(self) -> int:
        now = _utc_now()
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE jobs SET status = 'failed', progress = 100, error_msg = ?, updated_at = ? WHERE status IN ('queued', 'processing')",
                ("Worker restarted before this in-process job completed. Please retry ingestion.", now),
            )
        return int(cursor.rowcount or 0)

    def create_job(
        self,
        *,
        user_id: str,
        job_type: str,
        status: str = "queued",
        source_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        now = _utc_now()
        job_id = uuid4().hex
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO jobs (
                    job_id, user_id, job_type, status, progress, source_id,
                    error_msg, metadata_json, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (job_id, user_id, job_type, status, 0, source_id, None, _json_dumps(metadata), now, now),
            )
        return job_id

    def update_job(
        self,
        job_id: str,
        *,
        status: Optional[str] = None,
        progress: Optional[int] = None,
        error_msg: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        fields = []
        values: List[Any] = []
        if status is not None:
            fields.append("status = ?")
            values.append(status)
        if progress is not None:
            fields.append("progress = ?")
            values.append(max(0, min(100, int(progress))))
        if error_msg is not None:
            fields.append("error_msg = ?")
            values.append(error_msg)
        if metadata is not None:
            fields.append("metadata_json = ?")
            values.append(_json_dumps(metadata))
        if not fields:
            return
        fields.append("updated_at = ?")
        values.append(_utc_now())
        values.append(job_id)
        with self._connect() as connection:
            connection.execute("UPDATE jobs SET {0} WHERE job_id = ?".format(", ".join(fields)), values)

    def record_audit_event(self, event_type: str, metadata: Optional[Dict[str, Any]] = None, user_id: Optional[str] = None) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO audit_events (event_id, user_id, event_type, metadata_json, created_at) VALUES (?, ?, ?, ?, ?)",
                (uuid4().hex, user_id, event_type, _json_dumps(metadata), _utc_now()),
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=10.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = NORMAL")
        connection.execute("PRAGMA busy_timeout = 10000")
        return connection

    def _row_to_user(self, row: Optional[sqlite3.Row]) -> Optional[User]:
        if row is None:
            return None
        payload = dict(row)
        return User(
            user_id=payload["user_id"],
            email=payload["email"],
            display_name=payload["display_name"],
            password_hash=payload["password_hash"],
            role=payload["role"],
            is_active=bool(payload["is_active"]),
            created_at=payload["created_at"],
            updated_at=payload["updated_at"],
            last_login_at=payload.get("last_login_at"),
            metadata=_json_loads(payload.get("metadata_json")),
        )

    def _row_to_document(self, row: sqlite3.Row) -> DocumentSource:
        payload = dict(row)
        return DocumentSource(
            source_id=payload["source_id"],
            filename=payload["filename"],
            file_type=payload["file_type"],
            path=payload["path"],
            status=payload["status"],
            error_msg=payload["error_msg"],
            uploaded_at=payload["uploaded_at"],
            metadata=_json_loads(payload.get("metadata_json")),
        )
