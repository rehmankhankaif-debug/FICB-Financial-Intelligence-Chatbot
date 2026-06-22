from __future__ import annotations

import time
from threading import Event

import pytest

from src.config import settings
from src.jobs.manager import BackgroundJobManager, JobCapacityExceeded
from src.storage.sqlite_store import SQLiteStore


def _wait_for_status(manager: BackgroundJobManager, job_id: str, expected: str, timeout: float = 3.0):
    started = time.monotonic()
    while time.monotonic() - started < timeout:
        state = manager.get_job(job_id)
        if state and state.status == expected:
            return state
        time.sleep(0.02)
    return manager.get_job(job_id)


def test_background_job_manager_completes_job(tmp_path) -> None:
    store = SQLiteStore(tmp_path / "app.sqlite3")
    user = store.create_user(email="kaif@example.com", display_name="Kaif", password_hash="hash")
    manager = BackgroundJobManager(store, max_workers=1)

    job_id = manager.submit(user_id=user.user_id, job_type="demo", handler=lambda: {"ok": True})
    state = _wait_for_status(manager, job_id, "completed")

    assert state.status == "completed"
    assert state.progress == 100
    assert state.result == {"ok": True}


def test_background_job_manager_marks_failures(tmp_path) -> None:
    store = SQLiteStore(tmp_path / "app.sqlite3")
    user = store.create_user(email="kaif@example.com", display_name="Kaif", password_hash="hash")
    manager = BackgroundJobManager(store, max_workers=1)

    def fail():
        raise RuntimeError("boom")

    job_id = manager.submit(user_id=user.user_id, job_type="demo", handler=fail)
    state = _wait_for_status(manager, job_id, "failed")

    assert state.status == "failed"
    assert state.progress == 100
    assert "boom" in state.error_msg


def test_background_job_manager_rejects_over_capacity(tmp_path, monkeypatch) -> None:
    store = SQLiteStore(tmp_path / "app.sqlite3")
    user = store.create_user(email="kaif@example.com", display_name="Kaif", password_hash="hash")
    manager = BackgroundJobManager(store, max_workers=1)
    release = Event()
    monkeypatch.setattr(settings, "background_max_pending_jobs", 1)

    manager.submit(user_id=user.user_id, job_type="slow", handler=lambda: release.wait(2))
    with pytest.raises(JobCapacityExceeded):
        manager.submit(user_id=user.user_id, job_type="overflow", handler=lambda: None)
    release.set()
