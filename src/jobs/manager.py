from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from threading import RLock
from typing import Any, Callable, Dict, List, Optional

from src.config import settings
from src.jobs.models import JobState
from src.storage.sqlite_store import SQLiteStore
from src.utils.logging import log_error, log_event


class JobCapacityExceeded(RuntimeError):
    pass


class BackgroundJobManager:
    def __init__(self, store: Optional[SQLiteStore] = None, max_workers: Optional[int] = None) -> None:
        self.store = store or SQLiteStore()
        recovered_count = self.store.fail_interrupted_jobs()
        if recovered_count:
            log_event("interrupted_jobs_marked_failed", {"job_count": recovered_count})
        self.executor = ThreadPoolExecutor(max_workers=max_workers or settings.background_worker_count)
        self._lock = RLock()
        self._states: Dict[str, JobState] = {}
        self._futures: Dict[str, Future] = {}

    def submit(
        self,
        *,
        user_id: str,
        job_type: str,
        handler: Callable[[], Any],
        source_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        with self._lock:
            active_count = len([future for future in self._futures.values() if not future.done()])
            if active_count >= settings.background_max_pending_jobs:
                raise JobCapacityExceeded("Background ingestion capacity is full. Please wait for active jobs to finish.")
        job_id = self.store.create_job(
            user_id=user_id,
            job_type=job_type,
            status="queued",
            source_id=source_id,
            metadata=metadata or {},
        )
        state = JobState(
            job_id=job_id,
            user_id=user_id,
            job_type=job_type,
            status="queued",
            progress=0,
            source_id=source_id,
            metadata=metadata or {},
        )
        with self._lock:
            self._states[job_id] = state
            future = self.executor.submit(self._execute, job_id, handler)
            self._futures[job_id] = future
            future.add_done_callback(lambda completed, completed_job_id=job_id: self._forget_future(completed_job_id))
        log_event("background_job_submitted", state.public_dict())
        return job_id

    def get_job(self, job_id: str) -> Optional[JobState]:
        with self._lock:
            state = self._states.get(job_id)
            return _copy_state(state)

    def list_user_jobs(self, user_id: str) -> List[JobState]:
        with self._lock:
            return [
                _copy_state(state)
                for state in self._states.values()
                if state.user_id == user_id
            ]

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            future = self._futures.get(job_id)
            if future is None:
                return False
            canceled = future.cancel()
            if canceled:
                self._set_state(job_id, status="canceled", progress=100)
                self.store.update_job(job_id, status="canceled", progress=100)
                log_event("background_job_canceled", {"job_id": job_id})
            return canceled

    def _execute(self, job_id: str, handler: Callable[[], Any]) -> None:
        self._set_state(job_id, status="processing", progress=10)
        self.store.update_job(job_id, status="processing", progress=10)
        try:
            result = handler()
            self._set_state(job_id, status="completed", progress=100, result=result)
            self.store.update_job(job_id, status="completed", progress=100)
            log_event("background_job_completed", {"job_id": job_id})
        except Exception as exc:
            self._set_state(job_id, status="failed", progress=100, error_msg=str(exc))
            self.store.update_job(job_id, status="failed", progress=100, error_msg=str(exc))
            log_error(exc, {"stage": "background_job", "job_id": job_id})

    def _set_state(
        self,
        job_id: str,
        *,
        status: Optional[str] = None,
        progress: Optional[int] = None,
        result: Any = None,
        error_msg: Optional[str] = None,
    ) -> None:
        with self._lock:
            state = self._states.get(job_id)
            if state is None:
                return
            if status is not None:
                state.status = status
            if progress is not None:
                state.progress = max(0, min(100, int(progress)))
            if result is not None:
                state.result = result
            if error_msg is not None:
                state.error_msg = error_msg
            state.updated_at = datetime.now(timezone.utc)

    def _forget_future(self, job_id: str) -> None:
        with self._lock:
            self._futures.pop(job_id, None)


def _copy_state(state: Optional[JobState]) -> Optional[JobState]:
    if state is None:
        return None
    if hasattr(state, "model_copy"):
        return state.model_copy(deep=False)
    return state.copy(deep=False)
