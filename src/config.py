from __future__ import annotations

import os
import signal
import threading
from pathlib import Path
from typing import List, Optional, Set

from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
CHROMA_DIR = DATA_DIR / "chroma"
HISTORY_DIR = DATA_DIR / "history"
LOGS_DIR = DATA_DIR / "logs"
DATABASE_PATH = DATA_DIR / "app.sqlite3"

ALLOWED_FILE_EXTENSIONS = {"csv", "xlsx", "xls", "pdf", "docx", "txt", "html"}
MAX_FILE_SIZE_MB = 25
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_ENVIRONMENT = "local"
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_RETRY_MAX_ATTEMPTS = 3
DEFAULT_RETRY_BACKOFF_SECONDS = 0.25
DEFAULT_CIRCUIT_BREAKER_FAILURE_THRESHOLD = 3
DEFAULT_CIRCUIT_BREAKER_RECOVERY_SECONDS = 30.0
DEFAULT_URL_TIMEOUT_SECONDS = 10.0
DEFAULT_URL_RETRY_MAX_ATTEMPTS = 2
DEFAULT_BLOCK_PRIVATE_URLS = True
DEFAULT_BACKGROUND_WORKER_COUNT = 2
DEFAULT_SUPPORTED_LANGUAGES = ["en", "hi-en", "es"]
DEFAULT_QUERY_RATE_LIMIT_PER_MINUTE = 30
DEFAULT_UPLOAD_RATE_LIMIT_PER_MINUTE = 10
DEFAULT_BACKGROUND_MAX_PENDING_JOBS = 20
MAX_ENV_FILE_BYTES = 32 * 1024
ENV_LOAD_TIMEOUT_SECONDS = 1


class AppConfig(BaseModel):
    project_root: Path = PROJECT_ROOT
    data_dir: Path = DATA_DIR
    upload_dir: Path = UPLOAD_DIR
    chroma_dir: Path = CHROMA_DIR
    history_dir: Path = HISTORY_DIR
    logs_dir: Path = LOGS_DIR
    database_path: Path = DATABASE_PATH
    allowed_file_extensions: Set[str] = Field(default_factory=lambda: set(ALLOWED_FILE_EXTENSIONS))
    max_file_size_mb: int = MAX_FILE_SIZE_MB
    max_file_size_bytes: int = MAX_FILE_SIZE_BYTES
    default_embedding_model: str = DEFAULT_EMBEDDING_MODEL
    default_gemini_model: str = DEFAULT_GEMINI_MODEL
    environment: str = DEFAULT_ENVIRONMENT
    log_level: str = DEFAULT_LOG_LEVEL
    retry_max_attempts: int = DEFAULT_RETRY_MAX_ATTEMPTS
    retry_backoff_seconds: float = DEFAULT_RETRY_BACKOFF_SECONDS
    circuit_breaker_failure_threshold: int = DEFAULT_CIRCUIT_BREAKER_FAILURE_THRESHOLD
    circuit_breaker_recovery_seconds: float = DEFAULT_CIRCUIT_BREAKER_RECOVERY_SECONDS
    url_timeout_seconds: float = DEFAULT_URL_TIMEOUT_SECONDS
    url_retry_max_attempts: int = DEFAULT_URL_RETRY_MAX_ATTEMPTS
    block_private_urls: bool = DEFAULT_BLOCK_PRIVATE_URLS
    background_worker_count: int = DEFAULT_BACKGROUND_WORKER_COUNT
    supported_languages: List[str] = Field(default_factory=lambda: list(DEFAULT_SUPPORTED_LANGUAGES))
    query_rate_limit_per_minute: int = DEFAULT_QUERY_RATE_LIMIT_PER_MINUTE
    upload_rate_limit_per_minute: int = DEFAULT_UPLOAD_RATE_LIMIT_PER_MINUTE
    background_max_pending_jobs: int = DEFAULT_BACKGROUND_MAX_PENDING_JOBS
    gemini_api_key: Optional[str] = None
    gemini_api_key_available: bool = False


def ensure_directories() -> None:
    for directory in (DATA_DIR, UPLOAD_DIR, CHROMA_DIR, HISTORY_DIR, LOGS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_list(name: str, default: List[str]) -> List[str]:
    values = [item.strip().lower() for item in os.getenv(name, "").split(",") if item.strip()]
    return list(dict.fromkeys(values or default))


def _configured_secret(value: Optional[str]) -> Optional[str]:
    cleaned = str(value or "").strip()
    normalized = cleaned.lower()
    if not cleaned or normalized.startswith("your_") or normalized in {"replace_me", "changeme"}:
        return None
    return cleaned


def _safe_load_env_file(path: Path) -> bool:
    try:
        env_path = Path(path)
        if not env_path.exists() or not env_path.is_file():
            return False
        if env_path.stat().st_size > MAX_ENV_FILE_BYTES:
            return False
        return _read_env_with_timeout(env_path)
    except Exception:
        return False


def _read_env_with_timeout(path: Path) -> bool:
    def _timeout_handler(signum, frame):
        raise TimeoutError("Timed out while reading .env")

    previous_handler = None
    use_signal_timeout = hasattr(signal, "SIGALRM") and threading.current_thread() is threading.main_thread()
    try:
        if use_signal_timeout:
            previous_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(ENV_LOAD_TIMEOUT_SECONDS)
        text = path.read_text(encoding="utf-8", errors="replace")
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("\"'")
            # Let a real process-level value win, but do not let an inherited
            # empty/placeholder secret suppress a valid local .env setting.
            existing_value = os.environ.get(key)
            inherited_value_is_usable = bool(existing_value)
            if key == "GEMINI_API_KEY":
                inherited_value_is_usable = _configured_secret(existing_value) is not None
            if key and not inherited_value_is_usable:
                os.environ[key] = value
        return True
    finally:
        if use_signal_timeout:
            signal.alarm(0)
            if previous_handler is not None:
                signal.signal(signal.SIGALRM, previous_handler)


def load_config(load_env_file: bool = True) -> AppConfig:
    if load_env_file:
        _safe_load_env_file(PROJECT_ROOT / ".env")

    ensure_directories()
    gemini_api_key = _configured_secret(os.getenv("GEMINI_API_KEY"))
    gemini_model = (os.getenv("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL).strip() or DEFAULT_GEMINI_MODEL

    return AppConfig(
        default_gemini_model=gemini_model,
        database_path=Path(os.getenv("APP_DATABASE_PATH", str(DATABASE_PATH))),
        environment=os.getenv("APP_ENV", DEFAULT_ENVIRONMENT),
        log_level=os.getenv("LOG_LEVEL", DEFAULT_LOG_LEVEL).upper(),
        retry_max_attempts=max(1, _env_int("RETRY_MAX_ATTEMPTS", DEFAULT_RETRY_MAX_ATTEMPTS)),
        retry_backoff_seconds=max(0.0, _env_float("RETRY_BACKOFF_SECONDS", DEFAULT_RETRY_BACKOFF_SECONDS)),
        circuit_breaker_failure_threshold=max(
            1,
            _env_int("CIRCUIT_BREAKER_FAILURE_THRESHOLD", DEFAULT_CIRCUIT_BREAKER_FAILURE_THRESHOLD),
        ),
        circuit_breaker_recovery_seconds=max(
            0.0,
            _env_float("CIRCUIT_BREAKER_RECOVERY_SECONDS", DEFAULT_CIRCUIT_BREAKER_RECOVERY_SECONDS),
        ),
        url_timeout_seconds=max(0.1, _env_float("URL_TIMEOUT_SECONDS", DEFAULT_URL_TIMEOUT_SECONDS)),
        url_retry_max_attempts=max(1, _env_int("URL_RETRY_MAX_ATTEMPTS", DEFAULT_URL_RETRY_MAX_ATTEMPTS)),
        block_private_urls=_env_bool("BLOCK_PRIVATE_URLS", DEFAULT_BLOCK_PRIVATE_URLS),
        background_worker_count=max(1, _env_int("BACKGROUND_WORKER_COUNT", DEFAULT_BACKGROUND_WORKER_COUNT)),
        supported_languages=_env_list("SUPPORTED_LANGUAGES", DEFAULT_SUPPORTED_LANGUAGES),
        query_rate_limit_per_minute=max(1, _env_int("QUERY_RATE_LIMIT_PER_MINUTE", DEFAULT_QUERY_RATE_LIMIT_PER_MINUTE)),
        upload_rate_limit_per_minute=max(1, _env_int("UPLOAD_RATE_LIMIT_PER_MINUTE", DEFAULT_UPLOAD_RATE_LIMIT_PER_MINUTE)),
        background_max_pending_jobs=max(1, _env_int("BACKGROUND_MAX_PENDING_JOBS", DEFAULT_BACKGROUND_MAX_PENDING_JOBS)),
        gemini_api_key=gemini_api_key,
        gemini_api_key_available=bool(gemini_api_key),
    )


settings = load_config()
