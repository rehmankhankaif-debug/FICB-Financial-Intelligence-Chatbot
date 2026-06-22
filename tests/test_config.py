from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from src import config


def test_config_imports_successfully() -> None:
    assert config.settings.project_root.exists()


def test_required_directories_exist() -> None:
    assert config.settings.upload_dir.exists()
    assert config.settings.chroma_dir.exists()
    assert config.settings.history_dir.exists()
    assert config.settings.logs_dir.exists()


def test_allowed_extensions_are_configured() -> None:
    assert {"csv", "xlsx", "xls", "pdf", "docx", "txt", "html"}.issubset(
        config.settings.allowed_file_extensions
    )


def test_max_file_size_is_configured() -> None:
    assert config.settings.max_file_size_bytes > 0
    assert config.settings.max_file_size_mb > 0


def test_missing_gemini_key_does_not_crash(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    loaded = config.load_config(load_env_file=False)
    assert loaded.gemini_api_key is None
    assert loaded.gemini_api_key_available is False


def test_placeholder_gemini_key_is_treated_as_unavailable(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "your_gemini_api_key_here")
    loaded = config.load_config(load_env_file=False)

    assert loaded.gemini_api_key is None
    assert loaded.gemini_api_key_available is False


def test_gemini_model_can_be_configured_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-flash")
    loaded = config.load_config(load_env_file=False)

    assert loaded.default_gemini_model == "gemini-2.5-flash"


def test_env_file_replaces_inherited_empty_value(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "")
    env_file = tmp_path / ".env"
    env_file.write_text("GEMINI_API_KEY=configured-key\n", encoding="utf-8")

    assert config._safe_load_env_file(env_file) is True
    assert config.os.environ["GEMINI_API_KEY"] == "configured-key"


def test_env_file_replaces_inherited_placeholder_key(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "your_gemini_api_key_here")
    env_file = tmp_path / ".env"
    env_file.write_text("GEMINI_API_KEY=configured-key\n", encoding="utf-8")

    assert config._safe_load_env_file(env_file) is True
    assert config.os.environ["GEMINI_API_KEY"] == "configured-key"


def test_env_file_loads_from_streamlit_style_worker_thread(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("GEMINI_API_KEY=thread-loaded-key\n", encoding="utf-8")

    with ThreadPoolExecutor(max_workers=1) as executor:
        loaded = executor.submit(config._safe_load_env_file, env_file).result()

    assert loaded is True
    assert config.os.environ["GEMINI_API_KEY"] == "thread-loaded-key"


def test_production_knobs_have_safe_defaults() -> None:
    assert config.settings.retry_max_attempts >= 1
    assert config.settings.retry_backoff_seconds >= 0
    assert config.settings.circuit_breaker_failure_threshold >= 1
    assert config.settings.url_timeout_seconds > 0
    assert config.settings.url_retry_max_attempts >= 1
    assert config.settings.log_level in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
