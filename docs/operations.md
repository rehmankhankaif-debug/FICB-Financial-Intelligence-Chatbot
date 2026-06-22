# Operations Guide

## Local Run

```bash
source venv/bin/activate
streamlit run app.py
```

The app starts with a login/register screen. User accounts are stored in SQLite at `APP_DATABASE_PATH`, defaulting to `data/app.sqlite3`.

## Docker Run

```bash
cp .env.example .env
docker compose up --build
```

Open:

```text
http://localhost:8501
```

Runtime data is mounted through `./data:/app/data`, so uploads, ChromaDB, logs, history, and SQLite data survive container restarts.

## Configuration

Important environment variables:

- `GEMINI_API_KEY`: optional Gemini key for planning/narration.
- `APP_ENV`: `local`, `ci`, `container`, or your deployment name.
- `APP_DATABASE_PATH`: SQLite database path.
- `LOG_LEVEL`: structured log level.
- `BACKGROUND_WORKER_COUNT`: local ingestion worker count.
- `RETRY_MAX_ATTEMPTS`: retry attempts for supported external calls.
- `CIRCUIT_BREAKER_FAILURE_THRESHOLD`: Gemini circuit-breaker threshold.
- `URL_TIMEOUT_SECONDS`: URL ingestion timeout.
- `BLOCK_PRIVATE_URLS`: blocks private/local URL fetches when true.

## Health Check

Docker uses Streamlit's health endpoint:

```text
/_stcore/health
```

The app also records structured logs in `data/logs/app.log`.

## Data Locations

- Uploaded files: `data/uploads/<user_id>/`
- Vector store: `data/chroma/`
- Chat history JSONL: `data/history/`
- SQLite app DB: `data/app.sqlite3`
- Logs: `data/logs/`

These paths are intentionally ignored by Git.

## Recovery

If local state becomes corrupted during development:

1. Stop Streamlit or Docker.
2. Back up `data/` if needed.
3. Remove only the corrupted runtime artifact, for example `data/app.sqlite3` or `data/chroma/`.
4. Restart the app and re-ingest sources.

Do not commit runtime data, uploaded documents, `.env`, local database files, or ChromaDB artifacts.

## Current Production Limits

- SQLite is suitable for local single-instance use, not high-concurrency multi-instance deployment.
- The background worker is in-process. Use Redis + Celery/RQ for distributed workers.
- Object storage and malware scanning are still future production requirements.
- Centralized logs/metrics/alerts are not yet configured.
