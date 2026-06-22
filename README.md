# Financial Intelligence Chatbot

Production-grade financial document and data intelligence assistant built with Python, Streamlit, pandas, Plotly, Gemini, ChromaDB, and sentence-transformers.

This project goes beyond a basic chatbot. It ingests structured files and documents, understands natural language and Hinglish questions, plans tool usage, executes deterministic analysis, retrieves grounded document context, validates outputs, narrates final answers, and records history and observability traces.

## Features

- CSV and Excel upload with semantic table profiling.
- PDF, DOCX, TXT, HTML, and URL ingestion.
- RAG-based document question answering with citations, retrieval reranking, evidence thresholds, and document-artifact cleanup.
- Configurable English, Hinglish, and Spanish query/response handling.
- Query rewriting, query planning, source selection, tool planning, tool chaining, validation, and response narration.
- Deterministic pandas calculations for numeric table answers.
- Multi-chart Plotly generation with downloadable chart HTML and table CSV.
- Multi-source document comparison with balanced evidence retrieval, cited financial values, normalized units, absolute differences, and percentage changes.
- PDF table extraction, visual metadata, and optional OCR.
- Local login/register with password hashing and per-user upload isolation.
- SQLite-backed users, document metadata, job records, and audit events.
- Bounded background jobs with progress, overload protection, and restart recovery.
- Duplicate detection, role-aware uploads, rate limits, and malicious-content checks.
- Chat history, Markdown export, JSONL logs, trace events, and prompt-injection warnings.
- Reproducible evaluation framework for rewrite, intent, source, tool, CSV, executed RAG, hallucination, multilingual finance, cited document comparison, and error-handling checks.
- Streamlit UI with friendly errors and safe session state.

## Architecture

Runtime flow:

```text
Upload
-> Security validation
-> Background ingestion job
-> Parser
-> Table profiler or document chunker
-> Knowledge store

User query
-> Prompt-injection guard
-> Language detection
-> Query rewriter
-> Query planner
-> Source selector
-> Tool planner
-> Tool chain executor
-> Validator / reflection agent
-> Response narrator
-> Final answer with tables, charts, citations, warnings
```

The system separates reasoning from execution:

- Streamlit presents the UI, while reusable services handle ingestion, auth, storage, jobs, RAG, tools, and validation.
- Gemini may rewrite, plan, and narrate.
- pandas performs all table calculations.
- RAG retrieves document evidence, reranks candidates, validates evidence thresholds, and returns citations.
- Tools always return structured `ToolResult`.
- The UI never directly executes user-provided code or LLM-generated pandas code.

## Setup

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## .env Setup

Create `.env` from `.env.example`:

```bash
cp .env.example .env
```

Then add your Gemini key:

```bash
GEMINI_API_KEY=your_key_here
```

Useful local production settings:

```bash
APP_ENV=local
LOG_LEVEL=INFO
APP_DATABASE_PATH=data/app.sqlite3
BACKGROUND_WORKER_COUNT=2
```

The app still starts if the key is missing. In that case, deterministic fallbacks are used where possible, and Gemini narration/planning is unavailable.

## Run

```bash
streamlit run app.py
```

If you are using the local venv:

```bash
venv/bin/streamlit run app.py
```

## Test

```bash
pytest
```

Or with the local venv:

```bash
venv/bin/python -m pytest
```

Run the benchmark framework:

```bash
venv/bin/python evaluation/run_evaluation.py
```

Run the same non-writing benchmark gate used by CI:

```bash
venv/bin/python evaluation/run_evaluation.py --check-only --minimum-accuracy 100
```

The default run always loads the committed benchmark corpus, including generated semantic table cases, multilingual financial cases, actual RAG execution, and cited numeric PDF comparison. It does not depend on private files under `data/uploads/`.

## Operations

- Local and Docker runbook: `docs/operations.md`
- Architecture notes: `docs/architecture.md`
- Production readiness review: `docs/production_readiness_review.md`
- Assignment compliance and submission checklist: `docs/assignment_compliance.md`

## Supported File Types

- CSV: `.csv`
- Excel: `.xlsx`, `.xls`
- PDF: `.pdf`
- Word: `.docx`
- Text: `.txt`
- HTML: `.html`
- URL: `http://...` or `https://...` through the URL input

## Example Queries

- `Average monthly profit batao`
- `Top five products by sales`
- `manual aur automatic cars kitni hain bar graph bnao`
- `Virat ke max runs aur strike rate batao`
- `Outline this PDF`
- `What risks are mentioned in this report?`
- `Compare revenue trends in CSV with annual report PDF`
- `What is EBITDA?`

## Project Structure

```text
app.py
src/
  agents/
  auth/
  evaluation/
  history/
  ingestion/
  jobs/
  llm/
  models/
  observability/
  rag/
  security/
  services/
  storage/
  table_intelligence/
  tools/
  utils/
evaluation/
docs/
tests/
data/
```

## Limitations

- The deterministic fallback planner is intentionally conservative; Gemini improves planning quality when configured.
- RAG is cleaner and safer now, but enterprise-grade hybrid search with a dedicated cross-encoder reranker is still a future improvement.
- SQLite is used for local production-style durability. Postgres is the recommended next database for multi-instance deployment.
- Background jobs are local in-process workers. Celery/RQ plus Redis would be the next step for distributed workers.
- The local embedding model may fall back to hashed embeddings if the sentence-transformer model is unavailable locally.
- Numeric PDF comparison handles common financial statements with explicit metrics, periods, currencies, and scales; ambiguous values or incompatible currencies are returned with warnings instead of unsafe arithmetic.
- URL loading depends on page accessibility and may fail for heavily scripted or blocked sites.

## Demo Flow

1. Start the app with `streamlit run app.py`.
2. Create an account or log in.
3. Upload a CSV, Excel, PDF, DOCX, TXT, or HTML file.
4. Watch the ingestion job status, then ask a table or document question.
5. Ask for a chart from a table result.
6. Add a URL and ask a grounded question after the URL job completes.
7. Export chat history from the sidebar.
