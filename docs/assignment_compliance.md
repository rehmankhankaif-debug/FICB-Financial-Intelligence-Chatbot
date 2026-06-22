# Assignment Compliance Report

## Current verdict

The local application implements the assignment workflow: structured and unstructured ingestion, autonomous source/tool selection, pandas calculations, RAG citations, charts, multilingual responses, history, exports, validation, and fallbacks.

This is assignment coverage, not enterprise certification. Production approval still requires target-environment infrastructure, security review, and load evidence.

## Routing contract

| Source/query | Execution path |
|---|---|
| CSV/XLS/XLSX calculation | profiling -> semantic mapping -> pandas -> validation |
| CSV/XLS/XLSX visualization | pandas result -> Plotly |
| PDF/DOCX/TXT/HTML question | parsing -> chunks -> retrieval -> RAG -> citations |
| Full document summary | full-source chunks -> bounded summary -> narrator |
| URL report | URL safety -> extraction -> indexing -> URL/RAG tool |
| Multiple documents | multi-source selection -> balanced retrieval -> comparison |
| General finance | general-finance tool -> Gemini or deterministic fallback |

Gemini does not calculate table answers. It rewrites, plans, or narrates verified tool output.

## Assignment examples

All seven examples have routing contracts in `tests/test_assignment_acceptance.py` and real file/tool execution coverage in `tests/test_assignment_end_to_end.py`:

1. Excel Q1/Q2 revenue trends route to pandas plus a line chart.
2. 2023/2022 PDF expense comparison extracts cited values and calculates absolute and percentage change.
3. Average monthly CSV profit routes to deterministic mean aggregation.
4. DOCX market-outlook summary routes to the summarizer.
5. Top-five Excel products route to deterministic ranking.
6. The Spanish report query retains Spanish and routes to RAG.
7. Linked online trends route to URL lookup/RAG.

## Implemented controls

- Configurable English, Hinglish, and Spanish.
- Multi-source comparison and balanced document retrieval.
- Deterministic cross-PDF financial comparison with currency/scale normalization and page/chunk citations.
- PDF text/table extraction, visual metadata, and optional Tesseract OCR.
- RAG citations, reranking, evidence validation, and prompt-injection warnings.
- SHA-256 duplicate detection and malicious-content/unsafe-archive checks.
- Per-user storage, auth, role-aware uploads, audit events, and rate budgets.
- SQLite WAL, bounded jobs, restart recovery, and container resource limits.
- Progressive rendering plus answer, table, chart, and history downloads.
- Retry, circuit breaker, structured logs, Docker, Compose, and CI artifacts.

## Remaining production conditions

- Managed Postgres, object storage, and production vector storage for horizontal scaling.
- Redis-backed distributed workers and shared rate limits.
- Deployment antivirus such as ClamAV; built-in scanning is not antivirus certification.
- TLS gateway, IAM/SSO, tenant controls, retention/deletion, backup/restore, and incident response.
- Browser E2E, load, accessibility, penetration, OCR-corpus, and real-report evaluation.
- Legal/security approval for the target financial-data jurisdiction.

## Submission checklist

- [x] Source, setup, architecture, tool documentation, tests, Docker, Compose, CI, and local demo flow
- [x] Public Git repository initialized with secrets, uploads, databases, logs, and virtual environments ignored
- [x] Reproducible benchmark gate included in GitHub Actions
- [ ] Run container validation on a machine with Docker
- [ ] Record/deploy the final demo and attach its URL or video

Final checks:

```bash
venv/bin/python -m pytest -q
venv/bin/python evaluation/run_evaluation.py
docker compose config -q
docker compose up --build
```

Never commit `.env`, `data/`, logs, uploads, databases, or the virtual environment.

## Latest local verification

- Full implementation suite: 300 tests passed in one run.
- Reproducible committed benchmark suite: 394/394 cases passed (100.00%).
- The benchmark runner executes committed cases on clean clones and exits non-zero on any failed case or sub-100 configured gate.
- Docker runtime verification was not available because Docker CLI is not installed on the current Mac environment.
