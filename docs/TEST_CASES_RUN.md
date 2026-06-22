# Test Cases Run

This file lists the test cases and test files used to verify the Financial Intelligence Chatbot project.

## Main Test Command

Use this command to run the project test suite:

```bash
venv/bin/python -m pytest
```

CI also runs:

```bash
python -m pytest -q
python evaluation/run_evaluation.py --check-only --minimum-accuracy 100
```

## Test Coverage By Area

### 1. App and UI Tests

- `tests/test_streamlit_app_smoke.py`
  - Checks that the Streamlit app starts without UI exceptions.
- `tests/test_app_ui_helpers.py`
  - Checks chat history formatting, table export, chart export, language preference mapping, summary shortcuts, and PDF reprocess helpers.

### 2. Configuration and Security Tests

- `tests/test_config.py`
  - Checks config loading, directory creation, Gemini key handling, environment loading, and safe defaults.
- `tests/test_security.py`
  - Checks allowed file types, file size validation, filename sanitization, and file type detection.
- `tests/test_upload.py`
  - Checks upload saving and metadata creation.
- `tests/test_upload_hardening.py`
  - Checks upload hardening and defensive validation paths.
- `tests/test_prompt_guard.py`
  - Checks prompt injection detection and warning/block behavior.
- `tests/test_reliability.py`
  - Checks retry logic and circuit breaker behavior.

### 3. Authentication, Storage, and History Tests

- `tests/test_auth_service.py`
  - Checks user registration, login, duplicate email rejection, and weak password rejection.
- `tests/test_sqlite_store.py`
  - Checks SQLite persistence for users, documents, jobs, and audit events.
- `tests/test_history_store.py`
  - Checks history save/load, document-specific history, export to markdown, and sanitization.
- `tests/test_background_jobs.py`
  - Checks background job success, failure handling, and job capacity limits.

### 4. File Ingestion Tests

- `tests/test_file_loader.py`
  - Checks CSV and Excel loading and invalid file handling.
- `tests/test_pdf_loader.py`
  - Checks PDF text extraction, table extraction, wrong extension rejection, corrupted PDF handling, and page timeout safety.
- `tests/test_docx_loader.py`
  - Checks DOCX paragraph and table extraction and invalid/corrupted file handling.
- `tests/test_url_loader.py`
  - Checks URL fetching, cleanup, and safe failure conditions.
- `tests/test_ingestion_service.py`
  - Checks table ingestion and document indexing flow.

### 5. RAG and Document Intelligence Tests

- `tests/test_chunker.py`
  - Checks chunk creation, empty text handling, and cleanup of decorative separator lines.
- `tests/test_embeddings.py`
  - Checks embedding generation and fallback embedding behavior.
- `tests/test_vector_store.py`
  - Checks vector storage and retrieval integration.
- `tests/test_retriever.py`
  - Checks top-k retrieval, reranking, score filtering, and duplicate handling.
- `tests/test_citations.py`
  - Checks citation building and snippet cleanup.
- `tests/test_rag_qa_tool.py`
  - Checks grounded answer generation, low-confidence refusal, empty retrieval handling, and prompt-injection-like document warnings.
- `tests/test_summarize_tool.py`
  - Checks summary, outline, resume-style formatting, full-document coverage, and safe failure behavior.
- `tests/test_url_lookup_tool.py`
  - Checks URL question answering using retrieved URL content.

### 6. Table Intelligence Tests

- `tests/test_table_profiler.py`
  - Checks profiling of rows, columns, numeric fields, entity candidates, metrics, and summary generation.
- `tests/test_column_mapper.py`
  - Checks semantic mapping from user terms to real dataframe columns.
- `tests/test_value_matcher.py`
  - Checks matching of user values to real dataframe values.
- `tests/test_pandas_executor.py`
  - Checks filtering, grouping, sorting, top-k, bottom-k, mean, median, min, max, count, comparison, and correlation operations.
- `tests/test_table_analysis_tool.py`
  - Checks the full table-analysis tool for average, count, grouping, correlation, entity filtering, and insight generation.

### 7. Query Understanding and Planning Tests

- `tests/test_language.py`
  - Checks language detection and normalization.
- `tests/test_confidence.py`
  - Checks confidence scoring helpers.
- `tests/test_query_rewriter.py`
  - Checks fallback rewrites, Hinglish handling, Spanish handling, and safe empty-query behavior.
- `tests/test_query_planner.py`
  - Checks planning of average, chart, summary, comparison, finance survey, anomaly, count, and semantic table-analysis questions.
- `tests/test_source_selector.py`
  - Checks correct source selection for table, document, and comparison flows.
- `tests/test_tool_planner.py`
  - Checks which tool chain is selected for each intent.
- `tests/test_tool_chain_executor.py`
  - Checks ordered execution, dependency passing, skip behavior, and safe failures.
- `tests/test_validator_agent.py`
  - Checks validation of source-dependent outputs, grounding, and failure handling.
- `tests/test_planning_validator.py`
  - Checks validity of planned queries and clarification cases.
- `tests/test_response_narrator.py`
  - Checks final response generation, citation preservation, chart preservation, Hinglish fallback, and Gemini narration safety.

### 8. Tool System Tests

- `tests/test_base_tool.py`
  - Checks `safe_run()` behavior and exception safety.
- `tests/test_tool_registry.py`
  - Checks tool registration and lookup.
- `tests/test_tool_manager.py`
  - Checks default tool initialization and exposed registry.
- `tests/test_chart_tool.py`
  - Checks bar, line, pie, scatter, histogram chart generation.
- `tests/test_compare_tool.py`
  - Checks structured and cited comparison logic.
- `tests/test_general_finance_tool.py`
  - Checks conceptual finance answering and fallback behavior.

### 9. Evaluation and Benchmark Tests

- `tests/test_evaluator.py`
  - Checks evaluation report generation, benchmark scoring, CSV accuracy, RAG citation presence, hallucination safety, comparison accuracy, and error handling.
- `evaluation/run_evaluation.py`
  - Runs the project benchmark gate used for evaluation.
- `evaluation/generate_csv_benchmarks.py`
  - Generates benchmark cases for structured data testing.

### 10. Assignment and End-to-End Tests

- `tests/test_assignment_acceptance.py`
  - Checks assignment-style routing logic such as chart requests, PDF comparison, DOCX summary, CSV average, Excel top-k, multilingual report question, and URL report question.
- `tests/test_assignment_end_to_end.py`
  - Checks full real execution flow from loading actual files to generating table answers, chart outputs, PDF comparison, DOCX summary, Spanish response, and URL answer.

### 11. General Model and Observability Tests

- `tests/test_models.py`
  - Checks core model creation and default field behavior.
- `tests/test_observability.py`
  - Checks trace logging, tool-call logging, redaction, and summary logging.

## Important Targeted Test Files

These are especially useful to mention in viva:

- `tests/test_query_planner.py`
- `tests/test_evaluator.py`
- `tests/test_assignment_acceptance.py`
- `tests/test_assignment_end_to_end.py`
- `tests/test_table_analysis_tool.py`
- `tests/test_rag_qa_tool.py`
- `tests/test_summarize_tool.py`
- `tests/test_streamlit_app_smoke.py`

## Simple Spoken Summary

"I used Pytest to test my project. I tested the UI startup, config, authentication, upload security, ingestion of CSV, Excel, PDF, DOCX, and URL sources, RAG retrieval, summarization, table analysis, chart generation, tool planning, comparison, evaluation benchmarks, and full assignment-style end-to-end flows."
