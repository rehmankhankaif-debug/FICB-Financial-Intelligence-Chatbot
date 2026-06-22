# Architecture

## System Architecture

The Financial Intelligence Chatbot is an agentic financial intelligence platform. Its goal is not simple CSV question answering. Its goal is to understand messy, indirect, multilingual, business-style questions and map them to the right structured or unstructured evidence source before any answer is generated.

Success is measured by whether the system understands the user's meaning, selects the correct source, maps business terms to schema fields or document chunks, executes deterministic calculations, validates groundedness, and explains the result safely.

Core layers:

- Upload and ingestion layer
- Table intelligence layer
- Document intelligence / RAG layer
- Query rewriter agent
- Query planner agent
- Source selector
- Tool planner
- Tool chain executor
- Validator / reflection agent
- Response narrator
- Streamlit presentation layer
- Authentication and user isolation layer
- Service layer for reusable backend logic
- Local background job layer
- Durable SQLite storage adapter
- History and observability layer
- Evaluation layer

## End-to-End Flow

```text
Upload
-> File type detection
-> Security validation
-> Background ingestion job
-> File parser
-> Table profiler or document chunker
-> Knowledge store

User query
-> Prompt-injection guard
-> Language detector
-> Query rewriter agent
-> Query planner agent
-> Source selector
-> Tool planner
-> Tool chain executor
-> Validator / reflection agent
-> Response narrator
-> Streamlit final answer
```

## CSV / Excel Intelligence Flow

```text
CSV / Excel file
-> load_table()
-> pandas DataFrame
-> TableProfiler
-> TableProfile
-> auto-generated semantic benchmark questions
-> SemanticColumnMapper
-> ValueMatcher
-> PandasExecutor
-> TableResultValidator
-> ToolResult
```

Important safeguards:

- No raw LLM-generated pandas code is executed.
- Numeric operations are structured dictionaries, not generated Python.
- pandas calculates all numeric values.
- Semantic column mapping uses exact, normalized, fuzzy, synonym, schema, and representative-value evidence.
- Vague values such as `Virat` can match table values such as `Virat Kohli`.
- Values are bound to their semantic field when the planner knows the field, so `Diesel` maps to `fuel_type` instead of long `model` or `title` strings.
- Analytical shortcuts handle percentage share, median-segment distributions, entity comparisons, and numeric correlations deterministically.

Example mappings:

- `diesel waale vehicles mein transmission ka trend kya hai` -> filter `fuel_type=Diesel`, group by `transmission`, count rows.
- `what percentage are automatic among diesel vehicles` -> denominator `fuel_type=Diesel`, numerator `fuel_type=Diesel AND transmission=Automatic`.
- `are price and overall cost related` -> pandas Pearson correlation over `price` and `overall_cost`.
- `above median price fuel distribution` -> filter `price > median(price)`, group by `fuel_type`.

## RAG Flow

```text
PDF / DOCX / URL / TXT / HTML
-> loader
-> DocumentChunkSource
-> DocumentChunker
-> EmbeddingService
-> ChromaDB VectorStore
-> Retriever
-> candidate expansion
-> lexical reranking
-> evidence threshold validation
-> RetrievedChunk
-> CitationBuilder
-> RAG tool result
```

RAG rules:

- Document answers must be grounded in retrieved chunks.
- Citations are preserved as `Citation` models.
- If retrieval is empty, the system says the information was not found.
- If retrieved chunks are below the evidence threshold, the RAG tool refuses to answer confidently.
- PDF/OCR/layout artifacts such as long resume divider lines are removed before chunks become answers or citations.
- Prompt-injection-like instructions inside retrieved documents are treated as untrusted content and surfaced as warnings.
- Raw financial document content is not logged.

## Authentication, Storage, and Jobs

The local production-ready path now includes:

- Streamlit login/register/logout.
- PBKDF2 password hashing; plaintext passwords are never stored.
- Per-user upload directories under `data/uploads/<user_id>/`.
- SQLite-backed users, document metadata, ingestion jobs, and audit events.
- A reusable `IngestionService` that can be called by Streamlit today and a worker/API later.
- A local `BackgroundJobManager` for queued ingestion, processing status, completion, failure, and cancellation-ready state.

This is intentionally not FastAPI yet. The project now has backend-style services that can later be exposed through FastAPI, React, mobile clients, or distributed workers without rewriting ingestion/RAG logic.

## Tool Chaining Flow

The system does not route tools by simple keyword checks. It uses:

```text
QueryPlan
-> SourceSelection
-> ToolPlannerAgent
-> ExecutionPlan
-> ToolChainExecutor
-> list[ToolResult]
```

Examples:

- Table analysis: `table_analysis_tool`
- Chart request: `table_analysis_tool -> chart_tool`
- Document summary: `summarize_tool`
- RAG question: `rag_qa_tool`
- Comparison: `table_analysis_tool -> rag_qa_tool -> compare_tool`
- General finance: `general_finance_tool`

Dependencies are explicit through `ToolCall.depends_on`. If a dependency fails, the dependent tool is skipped safely with a failed `ToolResult`.

## Validation Flow

```text
QueryPlan + ExecutionPlan + ToolResults + SourceSelection
-> ValidatorAgent
-> ValidationResult
```

The validator checks:

- Required tools succeeded.
- Table answers contain pandas-grounded data.
- Document answers include citations.
- Confidence is acceptable.
- Chart requests produced chart artifacts.
- Failed or partial results produce warnings or clarification.

Invalid results are not narrated as facts.

## Semantic Planning Contract

Every query is planned as structured meaning, not as a keyword route:

```text
User Query
-> Language Detection
-> Query Rewriter
-> Semantic Meaning Extraction
-> Entity Extraction
-> Metric Extraction
-> Intent Planning
-> Source Selection
-> Column / Field Matching
-> Tool Planning
-> Deterministic Execution
-> Validation
-> Grounded Answer Generation
```

For table data, the planner stores metrics, entities, filters, grouping, sorting, comparison metadata, chart intent, and confidence. For documents, source selection routes to semantic RAG over chunks and requires citations for grounded answers.

## History and Observability

The app stores:

- Session history
- Document-specific history
- Selected source and tools
- Execution time
- Confidence scores
- Warnings and errors
- Citations and final answer

Logs are JSONL and sanitize:

- API keys
- tokens
- secrets
- raw financial documents
- full uploaded content

## Evaluation Layer

The benchmark framework evaluates:

- Query rewrite quality
- Intent accuracy
- Semantic plan accuracy
- Source selection accuracy
- Tool selection accuracy
- CSV answer accuracy
- RAG citation presence
- Hallucination risk
- Error handling

CSV benchmarks are generated from every table profile in categories including aggregation, filtering, grouping, trend analysis, comparisons, ranking, top-k, bottom-k, anomaly detection, correlation exploration, segmentation, executive summaries, chart generation, business insights, follow-up questions, and multilingual questions.
