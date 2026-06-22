# Viva Questions

## 1. What is the main goal of this project?

The project is a financial intelligence chatbot that can analyze uploaded data and documents, answer natural language questions, generate charts, retrieve cited document context, validate outputs, and preserve chat history.

## 2. How is this different from a basic chatbot?

It does not answer only from an LLM. It uses ingestion, semantic profiling, source selection, deterministic pandas tools, RAG retrieval, validation, and structured response narration.

## 3. What is the high-level architecture?

Upload flows into security validation, parsing, profiling or chunking, and knowledge storage. User queries flow through language detection, rewriting, planning, source selection, tool planning, tool execution, validation, and narration.

## 4. Why is Gemini not used for calculations?

LLMs can make arithmetic mistakes or hallucinate numeric values. The system uses Gemini for language tasks only, while pandas performs all numeric calculations deterministically.

## 5. Why is pandas used for numeric answers?

pandas is deterministic, testable, and reliable for structured data operations such as filtering, grouping, aggregation, sorting, and ranking.

## 6. What is RAG and why is it used?

RAG means Retrieval-Augmented Generation. It retrieves relevant document chunks before answering, so document answers are grounded and can include citations.

## 7. What files does the system support?

The system supports CSV, XLSX, XLS, PDF, DOCX, TXT, HTML, and URLs.

## 8. How does CSV intelligence work?

The file is loaded into a DataFrame, profiled, semantically mapped to query terms, filtered or aggregated through a structured operation plan, and executed by pandas.

## 9. How does semantic column mapping work?

It combines exact matches, normalized matches, fuzzy matching, and synonym groups. For example, `sales`, `revenue`, and `income` can map to related columns.

## 10. How does value matching work?

It matches user entities to dataframe values using exact, contains, and fuzzy matching. For example, `Virat` can match `Virat Kohli`.

## 11. What prevents raw code execution?

The pandas executor accepts structured operation dictionaries only. It never executes user-provided Python or LLM-generated pandas code.

## 12. What is a QueryPlan?

`QueryPlan` is a structured representation of the user intent, source type, entities, metrics, filters, aggregations, grouping, sorting, chart request, confidence, and clarification needs.

## 13. What is an ExecutionPlan?

`ExecutionPlan` contains selected sources, ordered tool calls, dependency information, confidence, and warnings.

## 14. How does autonomous tool invocation work?

The query planner creates a `QueryPlan`, the source selector ranks uploaded sources, and the tool planner maps the plan to a tool chain using tool capabilities and source compatibility.

## 15. Why is this not keyword routing?

Routing is based on structured intent, source compatibility, semantic metadata, confidence scores, and tool capability descriptions, not direct string checks like `if summary in query`.

## 16. What is ToolResult?

`ToolResult` is the structured output returned by every tool. It includes success, data, answer, table, chart, citations, confidence, warnings, errors, and metadata.

## 17. What happens if a tool fails?

`safe_run()` catches exceptions and returns `ToolResult(success=False)`. The UI receives a friendly warning instead of crashing.

## 18. How are charts generated?

The chart tool receives table output from pandas and builds a Plotly figure. It supports bar, line, pie, scatter, and histogram charts.

## 19. How does PDF processing work?

PDF text is extracted with `pypdf`, page metadata is preserved, text is chunked, embeddings are created, and chunks are stored in ChromaDB.

## 20. How does DOCX processing work?

Paragraphs and tables are extracted with `python-docx`, converted into a common document format, chunked, embedded, and stored.

## 21. How does URL processing work?

The URL loader fetches content with `requests`, cleans HTML with BeautifulSoup, removes scripts and navigation, chunks the text, embeds it, and stores it for retrieval.

## 22. What is ChromaDB used for?

ChromaDB stores document chunk embeddings and supports semantic search over uploaded documents and URLs.

## 23. What embedding model is used?

The default model is `all-MiniLM-L6-v2`. If it is not available locally, the embedding service can use a deterministic fallback.

## 24. How are citations created?

Retrieved chunks are converted into `Citation` objects containing source ID, filename, page, chunk ID, and snippet.

## 25. What does the validator agent check?

It checks tool success, table grounding, citation presence, confidence, missing columns, missing chunks, chart artifacts, and clarification requirements.

## 26. What happens with low confidence?

Low confidence triggers warnings, clarification questions, or safe fallback behavior. The system avoids presenting uncertain answers as final facts.

## 27. How does chat history work?

Each completed turn is stored as a `ChatHistoryRecord` with query, plan, selected source, selected tools, tool results, final answer, citations, warnings, errors, execution time, and confidence scores.

## 28. What is logged for observability?

The system logs safe metadata such as event type, trace ID, tool names, execution time, confidence scores, warnings, and errors.

## 29. How is sensitive data protected in logs?

API keys, tokens, secrets, and raw document content are redacted before being written to JSONL logs or history.

## 30. How would you improve this project next?

Future improvements include richer financial trend analytics, multi-user authentication, larger multilingual benchmarks, dashboard observability, stronger comparison tools, and enterprise privacy controls.
