# Tool System

Every tool inherits `BaseTool` and must return a structured `ToolResult`. Tools never return raw strings or `None`.

## Shared Tool Contract

Each tool declares:

- `name`
- `description`
- `supported_intents`
- `supported_source_types`
- `input_requirements`
- `output_types`
- `capabilities`
- `positive_examples`
- `negative_examples`
- `can_chain_after`
- `can_chain_before`
- `confidence`

Every tool is called through `safe_run()`, which converts exceptions into `ToolResult(success=False)`.

## Tools

### table_analysis_tool

Purpose: deterministic table analysis over CSV and Excel data.

Inputs:

- `query_plan`
- `dataframe` or file path
- optional `table_profile`

Outputs:

- `ToolResult.table`
- `ToolResult.data`
- `ToolResult.answer`
- pandas metadata

Supports:

- filtering
- groupby
- sum
- mean
- median
- count
- nunique
- min
- max
- top-k
- bottom-k
- comparison

### chart_tool

Purpose: generate Plotly charts from table results.

Inputs:

- table result from `table_analysis_tool`
- `query_plan.chart_type`

Outputs:

- Plotly figure in `ToolResult.chart`
- chart metadata

Supports:

- bar
- line
- pie
- scatter
- histogram

### summarize_tool

Purpose: create extractive document summaries and outlines.

Inputs:

- retrieved chunks or document chunks
- `query_plan`

Outputs:

- summary text
- citations

Supports:

- outline
- summary
- key points
- TLDR
- executive summary

### rag_qa_tool

Purpose: retrieve document evidence and answer from retrieved chunks.

Inputs:

- `query_plan`
- retriever or retrieved chunks
- optional metadata filter

Outputs:

- grounded answer
- retrieved chunk metadata
- citations

### compare_tool

Purpose: compare table and document outputs.

Inputs:

- dependency results from table and RAG tools

Outputs:

- comparison table
- comparison answer
- warnings for partial evidence

### url_lookup_tool

Purpose: load URL content, index it, and answer with grounded retrieval.

Inputs:

- URL or existing retriever
- `query_plan`

Outputs:

- grounded URL answer
- citations when available

### general_finance_tool

Purpose: answer general finance concept questions when no uploaded source is relevant.

Inputs:

- `query_plan`

Outputs:

- conceptual answer
- confidence and warnings

Gemini may be used for general conceptual narration. It is not used for calculations.

## Tool Chaining Behavior

Tool chains are selected by `ToolPlannerAgent` from `QueryPlan` and `SourceSelection`.

Common chains:

```text
table_analysis -> table_analysis_tool
chart_request -> table_analysis_tool -> chart_tool
summarize_document -> summarize_tool
rag_question -> rag_qa_tool
compare_documents -> table_analysis_tool -> rag_qa_tool -> compare_tool
general_finance -> general_finance_tool
```

Rules:

- `chart_tool` depends on `table_analysis_tool`.
- `compare_tool` depends on table and document evidence.
- If a dependency fails, the dependent tool is skipped safely.
- Every step returns a `ToolResult`, including skipped or failed tools.

## Why This Is Not Keyword Routing

The system uses semantic planning:

```text
query -> rewritten query -> QueryPlan -> source ranking -> tool capabilities -> ExecutionPlan
```

Tools are selected from structured intent, source compatibility, confidence, and dependency metadata.
