from __future__ import annotations

import json
from typing import Any, Dict, List


BASE_SAFETY_RULES = """
Return JSON only.
Do not calculate numeric answers.
Do not invent table values.
Do not invent document facts.
Do not execute code.
Include a confidence score between 0.0 and 1.0.
If the request is ambiguous, set clarification_needed true and provide a concise clarification_question.
""".strip()


def build_query_rewrite_prompt(query: str, language: str) -> str:
    schema = {
        "original_query": query,
        "rewritten_query": "clean analytical query",
        "language": language or "auto",
        "detected_language": "en",
        "confidence": 0.0,
        "notes": [],
    }
    return """
You are rewriting a financial/data analysis user query.

{rules}

Preserve intent, entities, metrics, output format, and language preference.

Schema:
{schema}

User query:
{query}
""".format(rules=BASE_SAFETY_RULES, schema=json.dumps(schema, indent=2), query=query)


def build_query_plan_prompt(
    query: str,
    rewritten_query: Dict[str, Any],
    available_sources: List[Dict[str, Any]],
    table_profiles: List[Dict[str, Any]] | None = None,
) -> str:
    schema = {
        "original_query": query,
        "rewritten_query": rewritten_query.get("rewritten_query", ""),
        "language": rewritten_query.get("language", "en"),
        "intent": "table_analysis",
        "required_source_type": "table",
        "entities": [],
        "metrics": [],
        "filters": [],
        "aggregations": [],
        "grouping": [],
        "sorting": {},
        "comparison": {},
        "chart_requested": False,
        "chart_type": None,
        "chart_types": [],
        "limit": None,
        "confidence": 0.0,
        "clarification_needed": False,
        "clarification_question": None,
        "reasoning_short": "",
    }
    return """
You are creating a structured plan for a financial intelligence system.

{rules}

Allowed intents:
table_analysis, chart_request, summarize_document, compare_documents, rag_question, url_lookup, general_finance.

Use only source metadata and provided table schema/profile data, not invented facts.
Understand the user's meaning before choosing intent. Users may use messy, multilingual, incomplete, paraphrased, or typo-filled language.
For table questions, infer likely metrics, entities, filters, and grouping from schema columns and sample values; do not require exact column names.
If the user requests multiple charts, put every requested type in chart_types and keep chart_type as the first type.

Schema:
{schema}

Available sources:
{sources}

Table schema profiles:
{profiles}

Original query:
{query}

Rewritten query:
{rewritten}
""".format(
        rules=BASE_SAFETY_RULES,
        schema=json.dumps(schema, indent=2),
        sources=json.dumps(available_sources, indent=2, default=str),
        profiles=json.dumps(table_profiles or [], indent=2, default=str),
        query=query,
        rewritten=json.dumps(rewritten_query, indent=2, default=str),
    )


def build_source_selection_prompt(query_plan: Dict[str, Any], available_sources: List[Dict[str, Any]]) -> str:
    schema = {
        "selected_source_id": None,
        "source_type": "",
        "confidence": 0.0,
        "reason": "",
        "alternatives": [],
    }
    return """
Select the most relevant uploaded source for a structured query plan.

{rules}

Rank using source type, schema/document metadata, query entities, metrics, grouping, filters, and comparison needs.

Schema:
{schema}

Query plan:
{plan}

Available sources:
{sources}
""".format(
        rules=BASE_SAFETY_RULES,
        schema=json.dumps(schema, indent=2),
        plan=json.dumps(query_plan, indent=2, default=str),
        sources=json.dumps(available_sources, indent=2, default=str),
    )


CLARIFICATION_PROMPT = """
Create one concise clarification question for an ambiguous financial intelligence query.
Return JSON only with fields: clarification_question, confidence.
""".strip()
