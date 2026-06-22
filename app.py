from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import pandas as pd
import streamlit as st

from src.auth.service import AuthError, AuthService, DuplicateUserError, InvalidCredentialsError, WeakPasswordError
from src.agents.query_planner import QueryPlannerAgent
from src.agents.query_rewriter import QueryRewriterAgent
from src.agents.response_narrator import ResponseNarrator
from src.agents.source_selector import SourceSelector
from src.agents.tool_chain_executor import ToolChainExecutor
from src.agents.tool_planner import ToolPlannerAgent
from src.agents.validator_agent import ValidatorAgent
from src.config import DATABASE_PATH, settings
from src.history.store import HistoryStore, export_records_to_markdown
from src.jobs.manager import BackgroundJobManager
from src.llm.gemini_client import GeminiClient
from src.models import ChatHistoryRecord, Citation, DocumentSource, ExecutionPlan, FinalResponse, QueryPlan, RewrittenQuery, ValidationResult
from src.models.source import SourceSelection
from src.models.table import TableProfile
from src.observability.tracing import TraceRecorder
from src.rag.chunker import DocumentChunker
from src.rag.embeddings import EmbeddingService
from src.rag.retriever import Retriever
from src.rag.vector_store import VectorStore
from src.security.prompt_guard import PromptInjectionGuard
from src.security.rate_limit import RateLimitExceeded, SlidingWindowRateLimiter, can_upload
from src.services.ingestion_service import DOCUMENT_FILE_TYPES, TABLE_FILE_TYPES, IngestionResult, IngestionService
from src.storage.sqlite_store import SQLiteStore
from src.tools.manager import ToolManager
from src.utils.language import detect_language, language_code_for_label, language_options
from src.utils.logging import log_error, log_event, log_query
from src.utils.security import sanitize_filename
from src.utils.upload import save_uploaded_file


DEFAULT_TOP_K = 5

st.set_page_config(page_title="Financial Intelligence Chatbot", page_icon="FI", layout="wide")


def dump_model(model: Any) -> Dict[str, Any]:
    if model is None:
        return {}
    if isinstance(model, dict):
        return dict(model)
    if hasattr(model, "model_dump"):
        return model.model_dump()
    if hasattr(model, "dict"):
        return model.dict()
    return {"value": str(model)}


def model_list_payload(items: List[Any]) -> List[Dict[str, Any]]:
    return [dump_model(item) for item in items or []]


def source_category(source: Dict[str, Any]) -> str:
    file_type = str(source.get("file_type") or "").lower()
    if file_type in TABLE_FILE_TYPES:
        return "table"
    if file_type in DOCUMENT_FILE_TYPES:
        return "document"
    return "unknown"


def is_simple_single_document_summary(query: str) -> bool:
    text = str(query or "").lower()
    summary_signal = any(term in text for term in ["summarize", "summarise", "summary", "outline", "key points", "tldr"])
    sources = uploaded_source_payloads()
    return summary_signal and len(sources) == 1 and source_category(sources[0]) == "document"


def language_preference_value(label: str) -> Optional[str]:
    return language_code_for_label(label, settings.supported_languages)


def dataframe_from_table(table: Any) -> Optional[pd.DataFrame]:
    if table is None:
        return None
    if isinstance(table, pd.DataFrame):
        return table
    if isinstance(table, list) and table:
        return pd.DataFrame(table)
    if isinstance(table, dict) and table:
        return pd.DataFrame([table])
    return None


def stream_text_chunks(text: str, words_per_chunk: int = 5):
    words = str(text or "").split()
    chunk_size = max(1, int(words_per_chunk))
    for index in range(0, len(words), chunk_size):
        suffix = " " if index + chunk_size < len(words) else ""
        yield " ".join(words[index:index + chunk_size]) + suffix


def table_export_csv(table: Any) -> Optional[str]:
    dataframe = dataframe_from_table(table)
    if dataframe is None:
        return None
    return dataframe.to_csv(index=False)


def chart_export_html(chart: Any) -> Optional[str]:
    if chart is None or not hasattr(chart, "to_html"):
        return None
    try:
        return chart.to_html(full_html=True, include_plotlyjs="cdn")
    except Exception:
        return None


def citation_label(citation: Citation) -> str:
    label = citation.filename or citation.source_id or "source"
    if citation.page is not None:
        label = "{0}, page {1}".format(label, citation.page)
    if citation.chunk_id:
        label = "{0}, chunk {1}".format(label, citation.chunk_id)
    return label


def build_chat_history_markdown(history_records: List[Any]) -> str:
    return export_records_to_markdown(history_records)


@st.cache_resource(show_spinner=False)
def get_tool_manager() -> ToolManager:
    return ToolManager()


@st.cache_resource(show_spinner=False)
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService(local_files_only=True)


@st.cache_resource(show_spinner=False)
def get_vector_store() -> VectorStore:
    return VectorStore(collection_name="financial_documents")


@st.cache_resource(show_spinner=False)
def get_chunker() -> DocumentChunker:
    return DocumentChunker()


@st.cache_resource(show_spinner=False)
def get_history_store() -> HistoryStore:
    return HistoryStore()


@st.cache_resource(show_spinner=False)
def get_trace_recorder() -> TraceRecorder:
    return TraceRecorder()


@st.cache_resource(show_spinner=False)
def get_app_store() -> SQLiteStore:
    return SQLiteStore(getattr(settings, "database_path", DATABASE_PATH))


@st.cache_resource(show_spinner=False)
def get_auth_service() -> AuthService:
    return AuthService(get_app_store())


@st.cache_resource(show_spinner=False)
def get_ingestion_service() -> IngestionService:
    return IngestionService(
        chunker=get_chunker(),
        vector_store=get_vector_store(),
        embedding_service=get_embedding_service(),
    )


@st.cache_resource(show_spinner=False)
def get_rate_limiter() -> SlidingWindowRateLimiter:
    return SlidingWindowRateLimiter()


@st.cache_resource(show_spinner=False)
def get_job_manager() -> BackgroundJobManager:
    return BackgroundJobManager(get_app_store(), max_workers=settings.background_worker_count)


WORKSPACE_STATE_KEYS = [
    "uploaded_sources",
    "dataframes",
    "table_profiles",
    "table_benchmarks",
    "document_chunks",
    "document_metadata",
    "processed_source_ids",
    "messages",
    "history_records",
    "latest_response",
    "ingestion_events",
]


def initialize_session_state() -> None:
    defaults = {
        "session_id": uuid4().hex,
        "auth_user": None,
        "uploaded_sources": [],
        "dataframes": {},
        "table_profiles": {},
        "table_benchmarks": {},
        "document_chunks": {},
        "document_metadata": {},
        "processed_source_ids": set(),
        "messages": [],
        "history_records": [],
        "latest_response": None,
        "ingestion_events": [],
        "active_ingestion_jobs": {},
        "applied_job_ids": set(),
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_workspace_state() -> None:
    st.session_state.session_id = uuid4().hex
    st.session_state.uploaded_sources = []
    st.session_state.dataframes = {}
    st.session_state.table_profiles = {}
    st.session_state.table_benchmarks = {}
    st.session_state.document_chunks = {}
    st.session_state.document_metadata = {}
    st.session_state.processed_source_ids = set()
    st.session_state.messages = []
    st.session_state.history_records = []
    st.session_state.latest_response = None
    st.session_state.ingestion_events = []
    st.session_state.active_ingestion_jobs = {}
    st.session_state.applied_job_ids = set()


def set_authenticated_user(user: Any) -> None:
    st.session_state.auth_user = user.public_dict() if hasattr(user, "public_dict") else dict(user or {})
    reset_workspace_state()


def current_user() -> Optional[Dict[str, Any]]:
    user = st.session_state.get("auth_user")
    return user if isinstance(user, dict) and user.get("user_id") else None


def current_user_id() -> Optional[str]:
    user = current_user()
    return str(user.get("user_id")) if user else None


def current_user_upload_dir() -> Path:
    user_id = current_user_id() or "anonymous"
    return settings.upload_dir / user_id


def render_auth_gate() -> bool:
    if current_user():
        return True

    st.title("Financial Intelligence Chatbot")
    st.caption("Production-grade financial document and data intelligence assistant.")
    st.caption("Sign in to keep uploads, chat history, and document intelligence isolated to your account.")

    login_tab, register_tab = st.tabs(["Login", "Create account"])

    with login_tab:
        with st.form("login_form"):
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Password", type="password", key="login_password")
            submitted = st.form_submit_button("Login", type="primary")
        if submitted:
            try:
                user = get_auth_service().authenticate(email, password)
                set_authenticated_user(user)
                st.success("Logged in as {0}".format(user.email))
                st.rerun()
            except InvalidCredentialsError as exc:
                st.error(str(exc))
            except Exception as exc:
                log_error(exc, {"stage": "login"})
                st.error("Login failed safely. Please try again.")

    with register_tab:
        with st.form("register_form"):
            display_name = st.text_input("Name", key="register_name")
            email = st.text_input("Email", key="register_email")
            password = st.text_input("Password", type="password", key="register_password")
            confirm_password = st.text_input("Confirm password", type="password", key="register_confirm_password")
            submitted = st.form_submit_button("Create account", type="primary")
        if submitted:
            try:
                if password != confirm_password:
                    raise AuthError("Passwords do not match.")
                user = get_auth_service().register(email=email, password=password, display_name=display_name)
                set_authenticated_user(user)
                st.success("Account created for {0}".format(user.email))
                st.rerun()
            except (AuthError, DuplicateUserError, WeakPasswordError) as exc:
                st.error(str(exc))
            except Exception as exc:
                log_error(exc, {"stage": "register"})
                st.error("Registration failed safely. Please try again.")

    return False


def logout_current_user() -> None:
    user = current_user()
    if user:
        get_app_store().record_audit_event("auth.logout", {"email": user.get("email")}, user_id=user.get("user_id"))
    st.session_state.auth_user = None
    reset_workspace_state()


def add_ingestion_event(level: str, message: str) -> None:
    st.session_state.ingestion_events.append(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": level,
            "message": message,
        }
    )
    st.session_state.ingestion_events = st.session_state.ingestion_events[-20:]


def submit_ingestion_job(source: DocumentSource) -> str:
    user_id = current_user_id()
    if not user_id:
        raise RuntimeError("Login is required before ingesting sources.")
    source.status = "processing"
    upsert_source(source)
    job_id = get_job_manager().submit(
        user_id=user_id,
        job_type="ingest_source",
        source_id=source.source_id,
        metadata={"filename": source.filename, "file_type": source.file_type},
        handler=lambda source=source: get_ingestion_service().ingest_source(source),
    )
    st.session_state.active_ingestion_jobs[job_id] = source.source_id
    add_ingestion_event("warning", "Queued ingestion for {0}".format(source.filename))
    return job_id


def process_completed_ingestion_jobs() -> None:
    active_jobs = dict(st.session_state.get("active_ingestion_jobs") or {})
    for job_id, source_id in active_jobs.items():
        if job_id in st.session_state.applied_job_ids:
            continue
        state = get_job_manager().get_job(job_id)
        if state is None:
            continue
        if state.status == "completed" and isinstance(state.result, IngestionResult):
            state.result.source.status = "uploaded"
            apply_ingestion_result(state.result)
            st.session_state.applied_job_ids.add(job_id)
            st.session_state.active_ingestion_jobs.pop(job_id, None)
            add_ingestion_event("success", "Indexed {0}".format(state.result.source.filename))
        elif state.status == "failed":
            mark_source_failed(source_id, state.error_msg or "Ingestion job failed.")
            st.session_state.applied_job_ids.add(job_id)
            st.session_state.active_ingestion_jobs.pop(job_id, None)
            add_ingestion_event("error", "Ingestion failed: {0}".format(state.error_msg or source_id))
        elif state.status == "canceled":
            mark_source_failed(source_id, "Ingestion job was canceled.")
            st.session_state.applied_job_ids.add(job_id)
            st.session_state.active_ingestion_jobs.pop(job_id, None)


def mark_source_failed(source_id: str, error_msg: str) -> None:
    for source_payload in st.session_state.uploaded_sources:
        if source_payload.get("source_id") == source_id:
            source_payload["status"] = "failed"
            source_payload["error_msg"] = error_msg
            try:
                get_app_store().upsert_document_source(current_user_id() or "", DocumentSource(**source_payload))
            except Exception:
                pass


def upsert_source(source: DocumentSource) -> None:
    payload = dump_model(source)
    existing = [
        item
        for item in st.session_state.uploaded_sources
        if item.get("source_id") != source.source_id or not source.source_id
    ]
    existing.append(payload)
    st.session_state.uploaded_sources = existing
    user_id = current_user_id()
    if user_id and source.source_id:
        try:
            get_app_store().upsert_document_source(user_id, source)
        except Exception as exc:
            log_error(exc, {"stage": "persist_document_source", "source_id": source.source_id, "user_id": user_id})


def uploaded_source_payloads() -> List[Dict[str, Any]]:
    return [
        source
        for source in st.session_state.uploaded_sources
        if source.get("status") == "uploaded" and source.get("source_id")
    ]


def known_source_payloads() -> List[Dict[str, Any]]:
    return [
        source
        for source in st.session_state.uploaded_sources
        if source.get("source_id") and source.get("status") != "failed"
    ]


def restore_duplicate_source(fresh_source: DocumentSource, duplicate: Any) -> DocumentSource:
    existing = duplicate if isinstance(duplicate, DocumentSource) else DocumentSource(**dict(duplicate or {}))
    fresh_path = Path(fresh_source.path) if fresh_source.path else None
    existing_path = Path(existing.path) if existing.path else None

    if existing_path is not None and existing_path.is_file():
        if fresh_path is not None and fresh_path != existing_path:
            fresh_path.unlink(missing_ok=True)
    else:
        existing.path = fresh_source.path
        existing.metadata.update(dict(fresh_source.metadata or {}))

    existing.metadata["user_id"] = current_user_id()
    upsert_source(existing)
    return existing


def reprocess_pdf_source(source_id: str) -> bool:
    user = current_user() or {}
    if not can_upload(user.get("role")):
        add_ingestion_event("error", "Your account role is not allowed to reprocess sources.")
        return False

    source_payload = next(
        (item for item in st.session_state.uploaded_sources if item.get("source_id") == source_id),
        None,
    )
    if not source_payload:
        add_ingestion_event("error", "The selected PDF is no longer available. Upload it again first.")
        return False
    if str(source_payload.get("file_type") or "").lower() != "pdf":
        add_ingestion_event("error", "Only PDF sources can be reprocessed with this action.")
        return False
    if source_id in set((st.session_state.get("active_ingestion_jobs") or {}).values()):
        add_ingestion_event("warning", "This PDF is already being processed.")
        return False

    source = DocumentSource(**source_payload)
    if not source.path or not Path(source.path).is_file():
        add_ingestion_event("error", "The stored PDF file is missing. Upload the PDF again to restore it.")
        return False

    try:
        get_rate_limiter().check(
            current_user_id() or "anonymous",
            "upload",
            settings.upload_rate_limit_per_minute,
        )
        source.status = "uploaded"
        source.error_msg = None
        job_id = submit_ingestion_job(source)
        log_event(
            "pdf_reprocessing_queued",
            {
                "source_id": source.source_id,
                "job_id": job_id,
                "filename": source.filename,
                "user_id": current_user_id(),
            },
        )
        return True
    except Exception as exc:
        add_ingestion_event("error", "PDF reprocessing failed safely: {0}".format(str(exc)))
        log_error(exc, {"stage": "reprocess_pdf_source", "source_id": source_id, "user_id": current_user_id()})
        return False


def table_profile_models() -> List[TableProfile]:
    profiles = []
    for payload in st.session_state.table_profiles.values():
        try:
            profiles.append(payload if isinstance(payload, TableProfile) else TableProfile(**payload))
        except Exception:
            continue
    return profiles


def document_metadata_payloads() -> List[Dict[str, Any]]:
    return list(st.session_state.document_metadata.values())


def save_and_ingest_uploaded_files(uploaded_files: List[Any]) -> None:
    user = current_user() or {}
    if not can_upload(user.get("role")):
        add_ingestion_event("error", "Your account role is not allowed to upload sources.")
        return
    for uploaded_file in uploaded_files or []:
        try:
            get_rate_limiter().check(
                current_user_id() or "anonymous",
                "upload",
                settings.upload_rate_limit_per_minute,
            )
            source = save_uploaded_file(uploaded_file, upload_dir=current_user_upload_dir())
            if current_user_id():
                source.metadata["user_id"] = current_user_id()
            if source.status != "uploaded":
                upsert_source(source)
                add_ingestion_event("error", "{0}: {1}".format(source.filename, source.error_msg))
                log_event("file_upload_failed", {"filename": source.filename, "error": source.error_msg, "user_id": current_user_id()})
                continue
            content_hash = source.metadata.get("content_sha256")
            session_duplicate = next(
                (
                    item for item in st.session_state.uploaded_sources
                    if item.get("metadata", {}).get("content_sha256") == content_hash
                ),
                None,
            )
            database_duplicate = get_app_store().find_document_by_content_hash(current_user_id(), content_hash) if current_user_id() else None
            duplicate = session_duplicate or database_duplicate
            if duplicate:
                existing = restore_duplicate_source(source, duplicate)
                active_source_ids = set((st.session_state.get("active_ingestion_jobs") or {}).values())
                if existing.source_id in active_source_ids:
                    message = "{0} is already being processed. Wait for the current ingestion job to finish.".format(existing.filename)
                else:
                    message = "Duplicate detected: {0} already exists. Select it below and click Reprocess PDF.".format(existing.filename)
                add_ingestion_event("warning", message)
                log_event("duplicate_upload_skipped", {"filename": source.filename, "content_sha256": content_hash, "user_id": current_user_id()})
                continue
            upsert_source(source)
            job_id = submit_ingestion_job(source)
            log_event(
                "file_uploaded_and_queued",
                {
                    "source_id": source.source_id,
                    "job_id": job_id,
                    "filename": source.filename,
                    "file_type": source.file_type,
                    "size_bytes": source.metadata.get("size_bytes"),
                    "user_id": current_user_id(),
                },
            )
        except Exception as exc:
            add_ingestion_event("error", "Upload failed safely: {0}".format(str(exc)))
            log_error(exc, {"stage": "save_and_ingest_uploaded_files", "user_id": current_user_id()})


def ingest_source(source: DocumentSource) -> None:
    if source.source_id in st.session_state.processed_source_ids:
        return
    result = get_ingestion_service().ingest_source(source)
    apply_ingestion_result(result)
    st.session_state.processed_source_ids.add(source.source_id)


def apply_ingestion_result(result: IngestionResult) -> None:
    source = result.source
    if result.source_category == "table" and result.dataframe is not None and result.table_profile is not None:
        st.session_state.dataframes[source.source_id] = result.dataframe
        st.session_state.table_profiles[source.source_id] = dump_model(result.table_profile)
        st.session_state.table_benchmarks[source.source_id] = list(result.table_benchmarks or [])
    elif result.source_category == "document":
        st.session_state.document_chunks[source.source_id] = result.chunk_payloads
        st.session_state.document_metadata[source.source_id] = dict(result.document_metadata or {})
    upsert_source(source)


def add_url_source(url: str) -> None:
    try:
        safe_url = (url or "").strip()
        if not safe_url:
            add_ingestion_event("warning", "Enter a URL before adding it.")
            return
        source = DocumentSource(
            source_id=uuid4().hex,
            filename=sanitize_filename(safe_url.replace("https://", "").replace("http://", "")) or "url_source",
            file_type="url",
            path=safe_url,
            metadata={"url": safe_url, "source_category": "document", "user_id": current_user_id()},
            status="uploaded",
        )
        upsert_source(source)
        job_id = submit_ingestion_job(source)
        log_event("url_uploaded_and_queued", {"source_id": source.source_id, "job_id": job_id, "url": safe_url, "user_id": current_user_id()})
    except Exception as exc:
        add_ingestion_event("error", "URL ingestion failed: {0}".format(str(exc)))
        log_error(exc, {"stage": "add_url_source", "url": url, "user_id": current_user_id()})


def source_for_tool(tool_name: str, selected_source: Optional[SourceSelection]) -> Optional[Dict[str, Any]]:
    sources = uploaded_source_payloads()
    selected_id = selected_source.selected_source_id if selected_source else None
    selected = next((source for source in sources if source.get("source_id") == selected_id), None)

    if tool_name == "table_analysis_tool":
        if selected and source_category(selected) == "table":
            return selected
        return next((source for source in sources if source_category(source) == "table"), None)

    if tool_name in {"rag_qa_tool", "summarize_tool", "url_lookup_tool"}:
        if selected and source_category(selected) == "document":
            return selected
        return next((source for source in sources if source_category(source) == "document"), None)

    return selected


def hydrate_execution_plan(execution_plan: ExecutionPlan, source_selection: Optional[SourceSelection], top_k: int) -> ExecutionPlan:
    retriever = Retriever(get_vector_store(), get_embedding_service())
    selected_ids = list(source_selection.selected_source_ids or []) if source_selection else []
    if source_selection and source_selection.selected_source_id and source_selection.selected_source_id not in selected_ids:
        selected_ids.append(source_selection.selected_source_id)
    selected_sources = [
        source for source in uploaded_source_payloads()
        if not selected_ids or source.get("source_id") in selected_ids
    ]
    for tool_call in execution_plan.tool_calls:
        payload = dict(tool_call.input_payload or {})
        payload["query_plan"] = dump_model(execution_plan.query_plan)
        source = source_for_tool(tool_call.tool_name, source_selection)
        if execution_plan.query_plan.intent == "compare_documents" and tool_call.tool_name == "table_analysis_tool":
            table_sources = [item for item in selected_sources if source_category(item) == "table"]
            payload["dataframes"] = {
                item.get("source_id"): st.session_state.dataframes.get(item.get("source_id"))
                for item in table_sources
            }
            payload["table_profiles"] = {
                item.get("source_id"): st.session_state.table_profiles.get(item.get("source_id"))
                for item in table_sources
            }
            payload["source_descriptors"] = {
                item.get("source_id"): item for item in table_sources
            }
        if execution_plan.query_plan.intent == "compare_documents" and tool_call.tool_name == "rag_qa_tool":
            document_sources = [item for item in selected_sources if source_category(item) == "document"]
            payload["source_ids"] = [item.get("source_id") for item in document_sources if item.get("source_id")]
            payload["source_descriptors"] = {
                item.get("source_id"): item for item in document_sources
            }
            payload["retriever"] = retriever
            payload["top_k"] = top_k
            payload.pop("metadata_filter", None)
        if source:
            source_id = source.get("source_id")
            payload["source_selection"] = source
            payload["source_id"] = source_id
            if tool_call.tool_name == "table_analysis_tool":
                payload["dataframe"] = st.session_state.dataframes.get(source_id)
                payload["table_profile"] = st.session_state.table_profiles.get(source_id)
                payload["path"] = source.get("path")
            elif tool_call.tool_name == "summarize_tool":
                # A document summary must cover the whole source. The summarize tool
                # applies its own bounded context budget before Gemini narration.
                payload["document_chunks"] = st.session_state.document_chunks.get(source_id, [])
            elif tool_call.tool_name in {"rag_qa_tool", "url_lookup_tool"}:
                payload["retriever"] = retriever
                payload["top_k"] = top_k
                payload["metadata_filter"] = {"source_id": source_id}
                payload["vector_store"] = get_vector_store()
                payload["embedding_service"] = get_embedding_service()
                if source.get("file_type") == "url":
                    payload["url"] = source.get("path") or source.get("metadata", {}).get("url")
        tool_call.input_payload = payload
    return execution_plan


def build_security_blocked_pipeline_result(user_query: str, language: Optional[str], warning: str) -> Dict[str, Any]:
    rewritten_query = RewrittenQuery(
        original_query=user_query,
        rewritten_query=user_query,
        language=language or "en",
        detected_language=language or "en",
        confidence=1.0,
        notes=["Blocked by prompt-injection guard."],
    )
    query_plan = QueryPlan(
        original_query=user_query,
        rewritten_query=user_query,
        language=language or "en",
        intent="security_blocked",
        confidence=1.0,
        reasoning_short="The query attempted to override system rules or expose secrets.",
    )
    execution_plan = ExecutionPlan(query_plan=query_plan, tool_calls=[], requires_tool_chain=False, confidence=1.0)
    validation = ValidationResult(
        is_valid=False,
        confidence=1.0,
        issues=["Query blocked by prompt-injection guard."],
        warnings=[warning],
        requires_retry=False,
    )
    final_response = FinalResponse(
        answer="I cannot follow instructions that try to override system rules, reveal secrets, or bypass safety controls. Please ask a normal question about your uploaded financial data or documents.",
        warnings=[warning],
        confidence=1.0,
        metadata={"security_blocked": True},
    )
    return {
        "trace_id": None,
        "execution_time_ms": 0.0,
        "confidence_scores": {"security": 1.0},
        "rewritten_query": rewritten_query,
        "query_plan": query_plan,
        "source_selection": SourceSelection(confidence=0.0),
        "execution_plan": execution_plan,
        "tool_results": [],
        "validation": validation,
        "final_response": final_response,
    }


def run_query_pipeline(user_query: str, language_preference: Optional[str], top_k: int) -> Dict[str, Any]:
    get_rate_limiter().check(
        current_user_id() or st.session_state.get("session_id") or "anonymous",
        "query",
        settings.query_rate_limit_per_minute,
    )
    recorder = get_trace_recorder()
    span = recorder.start_trace(
        "query_pipeline",
        {
            "session_id": st.session_state.session_id,
            "user_id": current_user_id(),
            "query_length": len(user_query or ""),
            "language_preference": language_preference or "auto",
            "uploaded_source_count": len(uploaded_source_payloads()),
        },
    )
    total_started_at = time.perf_counter()
    try:
        detected_language = detect_language(user_query)
        recorder.record_event(span.trace_id, "language_detected", {"language": detected_language})
        prompt_risk = PromptInjectionGuard().assess_user_query(user_query)
        security_warnings = [prompt_risk.warning()] if prompt_risk.is_suspicious else []
        if prompt_risk.should_block:
            recorder.record_event(
                span.trace_id,
                "prompt_injection_blocked",
                {"risk_score": prompt_risk.risk_score, "reasons": prompt_risk.reasons},
            )
            recorder.end_trace(span, status="blocked", metadata={"security": "prompt_injection"})
            return build_security_blocked_pipeline_result(user_query, language_preference or detected_language, prompt_risk.warning())
        if prompt_risk.is_suspicious:
            recorder.record_event(
                span.trace_id,
                "prompt_injection_warning",
                {"risk_score": prompt_risk.risk_score, "reasons": prompt_risk.reasons},
            )

        rewrite_language = language_preference or detected_language
        deterministic_summary_planning = is_simple_single_document_summary(user_query)
        planning_client = GeminiClient(api_key="", client=None) if deterministic_summary_planning else None
        if deterministic_summary_planning:
            recorder.record_event(
                span.trace_id,
                "llm_budget_optimized",
                {"reason": "obvious_single_document_summary", "reserved_for": "final_narration"},
            )
        rewritten_query = QueryRewriterAgent(gemini_client=planning_client).rewrite(user_query, language=rewrite_language)
        recorder.record_event(
            span.trace_id,
            "query_rewritten",
            {"confidence": rewritten_query.confidence, "language": rewritten_query.language},
        )

        query_plan = QueryPlannerAgent(gemini_client=planning_client).plan(
            user_query,
            rewritten_query,
            available_sources=uploaded_source_payloads(),
            table_profiles=table_profile_models(),
        )
        if language_preference:
            query_plan.language = language_preference
        recorder.record_event(
            span.trace_id,
            "query_planned",
            {"intent": query_plan.intent, "confidence": query_plan.confidence},
        )

        source_selection = SourceSelector().select_source(
            query_plan,
            uploaded_source_payloads(),
            table_profiles=table_profile_models(),
            document_metadata=document_metadata_payloads(),
        )
        selected_for_tools = source_selection if source_selection.selected_source_id else None
        recorder.record_event(
            span.trace_id,
            "source_selected",
            {
                "selected_source_id": source_selection.selected_source_id,
                "source_type": source_selection.source_type,
                "confidence": source_selection.confidence,
            },
        )

        tool_manager = get_tool_manager()
        execution_plan = ToolPlannerAgent(tool_manager.get_registry()).create_execution_plan(query_plan, selected_for_tools)
        execution_plan = hydrate_execution_plan(execution_plan, selected_for_tools, top_k)
        for tool_call in execution_plan.tool_calls:
            tool_call.input_payload["trace_id"] = span.trace_id
        recorder.record_event(
            span.trace_id,
            "tools_planned",
            {
                "tool_names": [tool_call.tool_name for tool_call in execution_plan.tool_calls],
                "confidence": execution_plan.confidence,
            },
        )

        tool_results = ToolChainExecutor(tool_manager.get_registry()).execute(execution_plan)
        recorder.record_event(
            span.trace_id,
            "tools_executed",
            {
                "tool_count": len(tool_results),
                "successful_tool_count": len([result for result in tool_results if result.success]),
            },
        )

        validation = ValidatorAgent().validate(query_plan, execution_plan, tool_results, selected_for_tools)
        validation.warnings = list(validation.warnings or []) + security_warnings
        recorder.record_event(
            span.trace_id,
            "validated",
            {"is_valid": validation.is_valid, "confidence": validation.confidence},
        )

        final_response = ResponseNarrator().narrate(
            original_query=user_query,
            rewritten_query=rewritten_query.rewritten_query,
            query_plan=query_plan,
            source_selection=selected_for_tools,
            execution_plan=execution_plan,
            tool_results=tool_results,
            validation_result=validation,
            language_preference=language_preference or detected_language,
        )
        recorder.record_event(
            span.trace_id,
            "response_narrated",
            {
                "used_gemini": bool(final_response.metadata.get("used_gemini")),
                "narration_mode": final_response.metadata.get("narration_mode"),
                "gemini_fallback_used": bool(final_response.metadata.get("gemini_fallback_used")),
                "gemini_error_type": final_response.metadata.get("gemini_error_type"),
            },
        )
        execution_time_ms = round((time.perf_counter() - total_started_at) * 1000.0, 4)
        confidence_scores = {
            "rewriter": rewritten_query.confidence,
            "planner": query_plan.confidence,
            "source_selector": source_selection.confidence,
            "tool_planner": execution_plan.confidence,
            "validator": validation.confidence,
            "final_response": final_response.confidence,
        }
        recorder.end_trace(
            span,
            status="success",
            metadata={
                "execution_time_ms": execution_time_ms,
                "intent": query_plan.intent,
                "confidence_scores": confidence_scores,
            },
        )
        return {
            "trace_id": span.trace_id,
            "execution_time_ms": execution_time_ms,
            "confidence_scores": confidence_scores,
            "rewritten_query": rewritten_query,
            "query_plan": query_plan,
            "source_selection": source_selection,
            "execution_plan": execution_plan,
            "tool_results": tool_results,
            "validation": validation,
            "final_response": final_response,
        }
    except Exception as exc:
        recorder.record_error(span.trace_id, exc, {"stage": "run_query_pipeline"})
        recorder.end_trace(span, status="error")
        raise


def append_chat_turn(user_query: str, pipeline_result: Dict[str, Any]) -> None:
    final_response = pipeline_result["final_response"]
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.session_state.messages.append({"role": "assistant", "content": final_response.answer, "response": final_response})
    st.session_state.latest_response = final_response
    record = ChatHistoryRecord(
        session_id=st.session_state.session_id,
        user_query=user_query,
        rewritten_query=pipeline_result["rewritten_query"],
        query_plan=pipeline_result["query_plan"],
        selected_source=dump_model(pipeline_result["source_selection"]),
        selected_tools=[tool_call.tool_name for tool_call in pipeline_result["execution_plan"].tool_calls],
        execution_plan=pipeline_result["execution_plan"],
        execution_time_ms=float(pipeline_result.get("execution_time_ms", 0.0)),
        confidence_scores=pipeline_result.get("confidence_scores", {}),
        tool_results=pipeline_result["tool_results"],
        final_answer=final_response.answer,
        citations=final_response.citations,
        warnings=final_response.warnings,
        errors=[
            {"tool_name": result.tool_name, "error_msg": result.error_msg}
            for result in pipeline_result["tool_results"]
            if not result.success and result.error_msg
        ],
        document_source_ids=sorted(
            {
                source_id
                for source_id in [
                    pipeline_result["source_selection"].selected_source_id,
                    *[citation.source_id for citation in final_response.citations],
                ]
                if source_id
            }
        ),
        trace_id=pipeline_result.get("trace_id"),
        metadata={"final_response_metadata": final_response.metadata},
    )
    record.metadata["user_id"] = current_user_id()
    record.metadata["user_email"] = (current_user() or {}).get("email")
    st.session_state.history_records.append(record)
    get_history_store().save_record(record)
    log_query(
        user_query,
        {"intent": pipeline_result["query_plan"].intent, "confidence": final_response.confidence, "user_id": current_user_id()},
    )


def render_sidebar() -> Dict[str, Any]:
    with st.sidebar:
        user = current_user() or {}
        st.header("Account")
        st.caption("{0} ({1})".format(user.get("display_name") or "User", user.get("email") or "unknown"))
        st.caption("Role: {0}".format(user.get("role") or "user"))
        if st.button("Logout"):
            logout_current_user()
            st.rerun()

        st.header("Settings")
        api_status = "available" if settings.gemini_api_key_available else "missing"
        if settings.gemini_api_key_available:
            st.success("Gemini API key: {0}".format(api_status))
        else:
            st.warning("Gemini API key: {0}".format(api_status))

        configured_language_options = language_options(settings.supported_languages)
        language_label = st.selectbox(
            "Response language",
            [label for label, _ in configured_language_options],
            index=0,
        )
        top_k = st.slider("Document retrieval depth", min_value=1, max_value=10, value=DEFAULT_TOP_K)
        st.caption("Embedding model: {0}".format(settings.default_embedding_model))
        st.caption("Gemini model: {0}".format(settings.default_gemini_model))

        with st.expander("Configured paths", expanded=False):
            st.write(
                {
                    "uploads": str(settings.upload_dir),
                    "chroma": str(settings.chroma_dir),
                    "history": str(settings.history_dir),
                    "logs": str(settings.logs_dir),
                }
            )

        if st.button("Clear chat history"):
            st.session_state.messages = []
            st.session_state.history_records = []
            st.session_state.latest_response = None
            st.rerun()

        markdown = build_chat_history_markdown(st.session_state.history_records)
        st.download_button(
            "Export chat history",
            data=markdown,
            file_name="financial_chat_history.md",
            mime="text/markdown",
            disabled=not st.session_state.history_records,
        )

    return {"language": language_preference_value(language_label), "top_k": top_k}


def render_ingestion_panel() -> None:
    process_completed_ingestion_jobs()
    st.subheader("Sources")
    upload_col, url_col = st.columns([2, 1])
    with upload_col:
        uploaded_files = st.file_uploader(
            "Upload CSV, Excel, PDF, DOCX, TXT, or HTML",
            type=sorted(settings.allowed_file_extensions),
            accept_multiple_files=True,
        )
        if st.button("Ingest uploaded files", type="primary", disabled=not uploaded_files):
            with st.spinner("Validating uploads and queueing ingestion jobs..."):
                save_and_ingest_uploaded_files(uploaded_files)
    with url_col:
        url = st.text_input("Add a URL", placeholder="https://example.com/report")
        if st.button("Ingest URL"):
            with st.spinner("Queueing URL ingestion job..."):
                add_url_source(url)

    render_ingestion_jobs()

    for event in st.session_state.ingestion_events[-5:]:
        if event["level"] == "success":
            st.success(event["message"])
        elif event["level"] == "warning":
            st.warning(event["message"])
        else:
            st.error(event["message"])

    render_source_list()
    render_table_profile_preview()


def render_ingestion_jobs() -> None:
    jobs = []
    for job_id in (st.session_state.get("active_ingestion_jobs") or {}).keys():
        state = get_job_manager().get_job(job_id)
        if state is not None:
            jobs.append(state)
    if not jobs:
        return
    st.markdown("#### Ingestion Jobs")
    for state in jobs:
        label = "{0} ({1})".format(state.metadata.get("filename") or state.source_id or state.job_id, state.status)
        st.progress(state.progress, text=label)
    if st.button("Refresh ingestion status"):
        st.rerun()


def render_source_list() -> None:
    sources = known_source_payloads()
    if not sources:
        st.info("No sources ingested yet. You can still ask general finance questions.")
        return

    st.markdown("#### Uploaded Source List")
    rows = []
    for source in sources:
        rows.append(
            {
                "source_id": source.get("source_id"),
                "filename": source.get("filename"),
                "type": source.get("file_type"),
                "category": source_category(source),
                "rows": source.get("metadata", {}).get("row_count"),
                "chunks": source.get("metadata", {}).get("chunk_count"),
                "status": source.get("status"),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    pdf_sources = [source for source in sources if str(source.get("file_type") or "").lower() == "pdf"]
    if pdf_sources:
        pdf_by_id = {str(source.get("source_id")): source for source in pdf_sources}
        selected_pdf_id = st.selectbox(
            "PDF to reprocess",
            options=list(pdf_by_id),
            format_func=lambda source_id: str(pdf_by_id[source_id].get("filename") or source_id),
            key="pdf_reprocess_source_id",
        )
        active_source_ids = set((st.session_state.get("active_ingestion_jobs") or {}).values())
        is_active = selected_pdf_id in active_source_ids
        if st.button("Reprocess PDF", disabled=is_active, key="reprocess_pdf_button"):
            if reprocess_pdf_source(selected_pdf_id):
                st.rerun()
        if is_active:
            st.caption("This PDF is currently being processed.")


def render_table_profile_preview() -> None:
    if not st.session_state.table_profiles:
        return
    st.markdown("#### Table Profile Preview")
    for source_id, profile_payload in st.session_state.table_profiles.items():
        profile = TableProfile(**profile_payload)
        with st.expander("{0} ({1} rows x {2} columns)".format(profile.filename, profile.shape[0], profile.shape[1])):
            st.write(profile.semantic_summary)
            left, right = st.columns(2)
            with left:
                st.write("Metric candidates")
                st.write(profile.metric_candidate_columns)
                st.write("Entity candidates")
                st.write(profile.entity_candidate_columns)
            with right:
                st.write("Result/status candidates")
                st.write(profile.result_candidate_columns)
                st.write("Numeric columns")
                st.write(profile.numeric_columns)
            preview_df = st.session_state.dataframes.get(source_id)
            if isinstance(preview_df, pd.DataFrame):
                st.dataframe(preview_df.head(10), use_container_width=True)
            benchmark_cases = st.session_state.table_benchmarks.get(source_id) or []
            if benchmark_cases:
                benchmark_rows = [
                    {"category": item.get("category"), "question": item.get("query")}
                    for item in benchmark_cases[:16]
                ]
                st.write("Auto-generated difficult benchmark questions")
                st.dataframe(pd.DataFrame(benchmark_rows), use_container_width=True, hide_index=True)


def render_response(response: FinalResponse, stream_answer: bool = False) -> None:
    narration_mode = (response.metadata or {}).get("narration_mode")
    if narration_mode == "gemini":
        st.caption("AI narration: Gemini")
    elif narration_mode == "deterministic_fallback":
        st.warning("AI narration fallback is active; this answer was not rewritten by Gemini.")
    answer = response.answer or "No final answer was generated."
    if stream_answer and hasattr(st, "write_stream"):
        st.write_stream(stream_text_chunks(answer))
    else:
        st.markdown(answer)
    response_key = str((response.metadata or {}).get("trace_id") or id(response))
    st.download_button(
        "Download answer",
        data=answer,
        file_name="financial_answer.md",
        mime="text/markdown",
        key="answer_download_{0}".format(response_key),
    )
    table_df = dataframe_from_table(response.table)
    if table_df is not None:
        st.dataframe(table_df, use_container_width=True, hide_index=True)
        csv_payload = table_export_csv(table_df)
        if csv_payload is not None:
            st.download_button(
                "Download table (CSV)",
                data=csv_payload,
                file_name="financial_result.csv",
                mime="text/csv",
                key="table_download_{0}".format(response_key),
            )
    if response.chart is not None:
        charts = list(response.chart) if isinstance(response.chart, (list, tuple)) else [response.chart]
        for chart_index, chart in enumerate(charts):
            try:
                st.plotly_chart(
                    chart,
                    use_container_width=True,
                    key="response_chart_{0}_{1}".format(id(response), chart_index),
                )
                chart_html = chart_export_html(chart)
                if chart_html:
                    st.download_button(
                        "Download chart {0} (HTML)".format(chart_index + 1),
                        data=chart_html,
                        file_name="financial_chart_{0}.html".format(chart_index + 1),
                        mime="text/html",
                        key="chart_download_{0}_{1}".format(response_key, chart_index),
                    )
            except Exception:
                st.warning("Chart object could not be rendered in the UI.")
    if response.citations:
        with st.expander("Citations", expanded=True):
            for index, citation in enumerate(response.citations, start=1):
                st.markdown("**[{0}] {1}**".format(index, citation_label(citation)))
                if citation.text_snippet:
                    st.caption(citation.text_snippet)
    if response.warnings:
        with st.expander("Warnings", expanded=True):
            for warning in response.warnings:
                st.warning(warning)
    st.caption("Confidence: {0:.2f}".format(response.confidence))


def render_chat(settings_payload: Dict[str, Any]) -> None:
    st.subheader("Ask a Question")
    st.caption("Ask in English, Hinglish, or incomplete business language. The system will plan and call tools autonomously.")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant" and message.get("response") is not None:
                render_response(message["response"])
            else:
                st.markdown(message["content"])

    prompt = st.chat_input("Ask about uploaded data, documents, URLs, charts, or finance concepts")
    if not prompt:
        return

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Understanding query, planning tools, validating output..."):
            try:
                pipeline_result = run_query_pipeline(prompt, settings_payload["language"], int(settings_payload["top_k"]))
                append_chat_turn(prompt, pipeline_result)
                render_response(pipeline_result["final_response"], stream_answer=True)
            except Exception as exc:
                log_error(exc, {"stage": "render_chat"})
                fallback = FinalResponse(
                    answer="I hit a recoverable UI error while processing that request. Please try again or rephrase the question.",
                    warnings=["UI recovered from error: {0}".format(str(exc))],
                    confidence=0.0,
                    metadata={"error_type": exc.__class__.__name__},
                )
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.session_state.messages.append({"role": "assistant", "content": fallback.answer, "response": fallback})
                render_response(fallback)


def render_chat_history_sidebar_preview() -> None:
    with st.sidebar.expander("Chat history", expanded=False):
        if not st.session_state.history_records:
            st.caption("No turns yet.")
            return
        for index, record in enumerate(st.session_state.history_records[-8:], start=1):
            st.markdown("**Turn {0}**".format(index))
            st.caption(record.user_query)
            st.write(record.final_answer[:250])


def main() -> None:
    initialize_session_state()
    if not render_auth_gate():
        return

    st.title("Financial Intelligence Chatbot")
    st.caption("Production-grade financial document and data intelligence assistant.")

    sidebar_settings = render_sidebar()
    render_chat_history_sidebar_preview()

    render_ingestion_panel()
    st.divider()
    render_chat(sidebar_settings)


if __name__ == "__main__":
    main()
