from __future__ import annotations

import ipaddress
from typing import Dict, List, Optional
from urllib.parse import urlparse
from uuid import uuid4

import requests
from bs4 import BeautifulSoup

from src.config import settings
from src.models.document import DocumentChunkSource
from src.reliability.policies import RetryPolicy, execute_with_retries
from src.utils.errors import IngestionError
from src.utils.logging import log_event
from src.utils.security import sanitize_filename


DEFAULT_TIMEOUT_SECONDS = 10
REMOVE_TAGS = ["script", "style", "nav", "header", "footer", "aside", "noscript", "svg"]
BLOCKED_HOSTNAMES = {"localhost", "metadata.google.internal"}


def _validate_url(url: str) -> None:
    parsed = urlparse(url or "")
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise IngestionError("Invalid URL.", metadata={"url": url})
    if settings.block_private_urls and _is_private_or_local_host(parsed.hostname or ""):
        raise IngestionError("Private or local network URLs are not allowed.", metadata={"url": url})


def _is_private_or_local_host(hostname: str) -> bool:
    normalized = (hostname or "").strip().lower().rstrip(".")
    if not normalized:
        return True
    if normalized in BLOCKED_HOSTNAMES or normalized.endswith(".localhost"):
        return True
    try:
        address = ipaddress.ip_address(normalized)
    except ValueError:
        return False
    return (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_reserved
        or address.is_unspecified
        or address.is_multicast
    )


def _clean_html(html: str) -> Dict[str, str]:
    soup = BeautifulSoup(html or "", "html.parser")
    for tag_name in REMOVE_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()

    text_parts = [part.strip() for part in soup.get_text(separator="\n").splitlines() if part.strip()]
    content = "\n".join(text_parts)
    return {"title": title, "content": content}


def load_url(url: str, source_id: Optional[str] = None, timeout: Optional[float] = None) -> List[DocumentChunkSource]:
    _validate_url(url)
    source_id = source_id or uuid4().hex
    timeout_seconds = float(timeout if timeout is not None else settings.url_timeout_seconds or DEFAULT_TIMEOUT_SECONDS)
    retry_policy = RetryPolicy(
        max_attempts=settings.url_retry_max_attempts,
        backoff_seconds=settings.retry_backoff_seconds,
    )

    try:
        def _request() -> requests.Response:
            response = requests.get(
                url,
                timeout=timeout_seconds,
                headers={"User-Agent": "financial-intelligence-chatbot/1.0"},
            )
            response.raise_for_status()
            return response

        response = execute_with_retries(
            _request,
            retry_policy=retry_policy,
            retry_exceptions=(requests.RequestException,),
            on_retry=lambda attempt, exc, delay: log_event(
                "url_retry",
                {
                    "url": url,
                    "attempt": attempt,
                    "delay_seconds": delay,
                    "error_type": exc.__class__.__name__,
                },
                level="warning",
            ),
        )
    except requests.Timeout as exc:
        raise IngestionError("URL request timed out.", metadata={"url": url, "error": str(exc)})
    except requests.RequestException as exc:
        raise IngestionError("Failed to load URL.", metadata={"url": url, "error": str(exc)})

    cleaned = _clean_html(response.text)
    parsed = urlparse(url)
    title = cleaned["title"] or parsed.netloc
    filename = sanitize_filename(title) or sanitize_filename(parsed.netloc)

    metadata = {
        "url": url,
        "title": title,
        "status_code": response.status_code,
        "domain": parsed.netloc,
    }

    return [
        DocumentChunkSource(
            source_id=source_id,
            filename=filename,
            source_type="url",
            content=cleaned["content"],
            metadata=metadata,
        )
    ]
