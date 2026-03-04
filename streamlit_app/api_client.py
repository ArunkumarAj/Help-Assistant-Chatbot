"""
HTTP client for the FastAPI backend.

Provides: health check, list documents, upload PDF, delete document, and chat.
All functions raise on HTTP errors. Document names are URL-encoded for delete.
"""
import logging
from typing import Any, List
from urllib.parse import quote

import requests

from streamlit_app.config import API_BASE_URL

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _api_url(path: str) -> str:
    """Build full URL for an API path (no trailing slash on base)."""
    return f"{API_BASE_URL.rstrip('/')}{path}"


# -----------------------------------------------------------------------------
# Health
# -----------------------------------------------------------------------------


def health() -> dict:
    """Check backend liveness. Returns {\"status\": \"ok\", \"service\": \"rag-api\"}."""
    response = requests.get(_api_url("/health"), timeout=5)
    response.raise_for_status()
    return response.json()


# -----------------------------------------------------------------------------
# Documents
# -----------------------------------------------------------------------------


def list_documents() -> List[str]:
    """Return list of document names currently in the vector store."""
    response = requests.get(_api_url("/documents"), timeout=30)
    response.raise_for_status()
    return response.json().get("documents", [])


# Large PDFs can take several minutes (extract, chunk, embed, index).
UPLOAD_TIMEOUT_SECONDS = 600  # 10 minutes


def upload_document(file_bytes: bytes, filename: str) -> dict:
    """Upload a PDF. Returns {\"filename\": ..., \"chunks_indexed\": N, \"errors\": [...]}."""
    response = requests.post(
        _api_url("/documents/upload"),
        files={"file": (filename, file_bytes, "application/pdf")},
        timeout=(10, UPLOAD_TIMEOUT_SECONDS),
    )
    response.raise_for_status()
    return response.json()


def delete_document(document_name: str) -> dict:
    """Delete document from vector store and uploaded files. Name is URL-encoded. Returns {\"deleted\": N}."""
    encoded_name = quote(document_name, safe=".")
    response = requests.delete(_api_url(f"/documents/{encoded_name}"), timeout=30)
    response.raise_for_status()
    return response.json()


# -----------------------------------------------------------------------------
# Chat
# -----------------------------------------------------------------------------


def chat(
    query: str,
    use_rag: bool = True,
    num_results: int = 5,
    temperature: float = 0.7,
    chat_history: List[dict] | None = None,
) -> dict:
    """
    Send a chat request. Returns {\"response\": str, \"citations\": [{\"index\", \"document_name\", \"page\", \"doc_id\"}, ...]}.
    """
    payload = {
        "query": query,
        "use_rag": use_rag,
        "num_results": num_results,
        "temperature": temperature,
        "chat_history": chat_history or [],
    }
    response = requests.post(_api_url("/chat"), json=payload, timeout=120)
    response.raise_for_status()
    return response.json()
