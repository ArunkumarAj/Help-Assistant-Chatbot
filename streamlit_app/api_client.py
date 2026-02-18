"""Simple HTTP client for the FastAPI backend."""
import logging
from typing import Any, List

import requests

from streamlit_app.config import API_BASE_URL

logger = logging.getLogger(__name__)


def _url(path: str) -> str:
    return f"{API_BASE_URL.rstrip('/')}{path}"


def health() -> dict:
    r = requests.get(_url("/health"), timeout=5)
    r.raise_for_status()
    return r.json()


def list_documents() -> List[str]:
    r = requests.get(_url("/documents"), timeout=30)
    r.raise_for_status()
    return r.json().get("documents", [])


# Upload can be slow for large PDFs (extract, chunk, embed, index)
UPLOAD_TIMEOUT_SECONDS = 600  # 10 minutes

def upload_document(file_bytes: bytes, filename: str) -> dict:
    r = requests.post(
        _url("/documents/upload"),
        files={"file": (filename, file_bytes, "application/pdf")},
        timeout=(10, UPLOAD_TIMEOUT_SECONDS),  # (connect, read)
    )
    r.raise_for_status()
    return r.json()


def delete_document(document_name: str) -> dict:
    r = requests.delete(_url(f"/documents/{document_name}"), timeout=30)
    r.raise_for_status()
    return r.json()


def chat(
    query: str,
    use_rag: bool = True,
    num_results: int = 5,
    temperature: float = 0.7,
    chat_history: List[dict] | None = None,
) -> str:
    payload = {
        "query": query,
        "use_rag": use_rag,
        "num_results": num_results,
        "temperature": temperature,
        "chat_history": chat_history or [],
    }
    r = requests.post(_url("/chat"), json=payload, timeout=120)
    r.raise_for_status()
    return r.json().get("response", "")
