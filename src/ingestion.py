import logging
from typing import Any, Dict, List, Tuple

from src.constants import ASSYMETRIC_EMBEDDING
from src.faiss_store import add_documents as faiss_add_documents
from src.faiss_store import create_index as faiss_create_index
from src.faiss_store import delete_documents_by_document_name as faiss_delete_by_name
from src.utils import setup_logging

# Initialize logger
setup_logging()
logger = logging.getLogger(__name__)


def create_index() -> None:
    """Ensures the FAISS vector store index exists."""
    faiss_create_index()


def bulk_index_documents(documents: List[Dict[str, Any]]) -> Tuple[int, List[Any]]:
    """
    Indexes multiple document chunks into the FAISS vector store.

    Args:
        documents: List of dicts with 'doc_id', 'text', 'embedding', and 'document_name'.

    Returns:
        Tuple of (number of indexed documents, list of errors).
    """
    prepared = []
    for doc in documents:
        text = doc["text"]
        if ASSYMETRIC_EMBEDDING:
            text = f"passage: {text}"
        prepared.append({
            "doc_id": doc["doc_id"],
            "text": text,
            "embedding": doc["embedding"],
            "document_name": doc["document_name"],
        })
    return faiss_add_documents(prepared)


def delete_documents_by_document_name(document_name: str) -> Dict[str, Any]:
    """
    Deletes all chunks for the given document from the FAISS store.

    Args:
        document_name: Name of the document to delete.

    Returns:
        Dict with 'deleted' count.
    """
    return faiss_delete_by_name(document_name)
