"""
Document ingestion: chunk, embed, and index into the vector store.

- PDF text is chunked per page; each chunk keeps its page number for citations.
- Embeddings are generated with the shared embedding model; optional "passage: "
  prefix when asymmetric_embedding is enabled.
- Indexing goes to ChromaDB and the BM25 index (see vector_store.store).
"""
import asyncio
import logging
from typing import Any, Dict, List, Tuple

from core.config import settings
from core.logging_config import setup_logging
from core.text_utils import chunk_text
from embedding.model import generate_embeddings
from vector_store.store import (
    add_documents as store_add_documents,
    create_index as store_create_index,
    delete_documents_by_document_name as store_delete_by_document_name,
)

setup_logging()
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Index lifecycle (delegate to vector store)
# -----------------------------------------------------------------------------


def create_index() -> None:
    """Ensure the vector store (ChromaDB + BM25) is ready. Idempotent."""
    store_create_index()


def delete_documents_by_document_name(document_name: str) -> Dict[str, Any]:
    """Remove all chunks for the given document from the vector store. Returns {\"deleted\": count}."""
    return store_delete_by_document_name(document_name)


# -----------------------------------------------------------------------------
# Prepare documents for the store (optional passage prefix, required fields)
# -----------------------------------------------------------------------------


def _prepare_chunks_for_store(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Add optional 'passage: ' prefix to text when asymmetric_embedding is on; ensure required fields exist."""
    prefix = "passage: " if settings.asymmetric_embedding else ""
    prepared = []
    for doc in documents:
        item = {
            "doc_id": doc["doc_id"],
            "text": prefix + doc["text"],
            "embedding": doc["embedding"],
            "document_name": doc["document_name"],
        }
        if doc.get("page") is not None:
            item["page"] = doc["page"]
        prepared.append(item)
    return prepared


def bulk_index_documents(documents: List[Dict[str, Any]]) -> Tuple[int, List[Any]]:
    """Index a list of chunk dicts (doc_id, text, embedding, document_name, optional page). Returns (count, errors)."""
    return store_add_documents(_prepare_chunks_for_store(documents))


# -----------------------------------------------------------------------------
# Process PDF pages: chunk per page, embed, index
# -----------------------------------------------------------------------------


async def process_and_index_document_with_pages(
    pages: List[Tuple[int, str]],
    document_name: str,
) -> Tuple[int, List[Any]]:
    """
    Chunk each page's text, attach page number to each chunk, embed, and index.
    pages: list of (page_number_one_based, page_text).
    Returns (number of chunks indexed, list of errors).
    """
    chunks_with_page: List[Tuple[str, int]] = []
    for page_num, text in pages:
        if not text.strip():
            continue
        for chunk in chunk_text(text):
            chunks_with_page.append((chunk, page_num))

    if not chunks_with_page:
        return 0, []

    chunk_texts_only = [chunk for chunk, _ in chunks_with_page]
    loop = asyncio.get_event_loop()
    embeddings = await loop.run_in_executor(
        None,
        lambda: generate_embeddings(chunk_texts_only),
    )

    documents = [
        {
            "doc_id": f"{document_name}_{i}",
            "text": chunk,
            "embedding": emb,
            "document_name": document_name,
            "page": page_num,
        }
        for i, ((chunk, page_num), emb) in enumerate(zip(chunks_with_page, embeddings))
    ]
    return bulk_index_documents(documents)


async def process_and_index_document(text: str, document_name: str) -> Tuple[int, List[Any]]:
    """
    Chunk full text and index (e.g. for notebooks). All chunks get page=1 for citation.
    Returns (number of chunks indexed, list of errors).
    """
    return await process_and_index_document_with_pages([(1, text)], document_name)
