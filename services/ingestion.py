"""Document ingestion: chunk, embed, index into FAISS."""
import asyncio
import logging
from typing import Any, Dict, List, Tuple

from core.config import settings
from core.logging_config import setup_logging
from core.text_utils import chunk_text
from embedding.model import generate_embeddings
from vector_store.store import (
    add_documents as vs_add_documents,
    create_index as vs_create_index,
    delete_documents_by_document_name as vs_delete_by_name,
)

setup_logging()
logger = logging.getLogger(__name__)


def create_index() -> None:
    vs_create_index()


def _prepare_documents(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    prepared = []
    prefix = "passage: " if settings.asymmetric_embedding else ""
    for doc in documents:
        prepared.append({
            "doc_id": doc["doc_id"],
            "text": prefix + doc["text"],
            "embedding": doc["embedding"],
            "document_name": doc["document_name"],
        })
    return prepared


def bulk_index_documents(documents: List[Dict[str, Any]]) -> Tuple[int, List[Any]]:
    return vs_add_documents(_prepare_documents(documents))


def delete_documents_by_document_name(document_name: str) -> Dict[str, Any]:
    return vs_delete_by_name(document_name)


async def process_and_index_document(text: str, document_name: str) -> Tuple[int, List[Any]]:
    """Chunk text, generate embeddings (in thread), bulk index. Async-friendly."""
    chunks = chunk_text(text)
    if not chunks:
        return 0, []
    # Run CPU-bound embedding in thread pool
    loop = asyncio.get_event_loop()
    embeddings = await loop.run_in_executor(None, lambda: generate_embeddings(chunks))
    documents = [
        {
            "doc_id": f"{document_name}_{i}",
            "text": chunk,
            "embedding": emb,
            "document_name": document_name,
        }
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings))
    ]
    return bulk_index_documents(documents)
