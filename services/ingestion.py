"""Document ingestion: chunk, embed, index into ChromaDB (vector store with hybrid search)."""
import asyncio
import logging
from typing import Any, Dict, List, Tuple  # Tuple for (page_num, text)

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
        out = {
            "doc_id": doc["doc_id"],
            "text": prefix + doc["text"],
            "embedding": doc["embedding"],
            "document_name": doc["document_name"],
        }
        if doc.get("page") is not None:
            out["page"] = doc["page"]
        prepared.append(out)
    return prepared


def bulk_index_documents(documents: List[Dict[str, Any]]) -> Tuple[int, List[Any]]:
    return vs_add_documents(_prepare_documents(documents))


def delete_documents_by_document_name(document_name: str) -> Dict[str, Any]:
    return vs_delete_by_name(document_name)


async def process_and_index_document_with_pages(
    pages: List[Tuple[int, str]], document_name: str
) -> Tuple[int, List[Any]]:
    """Chunk each page's text, assign page number to chunks, embed and index. Enables citations with document + page."""
    all_chunks: List[Tuple[str, int]] = []  # (chunk_text, page_num)
    for page_num, text in pages:
        if not text.strip():
            continue
        for chunk in chunk_text(text):
            all_chunks.append((chunk, page_num))
    if not all_chunks:
        return 0, []
    chunks_only = [c for c, _ in all_chunks]
    loop = asyncio.get_event_loop()
    embeddings = await loop.run_in_executor(None, lambda: generate_embeddings(chunks_only))
    documents = [
        {
            "doc_id": f"{document_name}_{i}",
            "text": chunk,
            "embedding": emb,
            "document_name": document_name,
            "page": page_num,
        }
        for i, ((chunk, page_num), emb) in enumerate(zip(all_chunks, embeddings))
    ]
    return bulk_index_documents(documents)


async def process_and_index_document(text: str, document_name: str) -> Tuple[int, List[Any]]:
    """Chunk full text and index (e.g. for notebooks). All chunks get page=1 for citation."""
    return await process_and_index_document_with_pages([(1, text)], document_name)
