"""
Vector store for RAG: ChromaDB (dense vectors) + BM25 (keyword search).

- Documents are stored in a ChromaDB collection with embeddings.
- A separate BM25 index (saved as pickle) gives keyword (sparse) search.
- Hybrid search runs both, then merges rankings using Reciprocal Rank Fusion (RRF).
"""

import logging
import os
import pickle
import re
from typing import Any, Dict, List, Optional, Tuple

from core.config import settings
from core.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Configuration and constants
# -----------------------------------------------------------------------------

COLLECTION_NAME = "rag_knowledge_base"
"""ChromaDB collection that holds all document chunks and their embeddings."""

# RRF constant used when merging dense and sparse rankings (common choice: 60).
RRF_K = 60


def _get_chroma_persist_directory() -> str:
    """Directory where ChromaDB stores its data. Uses settings or env, else default under data_dir."""
    path = getattr(settings, "vector_store_path", None)
    if path is not None:
        return str(path) if not isinstance(path, str) else path
    default = str(settings.data_dir / "chroma_db")
    return os.environ.get("CHROMA_PERSIST_DIR", default)


def _get_bm25_index_file_path() -> str:
    """Path to the pickle file that holds the BM25 index and related lists."""
    chroma_dir = _get_chroma_persist_directory()
    parent = os.path.dirname(chroma_dir) or "."
    return os.path.join(parent, "bm25_index.pkl")


CHROMA_PERSIST_DIR = _get_chroma_persist_directory()
BM25_INDEX_FILE = _get_bm25_index_file_path()


# -----------------------------------------------------------------------------
# Text helpers (for BM25 tokenization)
# -----------------------------------------------------------------------------


def _tokenize_for_bm25(text: str) -> List[str]:
    """Split text into lowercase alphanumeric tokens for BM25 scoring."""
    normalized = (text or "").lower()
    return re.findall(r"[a-z0-9]+", normalized)


# -----------------------------------------------------------------------------
# ChromaDB client and collection
# -----------------------------------------------------------------------------


def _get_chroma_client():
    """Create or reuse a persistent ChromaDB client. Ensures persist directory exists."""
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
    return chromadb.PersistentClient(
        path=CHROMA_PERSIST_DIR,
        settings=ChromaSettings(anonymized_telemetry=False),
    )


def _get_knowledge_base_collection():
    """Get the RAG knowledge-base collection; create it if it does not exist."""
    client = _get_chroma_client()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "RAG knowledge base with dense vectors"},
    )


# Backward compatibility for code that used the old name (e.g. notebooks).
_get_collection = _get_knowledge_base_collection


# -----------------------------------------------------------------------------
# BM25 index: load and save (doc_ids and tokenized corpus stay in sync)
# -----------------------------------------------------------------------------


def _load_bm25_index() -> Tuple[Optional[Any], List[str], List[List[str]]]:
    """
    Load BM25 index from disk.
    Returns:
        (bm25_index, list of document/chunk ids, list of tokenized documents).
    If the file is missing or invalid, returns (None, [], []).
    """
    if not os.path.exists(BM25_INDEX_FILE):
        return None, [], []

    try:
        with open(BM25_INDEX_FILE, "rb") as f:
            state = pickle.load(f)
        bm25 = state.get("bm25")
        doc_ids = state.get("doc_ids", [])
        corpus_tokens = state.get("corpus_tokens", [])
        return bm25, doc_ids, corpus_tokens
    except Exception as e:
        logger.warning("Could not load BM25 index: %s", e)
        return None, [], []


def _save_bm25_index(
    bm25_index: Any,
    doc_ids: List[str],
    corpus_tokens: List[List[str]],
) -> None:
    """Persist BM25 index and its doc_ids and tokenized corpus to disk."""
    parent = os.path.dirname(BM25_INDEX_FILE) or "."
    os.makedirs(parent, exist_ok=True)
    with open(BM25_INDEX_FILE, "wb") as f:
        pickle.dump(
            {
                "bm25": bm25_index,
                "doc_ids": doc_ids,
                "corpus_tokens": corpus_tokens,
            },
            f,
        )


# -----------------------------------------------------------------------------
# Index lifecycle: create / add / delete
# -----------------------------------------------------------------------------


def create_index() -> None:
    """
    Ensure the vector store is ready: Chroma collection exists and BM25 index is valid.
    Safe to call multiple times (idempotent).
    """
    _get_knowledge_base_collection()
    try:
        from rank_bm25 import BM25Okapi

        _, doc_ids, corpus_tokens = _load_bm25_index()
        if doc_ids and corpus_tokens:
            bm25 = BM25Okapi(corpus_tokens)
            _save_bm25_index(bm25, doc_ids, corpus_tokens)
    except Exception as e:
        logger.debug("BM25 init: %s", e)
    logger.info("ChromaDB vector store ready (hybrid: dense + BM25).")


def add_documents(documents: List[Dict[str, Any]]) -> Tuple[int, List[Any]]:
    """
    Add document chunks to ChromaDB and update the BM25 index.
    Each item in `documents` must have: embedding, document_name, text; optional: doc_id, page.
    Returns (number of chunks added, list of errors; currently errors are not populated).
    """
    if not documents:
        return 0, []

    collection = _get_knowledge_base_collection()

    chunk_ids: List[str] = []
    chunk_embeddings: List[List[float]] = []
    chunk_texts: List[str] = []
    chunk_metadatas: List[Dict[str, Any]] = []

    for doc in documents:
        emb = doc["embedding"]
        emb_list = emb.tolist() if hasattr(emb, "tolist") else list(emb)
        doc_id = doc.get("doc_id") or f"{doc['document_name']}_{len(chunk_ids)}"
        text = doc.get("text", "")

        chunk_ids.append(doc_id)
        chunk_embeddings.append(emb_list)
        chunk_texts.append(text)

        meta: Dict[str, Any] = {"document_name": doc["document_name"], "text": text}
        if doc.get("page") is not None:
            meta["page"] = int(doc["page"])
        chunk_metadatas.append(meta)

    collection.add(
        ids=chunk_ids,
        embeddings=chunk_embeddings,
        documents=chunk_texts,
        metadatas=chunk_metadatas,
    )

    # Keep BM25 in sync: append new chunks and rebuild the index.
    try:
        from rank_bm25 import BM25Okapi

        _, existing_doc_ids, existing_corpus = _load_bm25_index()
        new_tokens = [_tokenize_for_bm25(t) for t in chunk_texts]
        all_doc_ids = existing_doc_ids + chunk_ids
        all_corpus = existing_corpus + new_tokens
        bm25_new = BM25Okapi(all_corpus)
        _save_bm25_index(bm25_new, all_doc_ids, all_corpus)
    except Exception as e:
        logger.warning("BM25 update failed: %s", e)

    logger.info("Added %s documents to ChromaDB.", len(documents))
    return len(documents), []


def delete_documents_by_document_name(document_name: str) -> Dict[str, Any]:
    """
    Remove all chunks that belong to the given document name from ChromaDB,
    then rebuild the BM25 index from the remaining chunks.
    Returns {"deleted": count}.
    """
    collection = _get_knowledge_base_collection()
    all_data = collection.get(include=["metadatas"])

    ids_to_delete = []
    for i, meta in enumerate(all_data.get("metadatas") or []):
        name = (meta or {}).get("document_name")
        if name == document_name:
            ids_to_delete.append(all_data["ids"][i])

    if not ids_to_delete:
        return {"deleted": 0}

    collection.delete(ids=ids_to_delete)
    deleted_count = len(ids_to_delete)

    # Rebuild BM25 from whatever is left in the collection.
    try:
        all_data = collection.get(include=["documents", "metadatas"])
        if all_data["ids"]:
            remaining_ids = all_data["ids"]
            documents = all_data["documents"] or [""] * len(remaining_ids)
            corpus_tokens = [_tokenize_for_bm25(t) for t in documents]
            from rank_bm25 import BM25Okapi

            bm25 = BM25Okapi(corpus_tokens)
            _save_bm25_index(bm25, remaining_ids, corpus_tokens)
        else:
            _save_bm25_index(None, [], [])
    except Exception as e:
        logger.warning("BM25 rebuild after delete failed: %s", e)

    logger.info("Deleted %s chunks for document '%s'.", deleted_count, document_name)
    return {"deleted": deleted_count}


# -----------------------------------------------------------------------------
# Search: build hit dict from collection data (shared by dense and sparse)
# -----------------------------------------------------------------------------


def _build_search_hit(doc_id: str, document_text: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Build a single search result hit in the format expected by RAG: id + _source (text, document_name, optional page)."""
    source: Dict[str, Any] = {
        "text": document_text or metadata.get("text", ""),
        "document_name": metadata.get("document_name", ""),
    }
    if metadata.get("page") is not None:
        source["page"] = metadata["page"]
    return {"id": doc_id, "_source": source}


def _build_hits_from_chroma_result(
    ids: List[str],
    documents: Optional[List[str]],
    metadatas: Optional[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Convert Chroma query result (ids, documents, metadatas) into a list of hit dicts."""
    hits = []
    doc_list = documents or [""] * len(ids)
    meta_list = metadatas or [{}] * len(ids)
    for i, doc_id in enumerate(ids):
        doc_text = doc_list[i] if i < len(doc_list) else ""
        meta = meta_list[i] if i < len(meta_list) else {}
        hits.append(_build_search_hit(doc_id, doc_text, meta))
    return hits


def _get_doc_id_for_rrf(hit: Dict[str, Any]) -> str:
    """Get a stable document id from a hit for RRF deduplication."""
    doc_id = hit.get("id") or ""
    if not doc_id and hit.get("_source"):
        text = (hit["_source"].get("text") or "")[:100]
        doc_id = text
    return doc_id


def _rrf_merge(
    dense_hits: List[Dict[str, Any]],
    sparse_hits: List[Dict[str, Any]],
    top_k: int,
) -> List[Dict[str, Any]]:
    """
    Merge dense and sparse rankings using Reciprocal Rank Fusion (RRF).
    Score for each document = sum of 1 / (RRF_K + rank) across both lists.
    Deduplicates by document id and returns the top_k by combined score.
    """
    rrf_scores: Dict[str, float] = {}
    id_to_hit: Dict[str, Dict[str, Any]] = {}

    for rank, hit in enumerate(dense_hits):
        doc_id = _get_doc_id_for_rrf(hit)
        id_to_hit[doc_id] = hit
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (RRF_K + rank + 1)

    for rank, hit in enumerate(sparse_hits):
        doc_id = _get_doc_id_for_rrf(hit)
        id_to_hit[doc_id] = hit
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (RRF_K + rank + 1)

    sorted_ids = sorted(rrf_scores.keys(), key=lambda i: -rrf_scores[i])
    return [id_to_hit[doc_id] for doc_id in sorted_ids[:top_k]]


def _run_dense_search(
    collection: Any,
    query_embedding: List[float],
    n_results: int,
) -> List[Dict[str, Any]]:
    """Run vector (dense) search in ChromaDB and return hits in standard format."""
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas"],
    )
    ids = result.get("ids") or []
    if not ids or not ids[0]:
        return []
    ids = ids[0]
    # Chroma returns documents/metadatas as list-of-lists (one list per query).
    doc_list = result["documents"][0] if result.get("documents") and result["documents"] else [""] * len(ids)
    meta_list = result["metadatas"][0] if result.get("metadatas") and result["metadatas"] else [{}] * len(ids)
    return _build_hits_from_chroma_result(ids, doc_list, meta_list)


def _run_sparse_search(
    collection: Any,
    query_text: str,
    top_k: int,
) -> List[Dict[str, Any]]:
    """Run BM25 (sparse/keyword) search and return hits in standard format."""
    bm25, doc_ids, corpus_tokens = _load_bm25_index()
    if not bm25 or not doc_ids or not corpus_tokens:
        return []

    query_tokens = _tokenize_for_bm25(query_text)
    if not query_tokens:
        return []

    scores = bm25.get_scores(query_tokens)
    num_candidates = min(top_k * 2, len(scores))
    top_indices = sorted(range(len(scores)), key=lambda i: -scores[i])[:num_candidates]

    all_data = collection.get(include=["documents", "metadatas"])
    if not all_data["ids"]:
        return []

    id_to_source: Dict[str, Dict[str, Any]] = {}
    documents = all_data.get("documents") or [""] * len(all_data["ids"])
    metadatas = all_data.get("metadatas") or [{}] * len(all_data["ids"])
    for j, doc_id in enumerate(all_data["ids"]):
        doc_text = documents[j] if j < len(documents) else ""
        meta = metadatas[j] if j < len(metadatas) else {}
        source = {"text": doc_text or meta.get("text", ""), "document_name": meta.get("document_name", "")}
        if meta.get("page") is not None:
            source["page"] = meta["page"]
        id_to_source[doc_id] = source

    hits = []
    for idx in top_indices:
        if scores[idx] <= 0:
            continue
        doc_id = doc_ids[idx]
        if doc_id in id_to_source:
            hits.append({"id": doc_id, "_source": id_to_source[doc_id]})
    return hits


def hybrid_search(
    query_embedding: List[float],
    query_text: str,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Hybrid search: run dense (Chroma) and sparse (BM25) search, then merge with RRF.
    Returns up to top_k hits, each with id and _source (text, document_name, optional page).
    """
    collection = _get_knowledge_base_collection()
    total_chunks = collection.count()
    if total_chunks == 0:
        return []

    n_results = min(top_k * 2, total_chunks)

    dense_hits: List[Dict[str, Any]] = []
    try:
        dense_hits = _run_dense_search(collection, query_embedding, n_results)
    except Exception as e:
        logger.warning("Chroma dense query failed: %s", e)

    sparse_hits: List[Dict[str, Any]] = []
    try:
        sparse_hits = _run_sparse_search(collection, query_text, top_k)
    except Exception as e:
        logger.debug("BM25 query failed: %s", e)

    if dense_hits and sparse_hits:
        return _rrf_merge(dense_hits, sparse_hits, top_k)
    if dense_hits:
        return dense_hits[:top_k]
    if sparse_hits:
        return sparse_hits[:top_k]
    return []


def vector_search(
    query_embedding: List[float],
    top_k: int = 5,
    query_text: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Search the vector store. If query_text is provided, runs hybrid search (dense + BM25).
    Otherwise runs dense-only search. Returns list of hits with _source (text, document_name, optional page).
    """
    if query_text:
        return hybrid_search(query_embedding, query_text, top_k)

    collection = _get_knowledge_base_collection()
    if collection.count() == 0:
        return []
    n_results = min(top_k, collection.count())

    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas"],
    )
    ids = result.get("ids") or []
    if not ids or not ids[0]:
        return []
    ids = ids[0]
    doc_list = result["documents"][0] if result.get("documents") and result["documents"] else [""] * len(ids)
    meta_list = result["metadatas"][0] if result.get("metadatas") and result["metadatas"] else [{}] * len(ids)
    hits = _build_hits_from_chroma_result(ids, doc_list, meta_list)
    return hits


def list_document_names() -> List[str]:
    """Return sorted list of unique document names currently in the vector store."""
    collection = _get_knowledge_base_collection()
    try:
        data = collection.get(include=["metadatas"])
        names = set()
        for meta in data.get("metadatas") or []:
            if isinstance(meta, dict) and meta.get("document_name"):
                names.add(meta["document_name"])
        return sorted(names)
    except Exception:
        return []
