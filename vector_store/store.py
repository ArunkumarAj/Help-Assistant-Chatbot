"""ChromaDB vector store with hybrid search (dense + BM25, RRF). Replaces FAISS for better retrieval."""
import logging
import os
import pickle
import re
from typing import Any, Dict, List, Optional, Tuple

from core.config import settings
from core.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

COLLECTION_NAME = "rag_knowledge_base"


def _chroma_path():
    p = getattr(settings, "vector_store_path", None)
    if p:
        return str(p) if not isinstance(p, str) else p
    return os.environ.get("CHROMA_PERSIST_DIR", str(settings.data_dir / "chroma_db"))


CHROMA_PERSIST_DIR = _chroma_path()
BM25_INDEX_FILE = os.path.join(os.path.dirname(CHROMA_PERSIST_DIR) or ".", "bm25_index.pkl")
RRF_K = 60  # RRF constant (standard choice)


def _tokenize(text: str) -> List[str]:
    """Simple tokenizer: lowercase, alphanumeric tokens."""
    text = (text or "").lower()
    tokens = re.findall(r"[a-z0-9]+", text)
    return tokens


def _get_client():
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
    return chromadb.PersistentClient(
        path=CHROMA_PERSIST_DIR,
        settings=ChromaSettings(anonymized_telemetry=False),
    )


def _get_collection():
    client = _get_client()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "RAG knowledge base with dense vectors"},
    )


def _load_bm25_state() -> Tuple[Optional[Any], List[str], List[str]]:
    """Load BM25 index, doc_ids, and tokenized corpus. Returns (bm25_index, doc_ids, corpus_tokens)."""
    if not os.path.exists(BM25_INDEX_FILE):
        return None, [], []
    try:
        with open(BM25_INDEX_FILE, "rb") as f:
            state = pickle.load(f)
        return state.get("bm25"), state.get("doc_ids", []), state.get("corpus_tokens", [])
    except Exception as e:
        logger.warning("Could not load BM25 index: %s", e)
        return None, [], []


def _save_bm25_state(bm25_index: Any, doc_ids: List[str], corpus_tokens: List[List[str]]) -> None:
    os.makedirs(os.path.dirname(BM25_INDEX_FILE) or ".", exist_ok=True)
    with open(BM25_INDEX_FILE, "wb") as f:
        pickle.dump({
            "bm25": bm25_index,
            "doc_ids": doc_ids,
            "corpus_tokens": corpus_tokens,
        }, f)


def create_index() -> None:
    """Ensure Chroma collection and BM25 state exist (idempotent)."""
    _get_collection()
    try:
        from rank_bm25 import BM25Okapi
        _, doc_ids, corpus_tokens = _load_bm25_state()
        if doc_ids and corpus_tokens:
            bm25 = BM25Okapi(corpus_tokens)
            _save_bm25_state(bm25, doc_ids, corpus_tokens)
    except Exception as e:
        logger.debug("BM25 init: %s", e)
    logger.info("ChromaDB vector store ready (hybrid: dense + BM25).")


def add_documents(documents: List[Dict[str, Any]]) -> Tuple[int, List[Any]]:
    if not documents:
        return 0, []
    coll = _get_collection()
    ids = []
    embeddings = []
    docs_text = []
    metadatas = []
    for doc in documents:
        emb = doc["embedding"]
        emb_list = emb.tolist() if hasattr(emb, "tolist") else list(emb)
        doc_id = doc.get("doc_id") or f"{doc['document_name']}_{len(ids)}"
        ids.append(doc_id)
        embeddings.append(emb_list)
        text = doc.get("text", "")
        docs_text.append(text)
        meta = {"document_name": doc["document_name"], "text": text}
        if doc.get("page") is not None:
            meta["page"] = int(doc["page"])
        metadatas.append(meta)
    coll.add(ids=ids, embeddings=embeddings, documents=docs_text, metadatas=metadatas)
    # Update BM25 index: load existing, append new, save
    try:
        from rank_bm25 import BM25Okapi
        bm25_old, doc_ids_old, corpus_old = _load_bm25_state()
        new_tokens = [_tokenize(t) for t in docs_text]
        doc_ids_new = doc_ids_old + ids
        corpus_new = corpus_old + new_tokens
        bm25_new = BM25Okapi(corpus_new)
        _save_bm25_state(bm25_new, doc_ids_new, corpus_new)
    except Exception as e:
        logger.warning("BM25 update failed: %s", e)
    logger.info("Added %s documents to ChromaDB.", len(documents))
    return len(documents), []


def delete_documents_by_document_name(document_name: str) -> Dict[str, Any]:
    coll = _get_collection()
    all_data = coll.get(include=["metadatas"])
    to_delete = [
        all_data["ids"][i] for i, m in enumerate(all_data.get("metadatas") or [])
        if (m or {}).get("document_name") == document_name
    ]
    if not to_delete:
        return {"deleted": 0}
    coll.delete(ids=to_delete)
    removed = len(to_delete)
    # Rebuild BM25 from current collection
    try:
        all_data = coll.get(include=["documents", "metadatas"])
        if all_data["ids"]:
            doc_ids = all_data["ids"]
            texts = all_data["documents"] or [""] * len(doc_ids)
            corpus_tokens = [_tokenize(t) for t in texts]
            from rank_bm25 import BM25Okapi
            bm25 = BM25Okapi(corpus_tokens)
            _save_bm25_state(bm25, doc_ids, corpus_tokens)
        else:
            _save_bm25_state(None, [], [])
    except Exception as e:
        logger.warning("BM25 rebuild after delete failed: %s", e)
    logger.info("Deleted %s chunks for document '%s'.", removed, document_name)
    return {"deleted": removed}


def _rrf_merge(
    dense_results: List[Dict[str, Any]],
    sparse_results: List[Dict[str, Any]],
    top_k: int,
) -> List[Dict[str, Any]]:
    """Merge dense and sparse rankings using Reciprocal Rank Fusion (RRF). Dedupe by doc id."""
    rrf_scores: Dict[str, float] = {}
    id_to_hit: Dict[str, Dict[str, Any]] = {}
    for rank, hit in enumerate(dense_results):
        doc_id = hit.get("id") or ""
        if not doc_id and hit.get("_source"):
            doc_id = (hit["_source"].get("text") or "")[:100]
        id_to_hit[doc_id] = hit
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (RRF_K + rank + 1)
    for rank, hit in enumerate(sparse_results):
        doc_id = hit.get("id") or ""
        if not doc_id and hit.get("_source"):
            doc_id = (hit["_source"].get("text") or "")[:100]
        id_to_hit[doc_id] = hit
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (RRF_K + rank + 1)
    ordered_ids = sorted(rrf_scores.keys(), key=lambda i: -rrf_scores[i])
    return [id_to_hit[i] for i in ordered_ids[:top_k]]


def hybrid_search(
    query_embedding: List[float],
    query_text: str,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """Hybrid search: dense (Chroma) + sparse (BM25), merged with RRF."""
    coll = _get_collection()
    n_total = coll.count()
    if n_total == 0:
        return []
    top_k = min(top_k, n_total)
    # Dense search
    dense_hits = []
    try:
        result = coll.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k * 2, n_total),
            include=["documents", "metadatas"],
        )
        if result["ids"] and result["ids"][0]:
            for i, doc_id in enumerate(result["ids"][0]):
                doc = (result["documents"] or [[""]])[0][i] if result["documents"] else ""
                meta = (result["metadatas"] or [{}])[0][i] if result["metadatas"] else {}
                src = {"text": doc or meta.get("text", ""), "document_name": meta.get("document_name", "")}
                if meta.get("page") is not None:
                    src["page"] = meta["page"]
                dense_hits.append({"id": doc_id, "_source": src})
    except Exception as e:
        logger.warning("Chroma dense query failed: %s", e)
    # Sparse search (BM25)
    sparse_hits = []
    try:
        bm25, doc_ids, corpus_tokens = _load_bm25_state()
        if bm25 and doc_ids and corpus_tokens:
            q_tokens = _tokenize(query_text)
            if q_tokens:
                scores = bm25.get_scores(q_tokens)
                top_indices = sorted(range(len(scores)), key=lambda i: -scores[i])[: top_k * 2]
                coll_all = coll.get(include=["documents", "metadatas"])
                id_to_meta = {}
                if coll_all["ids"]:
                    for j, doc_id in enumerate(coll_all["ids"]):
                        doc = (coll_all["documents"] or [""])[j] if coll_all["documents"] else ""
                        meta = (coll_all["metadatas"] or [{}])[j] if coll_all["metadatas"] else {}
                        src = {"text": doc or meta.get("text", ""), "document_name": meta.get("document_name", "")}
                        if meta.get("page") is not None:
                            src["page"] = meta["page"]
                        id_to_meta[doc_id] = src
                for idx in top_indices:
                    if scores[idx] <= 0:
                        continue
                    doc_id = doc_ids[idx]
                    if doc_id in id_to_meta:
                        sparse_hits.append({"id": doc_id, "_source": id_to_meta[doc_id]})
    except Exception as e:
        logger.debug("BM25 query failed: %s", e)
    # Merge with RRF
    if dense_hits and sparse_hits:
        return _rrf_merge(dense_hits, sparse_hits, top_k)
    if dense_hits:
        return dense_hits[:top_k]
    if sparse_hits:
        return sparse_hits[:top_k]
    return []


def vector_search(query_embedding: List[float], top_k: int = 5, query_text: Optional[str] = None) -> List[Dict[str, Any]]:
    """Hybrid search by default (dense + BM25). Pass query_text for full hybrid; else dense-only."""
    if query_text:
        return hybrid_search(query_embedding, query_text, top_k)
    # Dense-only fallback
    coll = _get_collection()
    if coll.count() == 0:
        return []
    top_k = min(top_k, coll.count())
    result = coll.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas"],
    )
    if not result["ids"] or not result["ids"][0]:
        return []
    out = []
    for i, doc_id in enumerate(result["ids"][0]):
        doc = (result["documents"] or [[""]])[0][i] if result["documents"] else ""
        meta = (result["metadatas"] or [{}])[0][i] if result["metadatas"] else {}
        src = {"text": doc or meta.get("text", ""), "document_name": meta.get("document_name", "")}
        if meta.get("page") is not None:
            src["page"] = meta["page"]
        out.append({"_source": src})
    return out


def list_document_names() -> List[str]:
    coll = _get_collection()
    try:
        data = coll.get(include=["metadatas"])
        names = set()
        for m in (data.get("metadatas") or []):
            if isinstance(m, dict) and m.get("document_name"):
                names.add(m["document_name"])
        return sorted(names)
    except Exception:
        return []
