"""FAISS vector store: index and metadata on disk."""
import logging
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np

from core.config import settings
from core.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

INDEX_FILE = f"{settings.faiss_index_path}.faiss"
METADATA_FILE = f"{settings.faiss_index_path}_metadata.pkl"


def _ensure_index_dir() -> None:
    d = os.path.dirname(INDEX_FILE)
    if d:
        os.makedirs(d, exist_ok=True)


def _load_index_and_metadata() -> Tuple[Optional[faiss.Index], List[Dict[str, Any]]]:
    if not os.path.exists(INDEX_FILE) or not os.path.exists(METADATA_FILE):
        return None, []
    try:
        index = faiss.read_index(INDEX_FILE)
        with open(METADATA_FILE, "rb") as f:
            metadata = pickle.load(f)
        return index, metadata
    except Exception as e:
        logger.warning("Could not load FAISS index: %s", e)
        return None, []


def _save_index_and_metadata(index: faiss.Index, metadata: List[Dict[str, Any]]) -> None:
    _ensure_index_dir()
    faiss.write_index(index, INDEX_FILE)
    with open(METADATA_FILE, "wb") as f:
        pickle.dump(metadata, f)


def create_index() -> None:
    _ensure_index_dir()
    index, metadata = _load_index_and_metadata()
    if index is None:
        index = faiss.IndexFlatL2(settings.embedding_dimension)
        metadata = []
        _save_index_and_metadata(index, metadata)
        logger.info("Created new FAISS index.")


def add_documents(documents: List[Dict[str, Any]]) -> Tuple[int, List[Any]]:
    if not documents:
        return 0, []
    index, metadata = _load_index_and_metadata()
    if index is None:
        index = faiss.IndexFlatL2(settings.embedding_dimension)
        metadata = []
    vectors = []
    for doc in documents:
        emb = doc["embedding"]
        emb_list = emb.tolist() if hasattr(emb, "tolist") else list(emb)
        vectors.append(emb_list)
        metadata.append({
            "doc_id": doc["doc_id"],
            "text": doc["text"],
            "document_name": doc["document_name"],
            "embedding": emb_list,
        })
    matrix = np.array(vectors, dtype=np.float32)
    index.add(matrix)
    _save_index_and_metadata(index, metadata)
    logger.info("Added %s documents to FAISS.", len(documents))
    return len(documents), []


def delete_documents_by_document_name(document_name: str) -> Dict[str, Any]:
    index, metadata = _load_index_and_metadata()
    if not metadata:
        return {"deleted": 0}
    kept = [m for m in metadata if m["document_name"] != document_name]
    removed = len(metadata) - len(kept)
    if removed == 0:
        return {"deleted": 0}
    if not kept:
        new_index = faiss.IndexFlatL2(settings.embedding_dimension)
        _save_index_and_metadata(new_index, [])
    else:
        vectors = np.array([m["embedding"] for m in kept], dtype=np.float32)
        new_index = faiss.IndexFlatL2(settings.embedding_dimension)
        new_index.add(vectors)
        _save_index_and_metadata(new_index, kept)
    logger.info("Deleted %s chunks for document '%s'.", removed, document_name)
    return {"deleted": removed}


def vector_search(query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
    index, metadata = _load_index_and_metadata()
    if index is None or index.ntotal == 0:
        return []
    top_k = min(top_k, index.ntotal)
    query = np.array([query_embedding], dtype=np.float32)
    _, indices = index.search(query, top_k)
    return [
        {"_source": {"text": metadata[idx]["text"], "document_name": metadata[idx]["document_name"]}}
        for idx in indices[0]
        if idx >= 0
    ]


def list_document_names() -> List[str]:
    _, metadata = _load_index_and_metadata()
    return sorted({m["document_name"] for m in metadata})
