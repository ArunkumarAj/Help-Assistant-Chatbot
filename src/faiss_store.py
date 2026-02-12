"""
FAISS vector store for document embeddings and metadata.
Persists index and metadata to disk for reuse across sessions.
"""
import logging
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np

from src.constants import EMBEDDING_DIMENSION, FAISS_INDEX_PATH
from src.utils import setup_logging

# Initialize logger
setup_logging()
logger = logging.getLogger(__name__)

INDEX_FILE = f"{FAISS_INDEX_PATH}.faiss"
METADATA_FILE = f"{FAISS_INDEX_PATH}_metadata.pkl"


def _ensure_index_dir() -> None:
    """Ensures the directory for the FAISS index exists."""
    directory = os.path.dirname(INDEX_FILE)
    if directory:
        os.makedirs(directory, exist_ok=True)


def _load_index_and_metadata() -> Tuple[Optional[faiss.Index], List[Dict[str, Any]]]:
    """Loads FAISS index and metadata from disk if they exist."""
    if not os.path.exists(INDEX_FILE) or not os.path.exists(METADATA_FILE):
        return None, []
    try:
        index = faiss.read_index(INDEX_FILE)
        with open(METADATA_FILE, "rb") as f:
            metadata = pickle.load(f)
        logger.info(f"Loaded FAISS index with {index.ntotal} vectors from {INDEX_FILE}.")
        return index, metadata
    except Exception as e:
        logger.warning(f"Could not load FAISS index: {e}. Starting fresh.")
        return None, []


def _save_index_and_metadata(index: faiss.Index, metadata: List[Dict[str, Any]]) -> None:
    """Saves FAISS index and metadata to disk."""
    _ensure_index_dir()
    faiss.write_index(index, INDEX_FILE)
    with open(METADATA_FILE, "wb") as f:
        pickle.dump(metadata, f)
    logger.info(f"Saved FAISS index with {index.ntotal} vectors to {INDEX_FILE}.")


def create_index() -> None:
    """
    Ensures the FAISS index exists. Creates an empty index and metadata if they do not.
    """
    _ensure_index_dir()
    index, metadata = _load_index_and_metadata()
    if index is None:
        index = faiss.IndexFlatL2(EMBEDDING_DIMENSION)
        metadata = []
        _save_index_and_metadata(index, metadata)
        logger.info("Created new FAISS index.")
    else:
        logger.info("FAISS index already exists.")


def add_documents(documents: List[Dict[str, Any]]) -> tuple[int, List[Any]]:
    """
    Adds document chunks to the FAISS index and metadata store.

    Args:
        documents: List of dicts with 'doc_id', 'text', 'embedding', 'document_name'.
                   Embedding can be numpy array or list.

    Returns:
        Tuple of (number of added documents, list of errors).
    """
    if not documents:
        return 0, []

    index, metadata = _load_index_and_metadata()
    if index is None:
        index = faiss.IndexFlatL2(EMBEDDING_DIMENSION)
        metadata = []

    vectors = []
    for doc in documents:
        emb = doc["embedding"]
        if hasattr(emb, "tolist"):
            embedding_list = emb.tolist()
        else:
            embedding_list = list(emb)
        vectors.append(embedding_list)
        metadata.append({
            "doc_id": doc["doc_id"],
            "text": doc["text"],
            "document_name": doc["document_name"],
            "embedding": embedding_list,
        })

    matrix = np.array(vectors, dtype=np.float32)
    index.add(matrix)
    _save_index_and_metadata(index, metadata)
    logger.info(f"Added {len(documents)} documents to FAISS index.")
    return len(documents), []


def delete_documents_by_document_name(document_name: str) -> Dict[str, Any]:
    """
    Removes all chunks belonging to the given document from the index.

    Args:
        document_name: Name of the document to remove.

    Returns:
        Dict with 'deleted' count for compatibility.
    """
    index, metadata = _load_index_and_metadata()
    if index is None or not metadata:
        logger.info("No index or metadata to delete from.")
        return {"deleted": 0}

    kept_metadata = [m for m in metadata if m["document_name"] != document_name]
    removed = len(metadata) - len(kept_metadata)
    if removed == 0:
        logger.info(f"No documents found with name '{document_name}'.")
        return {"deleted": 0}

    if not kept_metadata:
        new_index = faiss.IndexFlatL2(EMBEDDING_DIMENSION)
        _save_index_and_metadata(new_index, [])
        logger.info(f"Deleted all vectors for document '{document_name}'.")
        return {"deleted": removed}

    # Rebuild index from remaining metadata
    vectors = np.array([m["embedding"] for m in kept_metadata], dtype=np.float32)
    new_index = faiss.IndexFlatL2(EMBEDDING_DIMENSION)
    new_index.add(vectors)
    _save_index_and_metadata(new_index, kept_metadata)
    logger.info(f"Deleted {removed} chunks for document '{document_name}'.")
    return {"deleted": removed}


def vector_search(query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Performs k-NN vector search over the FAISS index.

    Args:
        query_embedding: Query vector (list of floats).
        top_k: Number of nearest neighbors to return.

    Returns:
        List of hits in format compatible with previous OpenSearch response:
        [{"_source": {"text": "...", "document_name": "..."}}, ...]
    """
    index, metadata = _load_index_and_metadata()
    if index is None or index.ntotal == 0:
        logger.info("FAISS index empty or missing; returning no results.")
        return []

    top_k = min(top_k, index.ntotal)
    query = np.array([query_embedding], dtype=np.float32)
    distances, indices = index.search(query, top_k)

    hits = []
    for idx in indices[0]:
        if idx < 0:
            continue
        m = metadata[idx]
        hits.append({
            "_source": {
                "text": m["text"],
                "document_name": m["document_name"],
            }
        })
    logger.info(f"FAISS vector search returned {len(hits)} results.")
    return hits


def list_document_names() -> List[str]:
    """
    Returns the list of unique document names stored in the index.

    Returns:
        Sorted list of unique document_name values.
    """
    _, metadata = _load_index_and_metadata()
    if not metadata:
        return []
    names = sorted({m["document_name"] for m in metadata})
    logger.info(f"Listed {len(names)} document names from FAISS store.")
    return names
