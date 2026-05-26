"""
Shared RAG retrieval: citation labels, context formatting, hybrid search with cache.

Used by the linear eval path, legacy patterns, and LangGraph tools (avoids import cycles).
"""

from typing import Any, Dict, List, Tuple

from core.cache import cache_get_embedding, cache_get_retrieval, cache_set_embedding, cache_set_retrieval
from core.config import settings
from embedding.model import get_embedding_model
from vector_store.store import vector_search


def citation_label(hit: Dict[str, Any]) -> str:
    """(Source: document_name, p. N) or (Source: document_name)."""
    source = hit.get("_source") or {}
    document_name = source.get("document_name") or "document"
    page = source.get("page")
    if page is not None:
        return f"(Source: {document_name}, p. {page})"
    return f"(Source: {document_name})"


def build_citation_meta_list(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Metadata for the UI: index, document_name, page, doc_id."""
    meta_list: List[Dict[str, Any]] = []
    for i, hit in enumerate(results):
        source = hit.get("_source") or {}
        meta_list.append({
            "index": i + 1,
            "document_name": source.get("document_name") or "",
            "page": source.get("page"),
            "doc_id": hit.get("id") or "",
        })
    return meta_list


def retrieve_hybrid(
    query: str,
    num_results: int,
) -> Tuple[List[Dict[str, Any]], str, List[Dict[str, Any]]]:
    """
    Embed query, run cached hybrid search, return (raw hits, RAG context block, citation_meta).
    """
    query_prefix = f"passage: {query}" if settings.asymmetric_embedding else query

    def _search():
        query_embedding = cache_get_embedding(query_prefix)
        if query_embedding is None:
            model = get_embedding_model()
            query_embedding = model.encode(query_prefix).tolist()
            cache_set_embedding(query_prefix, query_embedding)
        results = cache_get_retrieval(query_prefix, num_results)
        if results is None:
            results = vector_search(
                query_embedding,
                top_k=num_results,
                query_text=query,
            )
            cache_set_retrieval(query_prefix, num_results, results)
        return results

    results = _search()
    context = ""
    for i, hit in enumerate(results):
        context += f"[{i + 1}] {citation_label(hit)}\n{hit['_source']['text']}\n\n"
    return results, context, build_citation_meta_list(results)
