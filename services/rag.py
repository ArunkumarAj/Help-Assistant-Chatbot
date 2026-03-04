"""RAG: vector search + prompt + LLM. Async-friendly by running blocking calls in executor."""
import asyncio
import logging
import re
import time
from typing import Dict, List, Optional, Tuple, Any

from core.cache import (
    cache_get_embedding,
    cache_get_retrieval,
    cache_get_response,
    cache_set_embedding,
    cache_set_retrieval,
    cache_set_response,
    hash_prompt,
)
from core.chat_log import SOURCE_LLM_ONLY, SOURCE_RAG, SOURCE_RAG_NO_HITS, write_chat_log
from core.config import settings
from core.logging_config import setup_logging
from embedding.model import get_embedding_model
from llm.client import get_llm
from vector_store.store import vector_search

setup_logging()
logger = logging.getLogger(__name__)


SUPPORT_ASSISTANT_SYSTEM_PROMPT = """You are a Help Support Assistant for dealers: answer questions briefly and strictly from the RAG Knowledge Base below.
Keep answers SHORT: one or two sentences, or a very brief paragraph (2–4 sentences). Do not list long bullet points or repeat the full policy—give the direct answer only. If the user asks for more detail, then you may add a bit more.

Rules:
1. Answer ONLY from the RAG knowledge articles. Do not invent or assume anything.
2. If the question is outside the knowledge base, say: "I don't have that information. Please contact support."
3. Be professional, clear, and concise. This is a quick Q&A for dealers, not a long report.
4. Cite sources with [1], [2] where needed, but keep the answer itself short.

Do NOT write long explanations or multiple bullet lists unless the user explicitly asks for detail.
"""


def _citation_label(hit: Dict[str, Any]) -> str:
    """Build (Source: document, p. N) or (Source: document) when page is missing."""
    src = hit.get("_source") or {}
    doc_name = src.get("document_name") or "document"
    page = src.get("page")
    if page is not None:
        return f"(Source: {doc_name}, p. {page})"
    return f"(Source: {doc_name})"


def _replace_citation_markers(response: str, citations: List[str]) -> str:
    """Replace [1], [2], ... in response with actual (Source: doc, p. N). Replaces from high index first."""
    for i in range(len(citations) - 1, -1, -1):
        n = i + 1
        # Replace [n] or [n] at word boundary so we don't break mid-number
        pattern = r"\[" + str(n) + r"\]"
        response = re.sub(pattern, " " + citations[i], response)
    return response


def _build_prompt(query: str, context: str, history: List[Dict[str, str]]) -> str:
    prompt = SUPPORT_ASSISTANT_SYSTEM_PROMPT + "\n\n"
    if context:
        prompt += "RAG Knowledge Base (cite each claim with the source number in square brackets [1], [2], etc.; each number refers to the block below):\n" + context + "\n\n"
    else:
        prompt += "No knowledge articles were found for this query. If the user's question cannot be answered from the knowledge base, respond with: \"I'm sorry, but I don't have the information to answer that. Please contact support for further assistance.\"\n\n"
    if history:
        prompt += "Conversation History:\n"
        for msg in history:
            role = "User" if msg["role"] == "user" else "Assistant"
            prompt += f"{role}: {msg['content']}\n"
        prompt += "\n"
    prompt += f"User: {query}\nAssistant:"
    return prompt


def eval_retrieve_and_build_prompt(query: str, top_k: int) -> Tuple[List[Dict[str, Any]], str, str]:
    """
    Synchronous helper for evaluation: run retrieval and return (results, context, prompt).
    Caller can then time LLM invoke separately.
    """
    prefix = f"passage: {query}" if settings.asymmetric_embedding else query
    model = get_embedding_model()
    q_emb = model.encode(prefix).tolist()
    results = vector_search(q_emb, top_k=top_k, query_text=query)
    context = ""
    for i, hit in enumerate(results):
        label = _citation_label(hit)
        context += f"[{i + 1}] {label}\n{hit['_source']['text']}\n\n"
    prompt = _build_prompt(query, context, [])
    return results, context, prompt


def _citation_meta_list(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build list of {index, document_name, page, doc_id} for UI (e.g. hover tooltip)."""
    meta = []
    for i, hit in enumerate(results):
        src = hit.get("_source") or {}
        meta.append({
            "index": i + 1,
            "document_name": src.get("document_name") or "",
            "page": src.get("page"),
            "doc_id": hit.get("id") or "",
        })
    return meta


async def chat_response(
    query: str,
    use_rag: bool = True,
    num_results: int = 5,
    temperature: float = 0.7,
    chat_history: Optional[List[Dict[str, str]]] = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Run RAG (optional) + LLM. Returns (response_text, citation_meta). citation_meta has index, document_name, page, doc_id for UI (e.g. details icon tooltip)."""
    history = (chat_history or [])[-10:]
    context = ""
    num_chunks = 0
    citation_labels: List[str] = []
    citation_meta: List[Dict[str, Any]] = []
    prefix = f"passage: {query}" if settings.asymmetric_embedding else query

    if use_rag:
        loop = asyncio.get_event_loop()

        def _encode_and_search():
            q_emb = cache_get_embedding(prefix)
            if q_emb is None:
                model = get_embedding_model()
                q_emb = model.encode(prefix).tolist()
                cache_set_embedding(prefix, q_emb)
            results = cache_get_retrieval(prefix, num_results)
            if results is None:
                results = vector_search(q_emb, top_k=num_results, query_text=query)
                cache_set_retrieval(prefix, num_results, results)
            return results

        t0 = time.perf_counter()
        results = await loop.run_in_executor(None, _encode_and_search)
        num_chunks = len(results)
        if getattr(settings, "eval_logging_enabled", False):
            logger.info("eval_latency_retrieve_seconds=%.4f", time.perf_counter() - t0)
        for i, hit in enumerate(results):
            citation_labels.append(_citation_label(hit))
            context += f"[{i + 1}] {citation_labels[-1]}\n{hit['_source']['text']}\n\n"
        citation_meta = _citation_meta_list(results)

    prompt = _build_prompt(query, context, history)

    if temperature == 0:
        ph = hash_prompt(prompt)
        cached = cache_get_response(ph)
        if cached is not None:
            # Leave [1], [2] in text so frontend can show details icon with tooltip
            _source = SOURCE_RAG if (use_rag and num_chunks > 0) else (SOURCE_RAG_NO_HITS if use_rag else SOURCE_LLM_ONLY)
            write_chat_log(
                query, cached, _source,
                num_chunks=num_chunks, from_cache=True, temperature=temperature,
            )
            return (cached, citation_meta)

    def _invoke_llm():
        llm = get_llm(temperature=temperature, top_p=0.9, max_tokens=512)
        return llm.invoke(prompt)

    loop = asyncio.get_event_loop()
    t0 = time.perf_counter()
    try:
        out = await loop.run_in_executor(None, _invoke_llm) or ""
        if getattr(settings, "eval_logging_enabled", False):
            logger.info("eval_latency_generate_seconds=%.4f", time.perf_counter() - t0)
        if temperature == 0:
            cache_set_response(hash_prompt(prompt), out)
        # Do not replace [1],[2] here; frontend will show details icon with document id + page on hover
        _source = SOURCE_RAG if (use_rag and num_chunks > 0) else (SOURCE_RAG_NO_HITS if use_rag else SOURCE_LLM_ONLY)
        write_chat_log(
            query, out, _source,
            num_chunks=num_chunks, from_cache=False, temperature=temperature,
        )
        return (out, citation_meta)
    except Exception as e:
        logger.exception("LLM invocation failed")
        err_msg = f"Sorry, an error occurred: {e!s}"
        write_chat_log(
            query, err_msg, SOURCE_LLM_ONLY,
            num_chunks=num_chunks, from_cache=False, temperature=temperature,
            extra={"error": str(e)},
        )
        return (err_msg, [])
