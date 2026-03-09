"""
RAG (Retrieval-Augmented Generation): search the knowledge base, then answer with the LLM.

Flow:
  1. Optionally run hybrid search (dense + BM25) to get relevant chunks.
  2. Build a prompt with system instructions, retrieved context, and chat history.
  3. Call the LLM; cache the response when temperature is 0.
Blocking work (embedding, search, LLM) runs in a thread executor so the API stays async.
"""

import asyncio
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

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


# -----------------------------------------------------------------------------
# System prompt and citation helpers
# -----------------------------------------------------------------------------

SUPPORT_ASSISTANT_SYSTEM_PROMPT = """You are a Help Support Assistant for dealers: answer questions briefly and strictly from the RAG Knowledge Base below.
Keep answers SHORT: one or two sentences, or a very brief paragraph (2–4 sentences). Do not list long bullet points or repeat the full policy—give the direct answer only. If the user asks for more detail, then you may add a bit more.

Rules:
1. Answer ONLY from the RAG knowledge articles. Do not invent or assume anything.
2. If the question is outside the knowledge base, say: "I don't have that information. Please contact support."
3. Be professional, clear, and concise. This is a quick Q&A for dealers, not a long report.
4. Cite sources with [1], [2] where needed, but keep the answer itself short.
5. When the user asks what something is (e.g. "what is X?" or "what does X mean?"), give the definition from the context when it appears—e.g. "X means Y"—then add any brief, relevant detail from the same context. Do not only describe what the knowledge base "says" in general; state the definition directly when the context contains it.

Do NOT write long explanations or multiple bullet lists unless the user explicitly asks for detail.
"""


def _citation_label(hit: Dict[str, Any]) -> str:
    """Format a single search hit as a citation label: (Source: document_name, p. N) or (Source: document_name)."""
    source = hit.get("_source") or {}
    document_name = source.get("document_name") or "document"
    page = source.get("page")
    if page is not None:
        return f"(Source: {document_name}, p. {page})"
    return f"(Source: {document_name})"


def _replace_citation_markers(response: str, citations: List[str]) -> str:
    """Replace [1], [2], ... in the response with the actual citation strings. Replaces from high index first to avoid shifting positions."""
    for i in range(len(citations) - 1, -1, -1):
        number = i + 1
        pattern = r"\[" + str(number) + r"\]"
        response = re.sub(pattern, " " + citations[i], response)
    return response


def _build_prompt(query: str, context: str, history: List[Dict[str, str]]) -> str:
    """Build the full prompt: system instructions + optional RAG context + conversation history + current query."""
    prompt = SUPPORT_ASSISTANT_SYSTEM_PROMPT + "\n\n"

    if context:
        prompt += (
            "RAG Knowledge Base (cite each claim with the source number in square brackets [1], [2], etc.; each number refers to the block below):\n"
            + context
            + "\n\n"
        )
    else:
        prompt += (
            "No knowledge articles were found for this query. If the user's question cannot be answered from the knowledge base, "
            'respond with: "I\'m sorry, but I don\'t have the information to answer that. Please contact support for further assistance."\n\n'
        )

    if history:
        prompt += "Conversation History:\n"
        for msg in history:
            role = "User" if msg["role"] == "user" else "Assistant"
            prompt += f"{role}: {msg['content']}\n"
        prompt += "\n"

    prompt += f"User: {query}\nAssistant:"
    return prompt


def _chat_log_source(use_rag: bool, num_chunks: int) -> str:
    """Decide the chat log source label: RAG, RAG_NO_HITS, or LLM_ONLY."""
    if not use_rag:
        return SOURCE_LLM_ONLY
    return SOURCE_RAG if num_chunks > 0 else SOURCE_RAG_NO_HITS


def _build_citation_meta_list(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build metadata for each citation (index, document_name, page, doc_id) for the UI (e.g. hover tooltip)."""
    meta_list = []
    for i, hit in enumerate(results):
        source = hit.get("_source") or {}
        meta_list.append({
            "index": i + 1,
            "document_name": source.get("document_name") or "",
            "page": source.get("page"),
            "doc_id": hit.get("id") or "",
        })
    return meta_list


# -----------------------------------------------------------------------------
# Evaluation helper (synchronous)
# -----------------------------------------------------------------------------


def eval_retrieve_and_build_prompt(query: str, top_k: int) -> Tuple[List[Dict[str, Any]], str, str]:
    """
    Run retrieval and build the prompt synchronously. Used for evaluation so the caller can time the LLM separately.
    Returns (search_results, context_string, full_prompt).
    """
    query_prefix = f"passage: {query}" if settings.asymmetric_embedding else query
    model = get_embedding_model()
    query_embedding = model.encode(query_prefix).tolist()
    results = vector_search(query_embedding, top_k=top_k, query_text=query)

    context = ""
    for i, hit in enumerate(results):
        label = _citation_label(hit)
        context += f"[{i + 1}] {label}\n{hit['_source']['text']}\n\n"
    full_prompt = _build_prompt(query, context, [])
    return results, context, full_prompt


# -----------------------------------------------------------------------------
# Main chat response (async)
# -----------------------------------------------------------------------------


async def chat_response(
    query: str,
    use_rag: bool = True,
    num_results: int = 5,
    temperature: float = 0.7,
    chat_history: Optional[List[Dict[str, str]]] = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Run RAG (optional) and LLM to produce an answer.
    Returns (response_text, citation_meta). citation_meta is a list of {index, document_name, page, doc_id} for the UI.
    """
    # Use only the last 10 turns of history to avoid huge prompts
    history = (chat_history or [])[-10:]

    context = ""
    num_chunks = 0
    citation_meta: List[Dict[str, Any]] = []
    query_prefix = f"passage: {query}" if settings.asymmetric_embedding else query

    if use_rag:
        loop = asyncio.get_event_loop()

        def _embed_query_and_search():
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

        start = time.perf_counter()
        results = await loop.run_in_executor(None, _embed_query_and_search)
        num_chunks = len(results)
        if getattr(settings, "eval_logging_enabled", False):
            logger.info("eval_latency_retrieve_seconds=%.4f", time.perf_counter() - start)

        for i, hit in enumerate(results):
            context += f"[{i + 1}] {_citation_label(hit)}\n{hit['_source']['text']}\n\n"
        citation_meta = _build_citation_meta_list(results)

    prompt = _build_prompt(query, context, history)

    # When temperature is 0, responses are cached by prompt hash
    if temperature == 0:
        prompt_hash = hash_prompt(prompt)
        cached_response = cache_get_response(prompt_hash)
        if cached_response is not None:
            log_source = _chat_log_source(use_rag, num_chunks)
            write_chat_log(
                query,
                cached_response,
                log_source,
                num_chunks=num_chunks,
                from_cache=True,
                temperature=temperature,
            )
            return (cached_response, citation_meta)

    def _call_llm():
        llm = get_llm(temperature=temperature, top_p=0.9, max_tokens=512)
        return llm.invoke(prompt)

    loop = asyncio.get_event_loop()
    start = time.perf_counter()
    try:
        response_text = await loop.run_in_executor(None, _call_llm) or ""
        if getattr(settings, "eval_logging_enabled", False):
            logger.info("eval_latency_generate_seconds=%.4f", time.perf_counter() - start)
        if temperature == 0:
            cache_set_response(hash_prompt(prompt), response_text)

        log_source = _chat_log_source(use_rag, num_chunks)
        write_chat_log(
            query,
            response_text,
            log_source,
            num_chunks=num_chunks,
            from_cache=False,
            temperature=temperature,
        )
        return (response_text, citation_meta)
    except Exception as e:
        logger.exception("LLM invocation failed")
        error_message = "Sorry, an error occurred. Please try again."
        write_chat_log(
            query,
            error_message,
            SOURCE_LLM_ONLY,
            num_chunks=num_chunks,
            from_cache=False,
            temperature=temperature,
            extra={"error": str(e)},  # logged server-side only; not returned to user
        )
        return (error_message, [])
