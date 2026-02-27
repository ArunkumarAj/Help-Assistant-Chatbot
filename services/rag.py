"""RAG: vector search + prompt + LLM. Async-friendly by running blocking calls in executor."""
import asyncio
import logging
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
from core.config import settings
from core.logging_config import setup_logging
from embedding.model import get_embedding_model
from llm.client import get_llm
from vector_store.store import vector_search

setup_logging()
logger = logging.getLogger(__name__)


SUPPORT_ASSISTANT_SYSTEM_PROMPT = """You are a Help Support Assistant designed to answer questions strictly based on the information provided in the RAG Knowledge Base.
Your tone must always be professional, friendly, clear, and helpful—like a dedicated customer support representative.

Rules you must always follow:
1. Only answer questions using the information available in the RAG knowledge articles.
2. Do NOT create, invent, assume, or guess any information that is not in the knowledge base.
3. Do NOT go out of scope. Stay focused only on the topics covered in the provided knowledge articles.
4. If a user asks anything outside the information available, respond politely with:
   "I'm sorry, but I don't have the information to answer that. Please contact support for further assistance."
5. If a user asks for something unrelated to the product, process, or knowledge content, respond with the same support‑redirect message.
6. Maintain the persona of a helpful support assistant at all times.

Your goals:
- Provide accurate, concise, and supportive answers.
- Avoid speculation.
- Assist users in understanding the policies, processes, and instructions exactly as documented.
- Redirect the user to contact live support when the answer cannot be determined from the knowledge base.

Do NOT break character. Always sound like a helpful support assistant.
"""


def _build_prompt(query: str, context: str, history: List[Dict[str, str]]) -> str:
    prompt = SUPPORT_ASSISTANT_SYSTEM_PROMPT + "\n\n"
    if context:
        prompt += "RAG Knowledge Base (cite each claim with the source number in square brackets, e.g. [1], [2]):\n" + context + "\n\n"
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
    results = vector_search(q_emb, top_k=top_k)
    context = ""
    for i, hit in enumerate(results):
        context += f"[{i + 1}] {hit['_source']['text']}\n\n"
    prompt = _build_prompt(query, context, [])
    return results, context, prompt


async def chat_response(
    query: str,
    use_rag: bool = True,
    num_results: int = 5,
    temperature: float = 0.7,
    chat_history: Optional[List[Dict[str, str]]] = None,
) -> str:
    """Run RAG (optional) + LLM. Uses Redis cache for embeddings, retrieval, and (when temperature=0) LLM responses."""
    history = (chat_history or [])[-10:]
    context = ""
    prefix = f"passage: {query}" if settings.asymmetric_embedding else query

    if use_rag:
        loop = asyncio.get_event_loop()

        def _encode_and_search():
            # Embedding cache
            q_emb = cache_get_embedding(prefix)
            if q_emb is None:
                model = get_embedding_model()
                q_emb = model.encode(prefix).tolist()
                cache_set_embedding(prefix, q_emb)
            # Retrieval cache
            results = cache_get_retrieval(prefix, num_results)
            if results is None:
                results = vector_search(q_emb, top_k=num_results)
                cache_set_retrieval(prefix, num_results, results)
            return results

        t0 = time.perf_counter()
        results = await loop.run_in_executor(None, _encode_and_search)
        if getattr(settings, "eval_logging_enabled", False):
            logger.info("eval_latency_retrieve_seconds=%.4f", time.perf_counter() - t0)
        for i, hit in enumerate(results):
            context += f"[{i + 1}] {hit['_source']['text']}\n\n"

    prompt = _build_prompt(query, context, history)

    # Response cache: only when temperature=0 for consistent, repeatable answers
    if temperature == 0:
        ph = hash_prompt(prompt)
        cached = cache_get_response(ph)
        if cached is not None:
            return cached

    def _invoke_llm():
        llm = get_llm(temperature=temperature, top_p=0.9, max_tokens=2000)
        return llm.invoke(prompt)

    loop = asyncio.get_event_loop()
    t0 = time.perf_counter()
    try:
        out = await loop.run_in_executor(None, _invoke_llm) or ""
        if getattr(settings, "eval_logging_enabled", False):
            logger.info("eval_latency_generate_seconds=%.4f", time.perf_counter() - t0)
        if temperature == 0:
            cache_set_response(hash_prompt(prompt), out)
        return out
    except Exception as e:
        logger.exception("LLM invocation failed")
        return f"Sorry, an error occurred: {e!s}"
