"""RAG: vector search + prompt + LLM. Async-friendly by running blocking calls in executor."""
import asyncio
import logging
from typing import Dict, List, Optional

from core.config import settings
from core.logging_config import setup_logging
from embedding.model import get_embedding_model
from llm.client import get_llm
from vector_store.store import vector_search

setup_logging()
logger = logging.getLogger(__name__)


def _build_prompt(query: str, context: str, history: List[Dict[str, str]]) -> str:
    prompt = "You are a knowledgeable chatbot assistant. "
    if context:
        prompt += "Use the following context to answer the question.\nContext:\n" + context + "\n\n"
    else:
        prompt += "Answer questions to the best of your knowledge.\n"
    if history:
        prompt += "Conversation History:\n"
        for msg in history:
            role = "User" if msg["role"] == "user" else "Assistant"
            prompt += f"{role}: {msg['content']}\n"
        prompt += "\n"
    prompt += f"User: {query}\nAssistant:"
    return prompt


async def chat_response(
    query: str,
    use_rag: bool = True,
    num_results: int = 5,
    temperature: float = 0.7,
    chat_history: Optional[List[Dict[str, str]]] = None,
) -> str:
    """Run RAG (optional) + LLM. Blocking work runs in executor."""
    history = (chat_history or [])[-10:]
    context = ""

    if use_rag:
        loop = asyncio.get_event_loop()
        prefix = f"passage: {query}" if settings.asymmetric_embedding else query

        def _encode_and_search():
            model = get_embedding_model()
            q_emb = model.encode(prefix).tolist()
            return vector_search(q_emb, top_k=num_results)

        results = await loop.run_in_executor(None, _encode_and_search)
        for i, hit in enumerate(results):
            context += f"Document {i}:\n{hit['_source']['text']}\n\n"

    prompt = _build_prompt(query, context, history)

    def _invoke_llm():
        llm = get_llm(temperature=temperature, top_p=0.9, max_tokens=2000)
        return llm.invoke(prompt)

    loop = asyncio.get_event_loop()
    try:
        return await loop.run_in_executor(None, _invoke_llm) or ""
    except Exception as e:
        logger.exception("LLM invocation failed")
        return f"Sorry, an error occurred: {e!s}"
