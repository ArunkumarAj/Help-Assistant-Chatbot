import logging
from typing import Dict, Iterable, List, Optional

from src.constants import ASSYMETRIC_EMBEDDING
from src.custom_llm import get_llm
from src.embeddings import get_embedding_model
from src.faiss_store import vector_search
from src.utils import setup_logging

# Initialize logger
setup_logging()
logger = logging.getLogger(__name__)


def prompt_template(query: str, context: str, history: List[Dict[str, str]]) -> str:
    """
    Builds the prompt with context, conversation history, and user query.

    Args:
        query (str): The user's query.
        context (str): Context text gathered from vector search.
        history (List[Dict[str, str]]): Conversation history to include in the prompt.

    Returns:
        str: Constructed prompt for the LLM.
    """
    prompt = "You are a knowledgeable chatbot assistant. "
    if context:
        prompt += (
            "Use the following context to answer the question.\nContext:\n"
            + context
            + "\n"
        )
    else:
        prompt += "Answer questions to the best of your knowledge.\n"

    if history:
        prompt += "Conversation History:\n"
        for msg in history:
            role = "User" if msg["role"] == "user" else "Assistant"
            content = msg["content"]
            prompt += f"{role}: {content}\n"
        prompt += "\n"

    prompt += f"User: {query}\nAssistant:"
    logger.info("Prompt constructed with context and conversation history.")
    return prompt


def _response_as_stream(text: str) -> Iterable[Dict]:
    """
    Yields a single chunk in the same format the UI expects from a stream.
    Allows the custom LLM (non-streaming) to work with the existing chat UI.
    """
    yield {"message": {"content": text}}


def generate_response_streaming(
    query: str,
    use_hybrid_search: bool,
    num_results: int,
    temperature: float,
    chat_history: Optional[List[Dict[str, str]]] = None,
) -> Optional[Iterable[Dict]]:
    """
    Generates a chatbot response by performing vector search and calling the custom LLM.

    Args:
        query (str): The user's query.
        use_hybrid_search (bool): Whether to use RAG (vector search) for context.
        num_results (int): The number of search results to include in the context.
        temperature (float): The temperature for the response generation.
        chat_history (Optional[List[Dict[str, str]]]): List of chat history messages.

    Returns:
        Optional[Iterable[Dict]]: An iterator yielding chunks in format {"message": {"content": "..."}}.
    """
    chat_history = chat_history or []
    max_history_messages = 10
    history = chat_history[-max_history_messages:]
    context = ""

    if use_hybrid_search:
        logger.info("Performing vector search for RAG context.")
        if ASSYMETRIC_EMBEDDING:
            prefixed_query = f"passage: {query}"
        else:
            prefixed_query = query
        embedding_model = get_embedding_model()
        query_embedding = embedding_model.encode(prefixed_query).tolist()
        search_results = vector_search(query_embedding, top_k=num_results)
        logger.info("Vector search completed.")

        for i, result in enumerate(search_results):
            context += f"Document {i}:\n{result['_source']['text']}\n\n"

    prompt = prompt_template(query, context, history)

    try:
        llm = get_llm(temperature=temperature, top_p=0.9, max_tokens=2000)
        response = llm.invoke(prompt)
        return _response_as_stream(response or "")
    except Exception as e:
        logger.error("LLM invocation failed: %s", e)
        return _response_as_stream(f"Sorry, an error occurred: {e!s}")
