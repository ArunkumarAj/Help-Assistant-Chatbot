"""
LangGraph ReAct agent: tool-calling help assistant (RAG + SQLite cases).
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.prebuilt import create_react_agent

from llm.openai_compat import get_openai_compatible_base_url
from services.agent_tools import CitationHolder, make_agent_tools
try:
    from langchain_openai import ChatOpenAI
except ImportError:  # pragma: no cover
    ChatOpenAI = None  # type: ignore

import os

logger = logging.getLogger(__name__)

AGENT_SYSTEM_PROMPT = """You are a Help Support Assistant for dealers. Be professional, clear, and concise (short answers unless the user asks for detail).

You have three tools:
1) search_knowledge_base — for questions that need information from the official RAG / knowledge articles (policies, definitions, procedures, IMF/Program documents, acronyms as documented).
2) get_open_active_cases — when the user asks to list, show, or get open/Active support cases.
3) create_active_support_case — when the user wants to create, open, or file a new support case (use a clear short title; optional description).

Use the right tool(s). You may use multiple tools in a row if needed. If the user asks for both knowledge and cases, use both. After tool results, answer the user in plain text; cite [1], [2] from knowledge search when the tool output includes numbered sources.

If information is not in the knowledge base, say you do not have that information and suggest support contact. Do not invent policy facts.
"""

def _history_to_messages(history: List[Dict[str, str]]) -> List[BaseMessage]:
    out: List[BaseMessage] = []
    for m in (history or [])[-10:]:
        role = (m.get("role") or "user").lower()
        c = m.get("content") or ""
        if role == "user":
            out.append(HumanMessage(content=c))
        else:
            out.append(AIMessage(content=c))
    return out


def _last_ai_text(messages: List[BaseMessage]) -> str:
    for m in reversed(messages or []):
        if isinstance(m, AIMessage) and m.content and not (getattr(m, "tool_calls", None) or None):
            if isinstance(m.content, str):
                return m.content.strip()
            if isinstance(m.content, list) and m.content:
                for block in m.content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        return (block.get("text") or "").strip()
    for m in reversed(messages or []):
        if isinstance(m, AIMessage) and isinstance(m.content, str) and m.content:
            return m.content.strip()
    return ""


def _replace_citation_brackets(
    response: str,
    citations: List[Dict[str, Any]],
) -> str:
    """Map [1], [2] to (Source: ...) for UI, using citation list from the last RAG tool call."""
    labels: List[str] = []
    for c in sorted(citations, key=lambda x: int(x.get("index", 0) or 0)):
        i = c.get("index", 0)
        doc = c.get("document_name") or "document"
        p = c.get("page")
        if p is not None:
            labels.append(f"(Source: {doc}, p. {p})")
        else:
            labels.append(f"(Source: {doc})")
    out = response
    for j in range(len(labels) - 1, -1, -1):
        num = j + 1
        out = re.sub(r"\[" + str(num) + r"\]", " " + labels[j], out)
    return out


def build_chat_model(temperature: float = 0.7) -> "ChatOpenAI":
    if ChatOpenAI is None:
        raise RuntimeError("langchain_openai is not installed. Run: uv sync  (or pip install langchain-openai)")

    base = get_openai_compatible_base_url()
    if not base or not (base.startswith("http://") or base.startswith("https://")):
        raise ValueError(
            "Set API_URL in .env to your OpenAI-compatible base (e.g. https://.../v1) with tool-calling support."
        )

    api_key = (os.environ.get("API_KEY") or os.environ.get("OPENAI_API_KEY") or "not-needed").strip()
    model_name = (os.environ.get("LLM_MODEL") or "gpt-4o-mini").strip()
    kwargs: Dict[str, Any] = {
        "model": model_name,
        "base_url": base,
        "api_key": api_key,
        "temperature": temperature,
        "max_tokens": 1024,
    }
    if os.environ.get("API_KEY") or os.environ.get("X_API_KEY"):
        kwargs["default_headers"] = {
            "X-API-KEY": (os.environ.get("API_KEY") or os.environ.get("X_API_KEY") or "").strip()
        }
    return ChatOpenAI(**kwargs)


async def run_support_agent(
    query: str,
    *,
    top_k: int = 5,
    temperature: float = 0.7,
    chat_history: Optional[List[Dict[str, str]]] = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Run the LangGraph ReAct agent with RAG and SQLite tools.
    Returns (answer_text, citation_metadata from last knowledge search, may be empty).
    """
    holder = CitationHolder()
    tools = make_agent_tools(top_k, holder)
    model = build_chat_model(temperature)
    graph = create_react_agent(
        model,
        tools,
        prompt=AGENT_SYSTEM_PROMPT,
    )

    msgs: List[BaseMessage] = _history_to_messages((chat_history or []))
    msgs.append(HumanMessage(content=query.strip()))

    result = await graph.ainvoke({"messages": msgs})
    raw_messages: List[BaseMessage] = result.get("messages", [])
    text = _last_ai_text(raw_messages)
    if not text:
        text = "I could not complete that. Please rephrase or contact support."

    citations = list(holder.items)
    if citations and re.search(r"\[(\d+)\]", text):
        text = _replace_citation_brackets(text, citations)
    return text, citations
