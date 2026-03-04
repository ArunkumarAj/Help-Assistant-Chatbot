"""
Structured chat logs: each turn logged with source (RAG vs LLM-only) and metadata.
Writes to a JSONL file and optionally emits a human-readable line to the app logger.
"""
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from core.config import settings

logger = logging.getLogger(__name__)

# Source meaning:
# - RAG:          Answer used retrieved knowledge-base context (hybrid search + LLM).
# - RAG_NO_HITS:  RAG was enabled but no chunks were retrieved; LLM answered with "no information" instruction.
# - LLM_ONLY:     RAG was disabled; answer from LLM without retrieval.
SOURCE_RAG = "RAG"
SOURCE_RAG_NO_HITS = "RAG_NO_HITS"
SOURCE_LLM_ONLY = "LLM_ONLY"


def _truncate(text: str, max_len: int) -> str:
    if not text or max_len <= 0:
        return ""
    text = text.strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3].rstrip() + "..."


def write_chat_log(
    query: str,
    response: str,
    source: str,
    *,
    num_chunks: int = 0,
    from_cache: bool = False,
    temperature: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Append one chat turn to the chat log file (JSONL) and log a readable line.
    source: one of RAG, RAG_NO_HITS, LLM_ONLY.
    """
    if not getattr(settings, "chat_log_enabled", True):
        return
    path = getattr(settings, "chat_log_path", None) or "logs/chat_logs.jsonl"
    preview_len = getattr(settings, "chat_log_preview_len", 200)
    log_dir = os.path.dirname(path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    ts = datetime.now(timezone.utc).isoformat()
    query_preview = _truncate(query, preview_len)
    response_preview = _truncate(response, preview_len)

    # In the log file, show cache in source so it's obvious cache is working
    source_in_log = f"{source} (cached)" if from_cache else source

    entry = {
        "timestamp_utc": ts,
        "source": source_in_log,
        "query": query_preview,
        "response_preview": response_preview,
        "num_chunks": num_chunks,
        "from_cache": from_cache,
        "temperature": temperature,
    }
    if extra:
        entry["extra"] = extra

    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning("Chat log write failed: %s", e)
        return

    # Human-readable line for app.log / console
    source_label = {
        SOURCE_RAG: "RAG (knowledge base)",
        SOURCE_RAG_NO_HITS: "RAG (no chunks found)",
        SOURCE_LLM_ONLY: "LLM only (no retrieval)",
    }.get(source, source)
    cache_note = " [from cache]" if from_cache else ""
    logger.info(
        "Chat | source=%s | chunks=%s%s | query=%s | response=%s",
        source_label,
        num_chunks,
        cache_note,
        query_preview,
        response_preview,
    )
