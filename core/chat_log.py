"""
Structured chat logs: each turn is written as one JSON line with source and metadata.

- Writes to a JSONL file (e.g. logs/chat_logs.jsonl).
- When the response is from cache, the source field in the file includes " (cached)".
- Optionally logs a human-readable line to the app logger.
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from core.config import settings

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Source labels (what produced the answer)
# -----------------------------------------------------------------------------
# RAG:          Answer used retrieved knowledge-base context (hybrid search + LLM).
# RAG_NO_HITS: RAG was on but no chunks were retrieved; LLM answered with "no information" instruction.
# LLM_ONLY:    RAG was off; answer from LLM without retrieval.

SOURCE_RAG = "RAG"
SOURCE_RAG_NO_HITS = "RAG_NO_HITS"
SOURCE_LLM_ONLY = "LLM_ONLY"


def _truncate_for_preview(text: str, max_length: int) -> str:
    """Truncate text to max_length, appending '...' if needed. Returns empty string if text is falsy or max_length <= 0."""
    if not text or max_length <= 0:
        return ""
    text = text.strip()
    if len(text) <= max_length:
        return text
    return text[: max_length - 3].rstrip() + "..."


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
    Append one chat turn to the chat log file (JSONL) and log a readable line to the app logger.
    source: one of SOURCE_RAG, SOURCE_RAG_NO_HITS, SOURCE_LLM_ONLY.
    """
    if not getattr(settings, "chat_log_enabled", True):
        return

    log_path = getattr(settings, "chat_log_path", None) or "logs/chat_logs.jsonl"
    preview_len = getattr(settings, "chat_log_preview_len", 200)
    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    timestamp_utc = datetime.now(timezone.utc).isoformat()
    query_preview = _truncate_for_preview(query, preview_len)
    response_preview = _truncate_for_preview(response, preview_len)

    # In the log file, show cache in source so it's obvious when cache is used
    source_display = f"{source} (cached)" if from_cache else source

    entry = {
        "timestamp_utc": timestamp_utc,
        "source": source_display,
        "query": query_preview,
        "response_preview": response_preview,
        "num_chunks": num_chunks,
        "from_cache": from_cache,
        "temperature": temperature,
    }
    if extra:
        entry["extra"] = extra

    try:
        with open(log_path, "a", encoding="utf-8") as f:
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
