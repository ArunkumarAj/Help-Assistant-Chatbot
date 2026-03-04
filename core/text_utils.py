"""
Text cleaning and chunking for RAG ingestion.

- clean_text: fix hyphenation at line breaks, normalize newlines and spaces.
- chunk_text: split text into overlapping word-based chunks (size and overlap from settings).
"""
import logging
import re
from typing import List

from core.config import settings
from core.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """
    Clean raw text for chunking: join hyphenated line breaks, collapse single newlines
    to spaces, normalize multiple newlines to one, and collapse spaces/tabs.
    """
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def chunk_text(
    text: str,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> List[str]:
    """
    Split text into overlapping word-based chunks.
    Uses settings.text_chunk_size and settings.text_chunk_overlap by default.
    """
    size = chunk_size or settings.text_chunk_size
    overlap_size = overlap or settings.text_chunk_overlap
    text = clean_text(text)
    tokens = text.split(" ")
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + size
        chunk_tokens = tokens[start:end]
        chunks.append(" ".join(chunk_tokens))
        start = end - overlap_size
    logger.info(
        "Chunked text into %s chunks (size=%s, overlap=%s).",
        len(chunks),
        size,
        overlap_size,
    )
    return chunks
