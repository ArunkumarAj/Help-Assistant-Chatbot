"""Text cleaning and chunking utilities."""
import logging
import re
from typing import List

from core.config import settings
from core.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """Clean text: fix hyphens at line breaks, normalize newlines and spaces."""
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int | None = None, overlap: int | None = None) -> List[str]:
    """Split text into overlapping word-based chunks."""
    chunk_size = chunk_size or settings.text_chunk_size
    overlap = overlap or settings.text_chunk_overlap
    text = clean_text(text)
    tokens = text.split(" ")
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunks.append(" ".join(chunk_tokens))
        start = end - overlap
    logger.info("Chunked text into %s chunks (size=%s, overlap=%s).", len(chunks), chunk_size, overlap)
    return chunks
