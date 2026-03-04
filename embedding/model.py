"""
Embedding model: load once, then encode text to vectors.

Uses SentenceTransformer (e.g. microsoft/mpnet-base). Model path and dimension
come from settings. Encoding is CPU-bound; callers running from async code
should use run_in_executor.
"""
import logging
from typing import Any, List

import numpy as np
from sentence_transformers import SentenceTransformer

from core.config import settings
from core.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

# Lazy-loaded model (single instance)
_model: Any = None


def get_embedding_model() -> Any:
    """Load and cache the SentenceTransformer model. Returns the same instance on every call."""
    global _model
    if _model is None:
        logger.info("Loading embedding model: %s", settings.embedding_model_path)
        _model = SentenceTransformer(settings.embedding_model_path)
    return _model


def generate_embeddings(chunks: List[str]) -> List[Any]:
    """Generate embeddings for a list of text chunks. Returns a list of numpy arrays."""
    model = get_embedding_model()
    embeddings = [np.array(model.encode(chunk)) for chunk in chunks]
    logger.info("Generated embeddings for %s chunks.", len(chunks))
    return embeddings
