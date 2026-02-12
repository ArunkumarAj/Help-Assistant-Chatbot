"""Embedding model loading and generation. CPU-bound; run in executor when called from async API."""
import logging
from typing import List, Any

import numpy as np
from sentence_transformers import SentenceTransformer

from core.config import settings
from core.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

_model: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    """Load and cache the SentenceTransformer model."""
    global _model
    if _model is None:
        logger.info("Loading embedding model: %s", settings.embedding_model_path)
        _model = SentenceTransformer(settings.embedding_model_path)
    return _model


def generate_embeddings(chunks: List[str]) -> List[Any]:
    """Generate embeddings for a list of text chunks. Returns list of numpy arrays."""
    model = get_embedding_model()
    embeddings = [np.array(model.encode(chunk)) for chunk in chunks]
    logger.info("Generated embeddings for %s chunks.", len(chunks))
    return embeddings
