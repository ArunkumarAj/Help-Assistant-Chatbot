"""Shared configuration. Load from env where needed."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

# Base paths (project root = parent of core/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "logs"
DATA_DIR = PROJECT_ROOT / "data"
UPLOAD_DIR = PROJECT_ROOT / "uploaded_files"

# Logging
LOG_FILE_PATH = os.environ.get("LOG_FILE_PATH", str(LOG_DIR / "app.log"))

# Embedding
EMBEDDING_MODEL_PATH = os.environ.get("EMBEDDING_MODEL_PATH", "microsoft/mpnet-base")
EMBEDDING_DIMENSION = int(os.environ.get("EMBEDDING_DIMENSION", "768"))
ASSYMETRIC_EMBEDDING = os.environ.get("ASSYMETRIC_EMBEDDING", "false").lower() in ("true", "1")
TEXT_CHUNK_SIZE = int(os.environ.get("TEXT_CHUNK_SIZE", "300"))
TEXT_CHUNK_OVERLAP = int(os.environ.get("TEXT_CHUNK_OVERLAP", "100"))

# Vector store (ChromaDB: persistent, hybrid search). Replaces FAISS.
VECTOR_STORE_PATH = os.environ.get("VECTOR_STORE_PATH", os.environ.get("CHROMA_PERSIST_DIR", str(DATA_DIR / "chroma_db")))

# LLM (from env in llm module; listed here for reference)
# API_URL, API_KEY, LLM_MODEL

# Eval: logging hooks for latency (and optional token estimation) in RAG pipeline
EVAL_LOGGING_ENABLED = os.environ.get("EVAL_LOGGING_ENABLED", "false").lower() in ("true", "1")
EVAL_REPORTS_DIR = os.environ.get("EVAL_REPORTS_DIR", str(PROJECT_ROOT / "eval" / "reports"))

# Redis cache (RAG: embeddings, retrieval, LLM responses)
REDIS_URL = os.environ.get("REDIS_URL", "")  # e.g. redis://localhost:6379/0 or redis://:pass@host:6379/0
CACHE_ENABLED = os.environ.get("CACHE_ENABLED", "false").lower() in ("true", "1")
CACHE_TTL_EMBEDDING = int(os.environ.get("CACHE_TTL_EMBEDDING", "86400"))   # 24h
CACHE_TTL_RETRIEVAL = int(os.environ.get("CACHE_TTL_RETRIEVAL", "3600"))   # 1h
CACHE_TTL_RESPONSE = int(os.environ.get("CACHE_TTL_RESPONSE", "3600"))     # 1h


class Settings:
    """Namespace for settings used across the app."""
    project_root = PROJECT_ROOT
    log_file_path = LOG_FILE_PATH
    log_dir = LOG_DIR
    data_dir = DATA_DIR
    upload_dir = UPLOAD_DIR
    embedding_model_path = EMBEDDING_MODEL_PATH
    embedding_dimension = EMBEDDING_DIMENSION
    asymmetric_embedding = ASSYMETRIC_EMBEDDING
    text_chunk_size = TEXT_CHUNK_SIZE
    text_chunk_overlap = TEXT_CHUNK_OVERLAP
    vector_store_path = VECTOR_STORE_PATH
    eval_logging_enabled = EVAL_LOGGING_ENABLED
    eval_reports_dir = Path(EVAL_REPORTS_DIR)
    redis_url = REDIS_URL
    cache_enabled = CACHE_ENABLED
    cache_ttl_embedding = CACHE_TTL_EMBEDDING
    cache_ttl_retrieval = CACHE_TTL_RETRIEVAL
    cache_ttl_response = CACHE_TTL_RESPONSE


settings = Settings()
