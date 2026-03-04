"""
Application configuration: paths, feature flags, and external service settings.

Values are read from environment variables (with defaults). The Settings class
exposes them for use across the app. Load .env via dotenv in the module that
first imports this (e.g. main or LLM client).
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)


# -----------------------------------------------------------------------------
# Paths (project root = parent of core/)
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "logs"
DATA_DIR = PROJECT_ROOT / "data"
UPLOAD_DIR = PROJECT_ROOT / "uploaded_files"

LOG_FILE_PATH = os.environ.get("LOG_FILE_PATH", str(LOG_DIR / "app.log"))


# -----------------------------------------------------------------------------
# Embedding model
# -----------------------------------------------------------------------------

EMBEDDING_MODEL_PATH = os.environ.get("EMBEDDING_MODEL_PATH", "microsoft/mpnet-base")
EMBEDDING_DIMENSION = int(os.environ.get("EMBEDDING_DIMENSION", "768"))
# When true, prefix queries with "passage: " for asymmetric retrieval.
ASSYMETRIC_EMBEDDING = os.environ.get("ASSYMETRIC_EMBEDDING", "false").lower() in ("true", "1")
TEXT_CHUNK_SIZE = int(os.environ.get("TEXT_CHUNK_SIZE", "300"))
TEXT_CHUNK_OVERLAP = int(os.environ.get("TEXT_CHUNK_OVERLAP", "100"))


# -----------------------------------------------------------------------------
# Vector store (ChromaDB persistent path)
# -----------------------------------------------------------------------------

VECTOR_STORE_PATH = os.environ.get(
    "VECTOR_STORE_PATH",
    os.environ.get("CHROMA_PERSIST_DIR", str(DATA_DIR / "chroma_db")),
)


# -----------------------------------------------------------------------------
# Chat logs (JSONL file + optional app logger line)
# -----------------------------------------------------------------------------

CHAT_LOG_ENABLED = os.environ.get("CHAT_LOG_ENABLED", "true").lower() in ("true", "1")
CHAT_LOG_PATH = os.environ.get("CHAT_LOG_PATH", str(LOG_DIR / "chat_logs.jsonl"))
CHAT_LOG_PREVIEW_LEN = int(os.environ.get("CHAT_LOG_PREVIEW_LEN", "200"))


# -----------------------------------------------------------------------------
# Evaluation (latency logging and report output)
# -----------------------------------------------------------------------------

EVAL_LOGGING_ENABLED = os.environ.get("EVAL_LOGGING_ENABLED", "false").lower() in ("true", "1")
EVAL_REPORTS_DIR = os.environ.get("EVAL_REPORTS_DIR", str(PROJECT_ROOT / "eval" / "reports"))


# -----------------------------------------------------------------------------
# Redis cache (embeddings, retrieval, LLM responses)
# -----------------------------------------------------------------------------

REDIS_URL = os.environ.get("REDIS_URL", "")
CACHE_ENABLED = os.environ.get("CACHE_ENABLED", "false").lower() in ("true", "1")
CACHE_TTL_EMBEDDING = int(os.environ.get("CACHE_TTL_EMBEDDING", "86400"))   # 24h
CACHE_TTL_RETRIEVAL = int(os.environ.get("CACHE_TTL_RETRIEVAL", "3600"))   # 1h
CACHE_TTL_RESPONSE = int(os.environ.get("CACHE_TTL_RESPONSE", "3600"))     # 1h


# -----------------------------------------------------------------------------
# Settings namespace (used by rest of app)
# -----------------------------------------------------------------------------


class Settings:
    """Single namespace for all config used across the app."""
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
    chat_log_enabled = CHAT_LOG_ENABLED
    chat_log_path = CHAT_LOG_PATH
    chat_log_preview_len = CHAT_LOG_PREVIEW_LEN
    eval_logging_enabled = EVAL_LOGGING_ENABLED
    eval_reports_dir = Path(EVAL_REPORTS_DIR)
    redis_url = REDIS_URL
    cache_enabled = CACHE_ENABLED
    cache_ttl_embedding = CACHE_TTL_EMBEDDING
    cache_ttl_retrieval = CACHE_TTL_RETRIEVAL
    cache_ttl_response = CACHE_TTL_RESPONSE


settings = Settings()
