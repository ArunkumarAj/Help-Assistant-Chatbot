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

# FAISS vector store
FAISS_INDEX_PATH = os.environ.get("FAISS_INDEX_PATH", str(DATA_DIR / "faiss_index"))

# LLM (from env in llm module; listed here for reference)
# API_URL, API_KEY, LLM_MODEL


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
    faiss_index_path = FAISS_INDEX_PATH


settings = Settings()
