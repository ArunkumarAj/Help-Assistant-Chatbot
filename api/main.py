"""
FastAPI application: RAG chat and document APIs.

- /health: liveness check
- /documents: list, upload, delete PDFs (ChromaDB + BM25)
- /chat: single RAG chat response (optional retrieval, citations in response)

On startup, log/data/upload directories are created. CORS allows all origins.
"""
from contextlib import asynccontextmanager
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import health, documents, chat, cases
from core.config import settings
from core.logging_config import setup_logging

setup_logging()


# -----------------------------------------------------------------------------
# Lifespan: ensure directories exist
# -----------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """On startup: create log, data, and upload directories if missing."""
    settings.log_dir.mkdir(parents=True, exist_ok=True)
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    yield


# -----------------------------------------------------------------------------
# App and routes
# -----------------------------------------------------------------------------

app = FastAPI(
    title="RAG Document API",
    description="Async API for document upload, ChromaDB vector store (hybrid search), and RAG chat.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, tags=["health"])
app.include_router(documents.router, prefix="/documents", tags=["documents"])
app.include_router(chat.router, prefix="/chat", tags=["chat"])
app.include_router(cases.router, prefix="/cases", tags=["cases"])


if __name__ == "__main__":
    import uvicorn
    # Do not watch .venv / site-packages — changes there (e.g. tqdm) trigger reloads and can
    # interrupt the worker mid-import on Windows (KeyboardInterrupt / odd tracebacks).
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_excludes=[".venv", "**/.venv/**", "**/site-packages/**", "**/Lib/site-packages/**"],
    )
