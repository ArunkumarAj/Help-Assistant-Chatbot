"""Document list, upload, delete. Upload: PDF bytes -> extract text -> chunk -> embed -> index."""
import asyncio
import io
import logging
from typing import List, Tuple

from fastapi import APIRouter, File, HTTPException, UploadFile
from PyPDF2 import PdfReader

from core.config import settings
from core.logging_config import setup_logging
from services.ingestion import (
    create_index,
    delete_documents_by_document_name,
    process_and_index_document_with_pages,
)
from vector_store.store import list_document_names

setup_logging()
logger = logging.getLogger(__name__)

router = APIRouter()


def _extract_text_from_pdf(bytes_content: bytes) -> str:
    reader = PdfReader(io.BytesIO(bytes_content))
    return "".join(page.extract_text() or "" for page in reader.pages)


def _extract_text_from_pdf_per_page(bytes_content: bytes) -> List[Tuple[int, str]]:
    """Extract text per page. Returns list of (page_number_1based, text)."""
    reader = PdfReader(io.BytesIO(bytes_content))
    return [(p + 1, (reader.pages[p].extract_text() or "").strip()) for p in range(len(reader.pages))]


@router.get("")
async def list_documents() -> dict:
    """List unique document names in the vector store."""
    create_index()
    names = list_document_names()
    return {"documents": names}


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)) -> dict:
    """Upload a PDF: save to disk, extract text, chunk, embed, index. Async."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="PDF file required")
    content = await file.read()
    pages = _extract_text_from_pdf_per_page(content)
    if not pages or not any(t for _, t in pages):
        raise HTTPException(status_code=400, detail="No text extracted from PDF")
    # Save file to uploads dir
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    save_path = settings.upload_dir / file.filename
    with open(save_path, "wb") as f:
        f.write(content)
    create_index()
    existing = list_document_names()
    if file.filename in existing:
        raise HTTPException(status_code=409, detail=f"Document '{file.filename}' already exists")
    count, errors = await process_and_index_document_with_pages(pages, file.filename)
    return {"filename": file.filename, "chunks_indexed": count, "errors": errors}


@router.delete("/{document_name:path}")
async def delete_document(document_name: str) -> dict:
    """
    Delete document everywhere: (1) all chunks for this document in the vector store (ChromaDB + BM25),
    (2) the PDF file from uploaded_files. Path param is URL-decoded by FastAPI.
    """
    create_index()
    # 1. Remove from vector DB (ChromaDB + BM25) so it no longer appears in RAG search
    result = delete_documents_by_document_name(document_name)
    # 2. Remove file from uploads directory
    file_path = settings.upload_dir / document_name
    if file_path.exists():
        file_path.unlink()
        logger.info("Removed uploaded file: %s", file_path)
    return result
