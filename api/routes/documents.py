"""
Document API: list, upload, and delete PDFs in the RAG knowledge base.

Flow for upload:
  1. Validate PDF and extract text per page.
  2. Save the file to the uploads directory.
  3. Chunk, embed, and index the text in the vector store (ChromaDB + BM25).
"""

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


# -----------------------------------------------------------------------------
# PDF text extraction
# -----------------------------------------------------------------------------


def _extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Read all pages of a PDF and concatenate their text."""
    reader = PdfReader(io.BytesIO(pdf_bytes))
    return "".join(page.extract_text() or "" for page in reader.pages)


def _extract_text_per_page(pdf_bytes: bytes) -> List[Tuple[int, str]]:
    """
    Extract text from each page separately.
    Returns a list of (page_number_one_based, page_text).
    """
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages = []
    for page_index in range(len(reader.pages)):
        page_number = page_index + 1
        text = (reader.pages[page_index].extract_text() or "").strip()
        pages.append((page_number, text))
    return pages


# -----------------------------------------------------------------------------
# API endpoints
# -----------------------------------------------------------------------------


@router.get("")
async def list_documents() -> dict:
    """
    List all document names currently in the vector store.
    Returns: {"documents": ["name1.pdf", "name2.pdf", ...]}
    """
    create_index()
    document_names = list_document_names()
    return {"documents": document_names}


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)) -> dict:
    """
    Upload a PDF: save to disk, extract text, chunk, embed, and index.
    Returns: {"filename": "...", "chunks_indexed": N, "errors": [...]}
    Rejects non-PDF files and PDFs that already exist (409).
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="PDF file required")

    pdf_bytes = await file.read()
    pages = _extract_text_per_page(pdf_bytes)

    if not pages or not any(text for _, text in pages):
        raise HTTPException(status_code=400, detail="No text extracted from PDF")

    # Save file to uploads directory
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    save_path = settings.upload_dir / file.filename
    with open(save_path, "wb") as f:
        f.write(pdf_bytes)

    create_index()
    existing_names = list_document_names()
    if file.filename in existing_names:
        raise HTTPException(
            status_code=409,
            detail=f"Document '{file.filename}' already exists",
        )

    chunks_indexed, errors = await process_and_index_document_with_pages(
        pages, file.filename
    )
    return {
        "filename": file.filename,
        "chunks_indexed": chunks_indexed,
        "errors": errors,
    }


@router.delete("/{document_name:path}")
async def delete_document(document_name: str) -> dict:
    """
    Delete a document everywhere:
      1. Remove all its chunks from the vector store (ChromaDB + BM25).
      2. Remove the PDF file from the uploads directory.
    Path parameter is URL-decoded by FastAPI.
    Returns: {"deleted": number_of_chunks_removed}
    """
    create_index()
    result = delete_documents_by_document_name(document_name)

    file_path = settings.upload_dir / document_name
    if file_path.exists():
        file_path.unlink()
        logger.info("Removed uploaded file: %s", file_path)

    return result
