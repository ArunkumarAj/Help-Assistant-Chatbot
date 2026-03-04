"""
RAG chat endpoint: single response per request (no streaming).

Accepts query, optional RAG toggle, num_results, temperature, and chat history.
Returns response text (with [1], [2] citation markers) and citation metadata for the UI.
"""
from typing import Any, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services.rag import chat_response

router = APIRouter()


# -----------------------------------------------------------------------------
# Request/response models
# -----------------------------------------------------------------------------


class ChatMessage(BaseModel):
    """Single message in chat history."""
    role: str
    content: str


class ChatRequest(BaseModel):
    """Body for POST /chat."""
    query: str
    use_rag: bool = True
    num_results: int = 5
    temperature: float = 0.7
    chat_history: Optional[List[ChatMessage]] = None


class CitationItem(BaseModel):
    """One citation for the UI (e.g. details icon tooltip: document name, page, doc_id)."""
    index: int
    document_name: str
    page: Any  # int or None
    doc_id: str


class ChatResponse(BaseModel):
    """Response body: answer text and list of citations."""
    response: str
    citations: List[CitationItem] = []


# -----------------------------------------------------------------------------
# Endpoint
# -----------------------------------------------------------------------------


@router.post("", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    """
    Generate one RAG chat response.
    If use_rag is true, relevant chunks are retrieved and included in the prompt.
    Returns response text (with [1], [2] markers) and citations for the UI.
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query is required")

    history = []
    if req.chat_history:
        history = [{"role": m.role, "content": m.content} for m in req.chat_history]

    response_text, citation_meta = await chat_response(
        query=req.query,
        use_rag=req.use_rag,
        num_results=req.num_results,
        temperature=req.temperature,
        chat_history=history,
    )
    return ChatResponse(
        response=response_text,
        citations=[CitationItem(**c) for c in citation_meta],
    )
