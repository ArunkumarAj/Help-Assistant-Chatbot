"""RAG chat: single response (no streaming from API for simplicity)."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Any, List, Optional

from services.rag import chat_response

router = APIRouter()


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    query: str
    use_rag: bool = True
    num_results: int = 5
    temperature: float = 0.7
    chat_history: Optional[List[ChatMessage]] = None


class CitationItem(BaseModel):
    index: int
    document_name: str
    page: Any  # int or None
    doc_id: str


class ChatResponse(BaseModel):
    response: str
    citations: List[CitationItem] = []  # for UI: details icon tooltip (document id + page)


@router.post("", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    """Generate a single RAG chat response. Returns response text (with [1],[2] markers) and citations for UI."""
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query is required")
    history = []
    if req.chat_history:
        history = [{"role": m.role, "content": m.content} for m in req.chat_history]
    text, citations = await chat_response(
        query=req.query,
        use_rag=req.use_rag,
        num_results=req.num_results,
        temperature=req.temperature,
        chat_history=history,
    )
    return ChatResponse(
        response=text,
        citations=[CitationItem(**c) for c in citations],
    )
