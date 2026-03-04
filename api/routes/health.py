"""
Health check endpoint for the RAG API.

Used by load balancers and the Streamlit app to verify the backend is up.
"""
from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health():
    """Return status and service name. No auth required."""
    return {"status": "ok", "service": "rag-api"}
