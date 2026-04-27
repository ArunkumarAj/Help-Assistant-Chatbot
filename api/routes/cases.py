"""
Cases API: list and create support cases (SQLite).

- GET /cases?status=Active — list cases, optional status filter
- POST /cases — create a new case (title, optional description, status default Active)
"""

from typing import Any, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from database import init_db, list_cases, create_case

router = APIRouter()


# -----------------------------------------------------------------------------
# Request/response models
# -----------------------------------------------------------------------------


class CreateCaseRequest(BaseModel):
    """Body for POST /cases."""
    title: str
    description: Optional[str] = None
    status: str = "Active"


class CaseItem(BaseModel):
    """One case for API response."""
    id: int
    title: str
    description: Optional[str]
    status: str
    created_at: str


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------


@router.get("", response_model=List[CaseItem])
async def get_cases(status: Optional[str] = None) -> List[CaseItem]:
    """
    List all cases. Optionally filter by status (e.g. Active, Closed).
    Example: GET /cases?status=Active
    """
    init_db()
    rows = list_cases(status=status)
    return [CaseItem(**r) for r in rows]


@router.post("", response_model=CaseItem, status_code=201)
async def post_case(req: CreateCaseRequest) -> CaseItem:
    """
    Create a new case. Default status is Active.
    """
    if not (req.title and req.title.strip()):
        raise HTTPException(status_code=400, detail="title is required")
    init_db()
    case = create_case(
        title=req.title.strip(),
        description=req.description,
        status=req.status or "Active",
    )
    return CaseItem(**case)
