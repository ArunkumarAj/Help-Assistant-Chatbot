"""
Detect cases-related intents from the user query and format DB results for the chat.

- List cases: "get me all open cases", "list active cases", "show open cases", etc.
- Create case: "create a case", "I want to create a case", "open a new case", etc.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

from database import list_cases, create_case

# Phrases that indicate the user wants to list (open/active) cases.
LIST_INTENT_PATTERNS = [
    r"\b(?:get|show|list|fetch|find)(?:\s+me)?\s+all?\s+(?:open|active)\s+cases?\b",
    r"\b(?:all|open|active)\s+cases?\b",
    r"\blist\s+(?:all\s+)?(?:open|active)?\s*cases?\b",
    r"\b(?:what|which)\s+are\s+(?:my\s+)?(?:open|active)\s+cases\??",
    r"\bopen\s+cases\b",
    r"\bactive\s+cases\b",
]

# Phrases that indicate the user wants to create a case.
CREATE_INTENT_PATTERNS = [
    r"\b(?:i\s+)?(?:would\s+like\s+to|want\s+to|need\s+to)\s+create\s+a\s+case\b",
    r"\bcreate\s+(?:a\s+)?(?:new\s+)?case\b",
    r"\bopen\s+a\s+(?:new\s+)?case\b",
    r"\bi\s+like\s+to\s+create\s+a\s+case\b",
]


def _normalize(query: str) -> str:
    return (query or "").strip().lower()


def detect_list_cases(query: str) -> bool:
    """True if the user is asking to list (open/active) cases."""
    q = _normalize(query)
    if not q:
        return False
    for pat in LIST_INTENT_PATTERNS:
        if re.search(pat, q, re.IGNORECASE):
            return True
    return False


def detect_create_case(query: str) -> bool:
    """True if the user is asking to create a case."""
    q = _normalize(query)
    if not q:
        return False
    for pat in CREATE_INTENT_PATTERNS:
        if re.search(pat, q, re.IGNORECASE):
            return True
    return False


def _extract_create_case_title(query: str) -> str:
    """Try to extract a case title from the query; otherwise return a default."""
    q = (query or "").strip()
    # "create a case about X" / "create a case: X" / "create case for X"
    for sep in ["about", ":", "for", "regarding"]:
        if sep in q:
            parts = q.split(sep, 1)
            if len(parts) == 2 and parts[1].strip():
                return parts[1].strip()[:200]
    return "Case from chat"


def handle_list_cases(status: Optional[str] = "Active") -> Tuple[str, List[Dict[str, Any]]]:
    """
    Fetch cases from DB and return a formatted response string and empty citations.
    status: typically 'Active' for "open" cases.
    """
    cases = list_cases(status=status)
    if not cases:
        return "There are no active open cases at the moment.", []
    lines = [f"**Active open cases ({len(cases)}):**", ""]
    for c in cases:
        title = c.get("title") or "No title"
        case_id = c.get("id", "?")
        desc = (c.get("description") or "").strip()
        if desc:
            lines.append(f"- **[{case_id}]** {title} — {desc}")
        else:
            lines.append(f"- **[{case_id}]** {title}")
    return "\n".join(lines), []


def handle_create_case(query: str) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Create a new Active case (title from query or default) and return confirmation and empty citations.
    """
    title = _extract_create_case_title(query)
    case = create_case(title=title, description=None, status="Active")
    return (
        f"I've created an **Active** case: **{case['title']}** (ID: {case['id']}). "
        "You can add more details or check status via support.",
        [],
    )


def try_cases_intent(query: str) -> Optional[Tuple[str, List[Dict[str, Any]]]]:
    """
    If the query is a list-cases or create-case intent, handle it and return (response_text, citations).
    Otherwise return None and the caller should proceed with RAG.
    """
    if detect_list_cases(query):
        return handle_list_cases(status="Active")
    if detect_create_case(query):
        return handle_create_case(query)
    return None
