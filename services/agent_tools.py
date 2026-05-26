"""
LangChain tool definitions for the Help Support agent: RAG search, open cases, create case.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import List

from langchain_core.tools import tool

from database import list_cases, create_case
from services.rag_helpers import retrieve_hybrid


@dataclass
class CitationHolder:
    """Fills on each knowledge-base search; returned to the API for the UI."""
    items: list = field(default_factory=list)


def make_agent_tools(
    top_k: int,
    holder: CitationHolder,
) -> list:
    """Build tool callables for this request (top_k, citation target)."""

    @tool
    def search_knowledge_base(user_query: str) -> str:
        """Search the Help Support knowledge base (RAG) for policy, acronyms, and article text. Use for questions that need information from official documents, not for listing or creating support cases in the database."""
        q = (user_query or "").strip() or "."
        _hits, context, meta = retrieve_hybrid(q, top_k)
        holder.items = list(meta)
        if not context.strip():
            return "No knowledge base articles were found for that query. Say you do not have that information in the knowledge base and suggest support contact."
        return (
            "Retrieved context (each block is numbered; cite in your answer with [1], [2], etc. as appropriate):\n\n"
            + context
        )

    @tool
    def get_open_active_cases() -> str:
        """List all open Active support cases from the local SQLite database. Use when the user asks for open cases, active cases, a list of cases, or to show their cases with status Active."""
        cases = list_cases(status="Active")
        if not cases:
            return "No Active cases in the database."
        out = [f"Active open cases: {len(cases)}", ""]
        for c in cases:
            t = c.get("title") or "No title"
            cid = c.get("id", "?")
            desc = (c.get("description") or "").strip()
            out.append(f"[{cid}] {t}" + (f" — {desc}" if desc else ""))
        return "\n".join(out)

    @tool
    def create_active_support_case(
        title: str,
        description: str = "",
    ) -> str:
        """Create a new support case with status Active. Use when the user wants to open, file, or create a case. Provide a short title; description is optional."""
        t = (title or "").strip() or "Case from chat"
        case = create_case(
            title=t,
            description=(description or "").strip() or None,
            status="Active",
        )
        return json.dumps(
            {
                "created": True,
                "id": case["id"],
                "title": case["title"],
                "status": "Active",
                "message": "Case created successfully (Active).",
            }
        )

    return [search_knowledge_base, get_open_active_cases, create_active_support_case]
