"""
Repository for Support Cases: list and create.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from database.schema import get_connection, init_db


def list_cases(status: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Return all cases, optionally filtered by status (e.g. 'Active', 'Closed').
    Each row is {"id", "title", "description", "status", "created_at"}.
    """
    init_db()
    conn = get_connection()
    try:
        conn.row_factory = _row_factory
        cur = conn.cursor()
        if status:
            cur.execute(
                "SELECT id, title, description, status, created_at FROM cases WHERE status = ? ORDER BY created_at DESC",
                (status.strip(),),
            )
        else:
            cur.execute(
                "SELECT id, title, description, status, created_at FROM cases ORDER BY created_at DESC"
            )
        rows = cur.fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def create_case(
    title: str,
    description: Optional[str] = None,
    status: str = "Active",
) -> Dict[str, Any]:
    """
    Insert a new case and return it with id and created_at.
    """
    init_db()
    created_at = datetime.now(timezone.utc).isoformat()
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO cases (title, description, status, created_at) VALUES (?, ?, ?, ?)",
            (title.strip(), (description or "").strip() or None, status.strip(), created_at),
        )
        conn.commit()
        row_id = cur.lastrowid
        return {
            "id": row_id,
            "title": title.strip(),
            "description": (description or "").strip() or None,
            "status": status.strip(),
            "created_at": created_at,
        }
    finally:
        conn.close()


def _row_factory(cursor: Any, row: tuple) -> dict:
    """Sqlite row_factory: return a dict with column names as keys."""
    names = [d[0] for d in cursor.description]
    return dict(zip(names, row))
