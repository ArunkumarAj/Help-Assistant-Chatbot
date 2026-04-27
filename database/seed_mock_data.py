"""
Load mock cases into the SQLite DB for testing and demos.

Run from project root:
  python -m database.seed_mock_data

Safe to run multiple times: it only inserts if the table is empty (or use --force to re-seed).
"""

import argparse
import sys
from datetime import datetime, timezone

# Add project root to path when run as script
if __name__ == "__main__":
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from database.schema import get_connection, init_db

MOCK_CASES = [
    {
        "title": "Login failure for dealer portal",
        "description": "User cannot log in with valid credentials; getting 401 after password reset.",
        "status": "Active",
        "created_at": "2026-01-15T10:30:00+00:00",
    },
    {
        "title": "Invoice export missing line items",
        "description": "Exported CSV for January 2026 is missing line items for orders after 15th.",
        "status": "Active",
        "created_at": "2026-01-18T14:00:00+00:00",
    },
    {
        "title": "Password reset email not received",
        "description": "Dealer requested password reset three times; no email received. Check spam folder.",
        "status": "Active",
        "created_at": "2026-01-20T09:15:00+00:00",
    },
    {
        "title": "Order status stuck at Pending",
        "description": "Order #ORD-2026-0042 has been Pending for 48 hours. Payment was confirmed.",
        "status": "Active",
        "created_at": "2026-01-22T11:45:00+00:00",
    },
    {
        "title": "Duplicate charges on bulk order",
        "description": "Bulk order was charged twice. Refund requested for second charge.",
        "status": "Closed",
        "created_at": "2026-01-10T16:20:00+00:00",
    },
    {
        "title": "Catalog PDF download 404",
        "description": "Link to Q4 2025 catalog returns 404. Resolved by updating link in portal.",
        "status": "Closed",
        "created_at": "2026-01-12T08:00:00+00:00",
    },
]


def seed(force: bool = False) -> int:
    """Insert mock cases. If not force, only insert when table is empty. Returns number inserted."""
    init_db()
    conn = get_connection()
    try:
        cur = conn.cursor()
        if not force:
            cur.execute("SELECT COUNT(*) FROM cases")
            if cur.fetchone()[0] > 0:
                print("Cases table already has rows. Use --force to insert mock data anyway.")
                return 0
        for row in MOCK_CASES:
            cur.execute(
                "INSERT INTO cases (title, description, status, created_at) VALUES (?, ?, ?, ?)",
                (row["title"], row["description"], row["status"], row["created_at"]),
            )
        conn.commit()
        print(f"Inserted {len(MOCK_CASES)} mock cases.")
        return len(MOCK_CASES)
    finally:
        conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed SQLite with mock cases.")
    parser.add_argument("--force", action="store_true", help="Insert even if table already has data")
    args = parser.parse_args()
    seed(force=args.force)
