"""
SQLite schema for Support Cases.

Creates the cases table and ensures the DB file and directory exist.
"""

import logging
import os
import sqlite3

from core.config import settings

logger = logging.getLogger(__name__)

CREATE_CASES_TABLE = """
CREATE TABLE IF NOT EXISTS cases (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    description TEXT,
    status TEXT NOT NULL DEFAULT 'Active',
    created_at TEXT NOT NULL
);
"""


def get_connection():
    """Return a connection to the SQLite DB. Creates the file and directory if missing."""
    path = getattr(settings, "sqlite_db_path", None) or str(settings.data_dir / "cases.db")
    dir_path = os.path.dirname(path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    return sqlite3.connect(path)


def init_db() -> None:
    """Create the cases table if it does not exist. Safe to call multiple times."""
    conn = get_connection()
    try:
        conn.execute(CREATE_CASES_TABLE)
        conn.commit()
        logger.info("Cases DB initialized at %s", getattr(settings, "sqlite_db_path", ""))
    finally:
        conn.close()
