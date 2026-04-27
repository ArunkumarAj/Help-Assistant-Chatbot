"""
SQLite database for Support Cases: list and create cases.

- schema: init_db(), create cases table
- cases_repo: list_cases(), create_case()
- seed_mock_data: load mock cases for testing
"""

from database.schema import init_db
from database.cases_repo import list_cases, create_case

__all__ = ["init_db", "list_cases", "create_case"]
