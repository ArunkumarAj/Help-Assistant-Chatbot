"""
Streamlit app config: backend API base URL.

Read from environment (e.g. .env or deployment). Default is localhost:8000.
"""
import os
from dotenv import load_dotenv

load_dotenv(override=True)

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
