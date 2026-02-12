"""Streamlit app config: backend API URL."""
import os
from dotenv import load_dotenv
load_dotenv(override=True)

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
