"""
Streamlit welcome page.

Run with: streamlit run streamlit_app/Welcome.py

Adds project root to sys.path so streamlit_app.config and api_client can be
resolved. This page does not call the backend; it only describes the app.
"""
import sys
from pathlib import Path

import streamlit as st

# Ensure project root is on path for config and api_client when running as streamlit run streamlit_app/Welcome.py
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# -----------------------------------------------------------------------------
# Styling
# -----------------------------------------------------------------------------


def _apply_page_style() -> None:
    """Inject CSS for colors and layout."""
    st.markdown(
        """
        <style>
        body { background-color: #f0f8ff; color: #002B5B; }
        .sidebar .sidebar-content { background-color: #006d77; color: white; padding: 20px; border-right: 2px solid #003d5c; }
        .sidebar h2, .sidebar h4 { color: white; }
        .block-container { background-color: white; border-radius: 10px; padding: 20px; box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.1); }
        .footer-text { font-size: 1.1rem; font-weight: bold; color: black; text-align: center; margin-top: 10px; }
        .stButton button { background-color: #118ab2; color: white; border-radius: 5px; border: none; padding: 10px 20px; font-size: 16px; }
        .stButton button:hover { background-color: #07a6c2; color: white; }
        h1, h2, h3, h4 { color: #006d77; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# Page content
# -----------------------------------------------------------------------------

st.set_page_config(page_title="RAG Document Assistant", page_icon="🤖")
_apply_page_style()

st.sidebar.markdown("<h2 style='text-align: center;'>RAG Document Assistant</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<h4 style='text-align: center;'>Your Conversational Platform</h4>", unsafe_allow_html=True)
st.sidebar.markdown("<div class='footer-text'>© 2024</div>", unsafe_allow_html=True)

st.title("Personal Document Assistant 📄🤖")
st.markdown(
    """
    Welcome to the AI-Powered Document Retrieval Assistant 👋

    This app uses a **FastAPI backend** for document upload and RAG chat.

    **Features:**
    - **Chatbot**: Chat with the AI; enable RAG to use your uploaded documents as context.
    - **Upload Documents**: Upload PDFs; they are processed and indexed via the API.

    **Choose a page from the sidebar to begin.**

    Make sure the API is running: `uvicorn api.main:app --reload --host 0.0.0.0 --port 8000`
    """
)
