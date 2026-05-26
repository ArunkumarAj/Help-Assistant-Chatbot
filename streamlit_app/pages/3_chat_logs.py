"""
Chat Logs page: view every chat turn in a table (time, json_request, json_response).

Run via Streamlit multipage nav: sidebar → Chat logs
"""
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from streamlit_app.chat_logs_view import render_chat_logs_page


def _apply_page_style() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
        html, body, [class*="css"] { font-family: 'Inter', system-ui, sans-serif; }
        .main .block-container { padding-top: 1.5rem; max-width: 1200px; }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        }
        [data-testid="stSidebar"] label, [data-testid="stSidebar"] p, [data-testid="stSidebar"] .stMarkdown {
            color: #e2e8f0 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown("### Chat logs")
        st.caption("Inspect request/response JSON for each chat turn")
        st.divider()
        st.caption("© 2026 · Help Support Assistant")


st.set_page_config(
    page_title="Chat logs",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded",
)

_apply_page_style()
_render_sidebar()
render_chat_logs_page()
