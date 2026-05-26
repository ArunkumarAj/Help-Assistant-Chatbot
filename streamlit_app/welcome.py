"""
Streamlit home page (entry point).

Run with: streamlit run streamlit_app/welcome.py
"""
import json
import re
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import settings
from streamlit_app.api_client import health, list_documents
from streamlit_app.chat_logs_view import render_chat_logs_tab
from streamlit_app.config import API_BASE_URL

PAGE_CHAT = "pages/1_chatbot.py"
PAGE_UPLOAD = "pages/2_upload_documents.py"
PAGE_LOGS = "pages/3_chat_logs.py"


# -----------------------------------------------------------------------------
# Styling
# -----------------------------------------------------------------------------


def _apply_page_style() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
        }

        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1100px;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        }
        [data-testid="stSidebar"] * {
            color: #e2e8f0 !important;
        }
        [data-testid="stSidebar"] .stMarkdown h1,
        [data-testid="stSidebar"] .stMarkdown h2,
        [data-testid="stSidebar"] .stMarkdown h3 {
            color: #f8fafc !important;
        }
        [data-testid="stSidebar"] hr {
            border-color: rgba(148, 163, 184, 0.25);
            margin: 1rem 0;
        }
        [data-testid="stSidebar"] [data-testid="stSidebarNav"] a {
            border-radius: 8px;
            padding: 0.35rem 0.5rem;
        }
        [data-testid="stSidebar"] [data-testid="stSidebarNav"] a:hover {
            background: rgba(255, 255, 255, 0.08);
        }

        .dash-hero {
            margin-bottom: 0.5rem;
        }
        .dash-hero h1 {
            font-size: 2.25rem;
            font-weight: 700;
            color: #0f172a;
            letter-spacing: -0.03em;
            margin: 0 0 0.35rem 0;
            line-height: 1.2;
        }
        .dash-hero .tagline {
            font-size: 1.05rem;
            color: #64748b;
            margin: 0;
            font-weight: 400;
        }

        .feature-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 14px;
            padding: 1.35rem 1.25rem 1rem;
            min-height: 220px;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
            transition: box-shadow 0.2s ease, border-color 0.2s ease;
        }
        .feature-card:hover {
            border-color: #cbd5e1;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
        }
        .feature-card .icon {
            font-size: 2rem;
            line-height: 1;
            margin-bottom: 0.65rem;
        }
        .feature-card h3 {
            font-size: 1.1rem;
            font-weight: 600;
            color: #0f172a;
            margin: 0 0 0.5rem 0;
        }
        .feature-card p {
            font-size: 0.875rem;
            color: #64748b;
            line-height: 1.55;
            margin: 0;
            min-height: 4.5rem;
        }

        .section-label {
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #94a3b8;
            margin: 2rem 0 0.75rem 0;
        }

        div[data-testid="stMetric"] {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 1rem 1.1rem;
        }
        div[data-testid="stMetric"] label {
            color: #64748b !important;
            font-size: 0.8rem !important;
        }
        div[data-testid="stMetric"] [data-testid="stMetricValue"] {
            color: #0f172a !important;
            font-weight: 600 !important;
        }

        .cta-wrap {
            margin-top: 2.5rem;
            padding-top: 2rem;
            border-top: 1px solid #e2e8f0;
            text-align: center;
        }

        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
            border: none;
            border-radius: 10px;
            font-weight: 600;
            padding: 0.65rem 1.5rem;
        }
        .stButton > button[kind="primary"]:hover {
            background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%);
            border: none;
            color: white;
        }

        .status-pill {
            display: inline-block;
            padding: 0.2rem 0.65rem;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 600;
        }
        .status-ok { background: rgba(34, 197, 94, 0.2); color: #4ade80; }
        .status-down { background: rgba(248, 113, 113, 0.2); color: #fca5a5; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# Stats helpers
# -----------------------------------------------------------------------------


@st.cache_data(ttl=30, show_spinner=False)
def _count_chat_turns() -> int:
    log_path = Path(settings.chat_log_path)
    if not log_path.is_file():
        return 0
    count = 0
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


@st.cache_data(ttl=30, show_spinner=False)
def _avg_response_time_seconds() -> Optional[float]:
    log_path = Path(settings.log_file_path)
    if not log_path.is_file():
        return None
    pattern = re.compile(r"eval_latency_(?:retrieve|generate)_seconds=([\d.]+)")
    values: list[float] = []
    try:
        with open(log_path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                for match in pattern.finditer(line):
                    values.append(float(match.group(1)))
    except OSError:
        return None
    if not values:
        return None
    recent = values[-40:]
    return sum(recent) / len(recent)


@st.cache_data(ttl=30, show_spinner=False)
def _document_count() -> Tuple[int, bool]:
    try:
        docs = list_documents()
        return len(docs), True
    except Exception:
        return 0, False


@st.cache_data(ttl=15, show_spinner=False)
def _api_health_ok() -> Tuple[bool, float]:
    start = time.perf_counter()
    try:
        health()
        return True, time.perf_counter() - start
    except Exception:
        return False, time.perf_counter() - start


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------


def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown("### Personal Document Assistant")
        st.caption("AI-powered RAG workspace")
        st.divider()

        api_ok, ping_s = _api_health_ok()
        if api_ok:
            st.markdown(
                f'<span class="status-pill status-ok">API online</span>',
                unsafe_allow_html=True,
            )
            st.caption(f"{API_BASE_URL} · {ping_s * 1000:.0f} ms")
        else:
            st.markdown(
                f'<span class="status-pill status-down">API offline</span>',
                unsafe_allow_html=True,
            )
            st.caption(f"Cannot reach {API_BASE_URL}")

        st.divider()
        st.markdown("**Navigate**")
        st.caption("Open **Chat logs** in the sidebar to inspect request/response JSON.")
        st.divider()
        st.caption("© 2026 · Help Support Assistant")


# -----------------------------------------------------------------------------
# UI sections
# -----------------------------------------------------------------------------


def _render_header() -> None:
    st.markdown(
        """
        <div class="dash-hero">
            <h1>Personal Document Assistant</h1>
            <p class="tagline">
                Upload PDFs, search your knowledge base, and chat with grounded AI answers—all in one place.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _feature_card_html(icon: str, title: str, description: str) -> str:
    return f"""
    <div class="feature-card">
        <div class="icon">{icon}</div>
        <h3>{title}</h3>
        <p>{description}</p>
    </div>
    """


def _render_feature_cards() -> None:
    st.markdown('<p class="section-label">Workspace</p>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4, gap="medium")

    cards = [
        (
            col1,
            "📄",
            "Upload Documents",
            "Add PDFs to your knowledge base. Files are chunked, embedded, and indexed for hybrid RAG search.",
            PAGE_UPLOAD,
            "upload_open",
            "Upload",
        ),
        (
            col2,
            "💬",
            "Chatbot",
            "Ask questions with optional RAG context. Get cited answers grounded in your uploaded documents.",
            PAGE_CHAT,
            "chat_open",
            "Open chat",
        ),
        (
            col3,
            "📋",
            "Chat logs",
            "View a table of every chat: timestamp, json_request, and json_response for debugging and audit.",
            PAGE_LOGS,
            "logs_open",
            "View logs",
        ),
        (
            col4,
            "⚙️",
            "RAG settings",
            "Tune retrieval depth, temperature, and RAG mode from the chat sidebar before you start a session.",
            PAGE_CHAT,
            "settings_open",
            "RAG settings",
        ),
    ]

    for col, icon, title, desc, page, key, label in cards:
        with col:
            st.markdown(_feature_card_html(icon, title, desc), unsafe_allow_html=True)
            if st.button(label, key=key, use_container_width=True):
                st.switch_page(page)


def _render_quick_stats() -> None:
    st.markdown('<p class="section-label">Quick stats</p>', unsafe_allow_html=True)

    doc_count, api_reachable = _document_count()
    total_chats = _count_chat_turns()
    avg_latency = _avg_response_time_seconds()
    _, ping_s = _api_health_ok()

    if avg_latency is not None:
        avg_display = f"{avg_latency:.2f}s"
        avg_help = "Average of recent retrieve + generate steps from server logs."
    else:
        avg_display = f"{ping_s * 1000:.0f} ms"
        avg_help = "Health-check latency (enable EVAL_LOGGING_ENABLED for full RAG timings)."

    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        delta = "Indexed via API" if api_reachable else "API unavailable"
        st.metric("Documents", doc_count, delta=delta, help="PDFs currently in the vector store.")
    with c2:
        st.metric("Total chats", total_chats, help="Logged assistant turns in chat_logs.jsonl.")
    with c3:
        st.metric("Avg response time", avg_display, help=avg_help)


def _render_cta() -> None:
    st.markdown('<div class="cta-wrap">', unsafe_allow_html=True)
    _, center, _ = st.columns([1, 2, 1])
    with center:
        if st.button("Start Chatting", type="primary", use_container_width=True, key="cta_start"):
            st.switch_page(PAGE_CHAT)
    st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


st.set_page_config(
    page_title="Personal Document Assistant",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)
_apply_page_style()
_render_sidebar()
_render_header()

tab_overview, tab_logs = st.tabs(["Overview", "Chat logs"])

with tab_overview:
    st.divider()
    _render_feature_cards()
    st.divider()
    _render_quick_stats()
    _render_cta()

with tab_logs:
    st.divider()
    render_chat_logs_tab()
