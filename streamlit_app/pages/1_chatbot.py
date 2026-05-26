"""
Chatbot page: ChatGPT-style UI with RAG controls in the sidebar.
"""
import html
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
from streamlit_app.api_client import chat as api_chat
from streamlit_app.config import API_BASE_URL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHAT_INPUT_PLACEHOLDER = "Ask something about your documents..."


# -----------------------------------------------------------------------------
# Styling
# -----------------------------------------------------------------------------


def _apply_page_style() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', system-ui, sans-serif;
        }

        .main .block-container {
            padding-top: 1rem;
            padding-bottom: 6rem;
            max-width: 48rem;
        }

        .chat-header {
            position: sticky;
            top: 0;
            z-index: 100;
            background: rgba(255, 255, 255, 0.92);
            backdrop-filter: blur(8px);
            border-bottom: 1px solid #e5e7eb;
            padding: 0.75rem 0 1rem 0;
            margin-bottom: 0.5rem;
        }
        .chat-header h1 {
            font-size: 1.35rem;
            font-weight: 600;
            color: #111827;
            margin: 0;
            letter-spacing: -0.02em;
        }
        .chat-header p {
            font-size: 0.8rem;
            color: #6b7280;
            margin: 0.25rem 0 0 0;
        }

        [data-testid="stChatMessage"] {
            padding: 0.65rem 0;
            margin-bottom: 0.35rem;
        }

        div[data-testid="stChatMessageContent"] {
            border-radius: 1rem;
            padding: 0.85rem 1.1rem;
            line-height: 1.55;
            font-size: 0.95rem;
        }

        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"])
        div[data-testid="stChatMessageContent"] {
            background: #2563eb !important;
            color: #ffffff !important;
            border: none !important;
            border-bottom-right-radius: 0.25rem !important;
        }

        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"])
        div[data-testid="stChatMessageContent"] {
            background: #f3f4f6 !important;
            color: #1f2937 !important;
            border: 1px solid #e5e7eb !important;
            border-bottom-left-radius: 0.25rem !important;
        }

        .assistant-body p {
            margin-bottom: 0.65rem;
        }
        .assistant-body ul, .assistant-body ol {
            margin: 0.5rem 0 0.75rem 1.1rem;
        }
        .assistant-body strong {
            color: #111827;
        }

        .sources-block {
            margin-top: 0.85rem;
            padding-top: 0.65rem;
            border-top: 1px dashed #d1d5db;
        }
        .sources-label {
            font-size: 0.7rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: #6b7280;
            margin-bottom: 0.4rem;
        }
        .source-chip {
            display: inline-block;
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 0.35rem 0.65rem;
            margin: 0.2rem 0.35rem 0.2rem 0;
            font-size: 0.8rem;
            color: #374151;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        }
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] .stMarkdown,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span {
            color: #e2e8f0 !important;
        }
        [data-testid="stSidebar"] hr {
            border-color: rgba(148, 163, 184, 0.3);
        }

        [data-testid="stChatInput"] textarea {
            border-radius: 1.25rem !important;
            border: 1px solid #d1d5db !important;
            padding: 0.75rem 1rem !important;
        }
        [data-testid="stChatInput"] textarea:focus {
            border-color: #2563eb !important;
            box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.15) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# Message rendering
# -----------------------------------------------------------------------------


def _normalize_history_entry(msg: Dict[str, Any]) -> Dict[str, Any]:
    """Support legacy history entries that used HTML-only assistant content."""
    if msg.get("role") != "assistant":
        return msg
    if "citations" in msg:
        return msg
    if msg.get("content_is_html") and msg.get("content"):
        return {
            "role": "assistant",
            "content": _strip_html_to_text(msg["content"]),
            "citations": [],
        }
    return {
        "role": "assistant",
        "content": msg.get("content") or "",
        "citations": msg.get("citations") or [],
    }


def _strip_html_to_text(raw: str) -> str:
    text = re.sub(r"<[^>]+>", "", raw)
    return html.unescape(text).strip()


def _enhance_markdown(text: str) -> str:
    if not text:
        return ""
    lines: List[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(("- ", "* ", "• ")):
            bullet = stripped[2:].strip() if stripped[0] in "*•" else stripped[2:].strip()
            lines.append(f"- {bullet}" if not stripped.startswith("-") else stripped)
        else:
            lines.append(line)
    return "\n".join(lines)


def _render_sources(citations: List[Dict[str, Any]]) -> None:
    if not citations:
        return
    chips = []
    for c in citations:
        doc = html.escape(str(c.get("document_name") or "Document"))
        page = c.get("page")
        idx = c.get("index", "")
        page_part = f" · p. {page}" if page is not None else ""
        chips.append(f'<span class="source-chip">[{idx}] {doc}{page_part}</span>')
    st.markdown(
        '<div class="sources-block">'
        '<div class="sources-label">Sources</div>'
        f'{"".join(chips)}</div>',
        unsafe_allow_html=True,
    )


def _render_assistant_message(content: str, citations: Optional[List[Dict[str, Any]]] = None) -> None:
    st.markdown(_enhance_markdown(content))
    _render_sources(citations or [])


def _render_user_message(content: str) -> None:
    st.markdown(content)


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------


def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown("### RAG controls")
        st.caption("Applied to each new message")

        st.session_state["use_rag"] = st.checkbox(
            "Enable RAG",
            value=st.session_state.get("use_rag", True),
            help="Retrieve context from uploaded documents before answering.",
        )
        st.session_state["num_results"] = st.number_input(
            "Number of results",
            min_value=1,
            max_value=10,
            value=int(st.session_state.get("num_results", 5)),
            step=1,
            help="Chunks passed into the context window.",
        )
        st.session_state["temperature"] = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=float(st.session_state.get("temperature", 0.7)),
            step=0.1,
            help="Higher = more creative; lower = more focused.",
        )

        st.divider()
        if st.button("Clear conversation", use_container_width=True):
            st.session_state["chat_history"] = []
            st.session_state.pop("_last_chat_input", None)
            st.rerun()

        st.divider()
        st.caption(f"API: {API_BASE_URL}")


# -----------------------------------------------------------------------------
# Chat UI
# -----------------------------------------------------------------------------


def _init_session() -> None:
    for key, val in (
        ("use_rag", True),
        ("num_results", 5),
        ("temperature", 0.7),
        ("chat_history", []),
    ):
        if key not in st.session_state:
            st.session_state[key] = val


def _render_header() -> None:
    st.markdown(
        """
        <div class="chat-header">
            <h1>AI Chatbot 🤖</h1>
            <p>Grounded answers from your document knowledge base</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_history() -> None:
    history = st.session_state.get("chat_history") or []
    with st.container(height=520, border=False):
        if not history:
            with st.chat_message("assistant"):
                st.markdown(
                    "Hi! Ask me anything about your uploaded documents, "
                    "or try **list open cases** if case management is enabled."
                )
        for raw in history:
            msg = _normalize_history_entry(raw)
            role = msg.get("role", "user")
            with st.chat_message(role):
                if role == "user":
                    _render_user_message(msg.get("content") or "")
                else:
                    _render_assistant_message(
                        msg.get("content") or "",
                        msg.get("citations") or [],
                    )


def _process_new_message(user_input: str) -> None:
    """Call API and append user + assistant turns to session history."""
    history: List[Dict[str, Any]] = st.session_state["chat_history"]
    history.append({"role": "user", "content": user_input})

    api_history = [
        {"role": m["role"], "content": m["content"]}
        for m in history[:-1]
        if m.get("role") in ("user", "assistant") and m.get("content")
    ]

    with st.spinner("Thinking..."):
        try:
            data = api_chat(
                query=user_input,
                use_rag=st.session_state["use_rag"],
                num_results=st.session_state["num_results"],
                temperature=st.session_state["temperature"],
                chat_history=api_history,
            )
            response_text = (data.get("response") or "").strip()
            citations = data.get("citations") or []
            history.append({
                "role": "assistant",
                "content": response_text,
                "citations": citations,
            })
        except Exception as e:
            err = (
                f"Could not reach the API at **{API_BASE_URL}**. "
                f"Please ensure the backend is running.\n\n`{e!s}`"
            )
            history.append({
                "role": "assistant",
                "content": err,
                "citations": [],
            })


# -----------------------------------------------------------------------------
# Page entry (Streamlit runs this file directly)
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="AI Chatbot",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
)

_init_session()
_render_sidebar()
_apply_page_style()
_render_header()
_render_history()

if user_input := st.chat_input(CHAT_INPUT_PLACEHOLDER):
    last = st.session_state.get("_last_chat_input")
    if user_input != last:
        st.session_state["_last_chat_input"] = user_input
        _process_new_message(user_input)
        st.rerun()
