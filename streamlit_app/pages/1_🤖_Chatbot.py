"""Chatbot page: calls POST /chat API. RAG citations shown as details icon with hover (document id + page)."""
import html
import logging
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
from streamlit_app.api_client import chat as api_chat
from streamlit_app.config import API_BASE_URL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Chatbot", page_icon="🤖")
st.markdown(
    """
    <style>
    body { background-color: #f0f8ff; color: #002B5B; }
    .sidebar .sidebar-content { background-color: #006d77; color: white; padding: 20px; border-right: 2px solid #003d5c; }
    .sidebar h2, .sidebar h4 { color: white; }
    .block-container { background-color: white; border-radius: 10px; padding: 20px; box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.1); }
    .footer-text { font-size: 1.1rem; font-weight: bold; color: black; text-align: center; margin-top: 10px; }
    .stButton button { background-color: #118ab2; color: white; border-radius: 5px; padding: 10px 20px; font-size: 16px; }
    .stButton button:hover { background-color: #07a6c2; color: white; }
    h1, h2, h3, h4 { color: #006d77; }
    .stChatMessage { background-color: #e0f7fa; color: #006d77; padding: 10px; border-radius: 5px; margin-bottom: 10px; }
    .stChatMessage.user { background-color: #118ab2; color: white; }
    .rag-cite { display: inline-block; width: 16px; height: 16px; border-radius: 50%; background: #5eb3d6; color: white; text-align: center; line-height: 16px; font-size: 11px; font-weight: bold; cursor: help; vertical-align: middle; margin-left: 2px; }
    </style>
    """,
    unsafe_allow_html=True,
)


def _response_to_display_html(response_text: str, citations: list) -> str:
    """Replace [1], [2], ... with a details icon; hover shows document id and page."""
    if not citations:
        return html.escape(response_text)
    text_escaped = html.escape(response_text)
    # Build lookup: index (1-based) -> tooltip string
    by_index = {c.get("index"): c for c in citations if c.get("index")}
    # Replace from high index first so [10] is not confused with [1]
    for idx in sorted(by_index.keys(), reverse=True):
        c = by_index[idx]
        doc_name = c.get("document_name") or "—"
        page = c.get("page")
        page_str = str(page) if page is not None else "—"
        doc_id = c.get("doc_id") or "—"
        tooltip = f"Document: {doc_name} | Page: {page_str} | ID: {doc_id}"
        icon = f'<span class="rag-cite" title="{html.escape(tooltip)}">i</span>'
        text_escaped = re.sub(r"\[" + str(idx) + r"\]", icon, text_escaped)
    return text_escaped


def render_chatbot_page() -> None:
    st.title("Chatbot 🤖")

    if "use_rag" not in st.session_state:
        st.session_state["use_rag"] = True
    if "num_results" not in st.session_state:
        st.session_state["num_results"] = 5
    if "temperature" not in st.session_state:
        st.session_state["temperature"] = 0.7
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    st.session_state["use_rag"] = st.sidebar.checkbox("Enable RAG mode", value=st.session_state["use_rag"])
    st.session_state["num_results"] = st.sidebar.number_input(
        "Number of Results in Context Window", min_value=1, max_value=10,
        value=st.session_state["num_results"], step=1,
    )
    st.session_state["temperature"] = st.sidebar.slider(
        "Response Temperature", min_value=0.0, max_value=1.0,
        value=st.session_state["temperature"], step=0.1,
    )
    st.sidebar.markdown("<h2 style='text-align: center;'>RAG Chatbot</h2>", unsafe_allow_html=True)
    st.sidebar.markdown("<h4 style='text-align: center;'>Your Conversational Platform</h4>", unsafe_allow_html=True)
    st.sidebar.markdown("<div class='footer-text'>© 2024</div>", unsafe_allow_html=True)

    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant" and msg.get("content_is_html"):
                st.markdown(msg["content"], unsafe_allow_html=True)
            else:
                st.markdown(msg["content"])

    if prompt := st.chat_input("Type your message here..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state["chat_history"].append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("Generating response..."):
                try:
                    data = api_chat(
                        query=prompt,
                        use_rag=st.session_state["use_rag"],
                        num_results=st.session_state["num_results"],
                        temperature=st.session_state["temperature"],
                        chat_history=st.session_state["chat_history"][:-1],
                    )
                    response_text = data.get("response", "")
                    citations = data.get("citations") or []
                    display_html = _response_to_display_html(response_text, citations)
                    st.markdown(display_html, unsafe_allow_html=True)
                    to_store = display_html
                    content_is_html = True
                except Exception as e:
                    to_store = f"Error calling API ({API_BASE_URL}): {e!s}"
                    content_is_html = False
                    st.markdown(to_store)
        st.session_state["chat_history"].append({
            "role": "assistant",
            "content": to_store,
            "content_is_html": content_is_html,
        })


if __name__ == "__main__":
    render_chatbot_page()
