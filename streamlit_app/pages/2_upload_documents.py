"""
Upload Documents page: list, upload, and delete PDFs via the API.

Calls GET /documents, POST /documents/upload, DELETE /documents/{name}.
Large PDFs may take several minutes to process.
"""
import logging
import sys
import time
from pathlib import Path

import streamlit as st

# Project root on path for imports
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from streamlit_app.api_client import list_documents, upload_document, delete_document
from streamlit_app.config import API_BASE_URL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Page config and style
# -----------------------------------------------------------------------------

st.set_page_config(page_title="Upload Documents", page_icon="📂")
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
    .stButton.delete-button button { background-color: #e63946; color: white; font-size: 14px; }
    .stButton.delete-button button:hover { background-color: #ff4c4c; }
    h1, h2, h3, h4 { color: #006d77; }
    </style>
    """,
    unsafe_allow_html=True,
)
st.sidebar.markdown("<h2 style='text-align: center;'>Document Assistant</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<h4 style='text-align: center;'>Upload & Manage PDFs</h4>", unsafe_allow_html=True)
st.sidebar.markdown("<div class='footer-text'>© 2024</div>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Upload page logic
# -----------------------------------------------------------------------------


def _render_upload_page() -> None:
    st.title("Upload Documents")

    if "documents" not in st.session_state:
        st.session_state["documents"] = []

    if "deleted_file" in st.session_state:
        default_msg = f"Deleted '{st.session_state['deleted_file']}' from the knowledge base and uploaded files."
        msg = st.session_state.get("deleted_message", default_msg)
        st.success(msg)
        del st.session_state["deleted_file"]
        if "deleted_message" in st.session_state:
            del st.session_state["deleted_message"]

    try:
        document_names = list_documents()
    except Exception as e:
        st.error(f"Cannot reach API at {API_BASE_URL}. Start it with: uvicorn api.main:app --reload --port 8000")
        st.code("uvicorn api.main:app --reload --host 0.0.0.0 --port 8000")
        return

    st.session_state["documents"] = [
        {"filename": name, "file_path": None} for name in document_names
    ]

    st.caption(
        "Large PDFs may take several minutes to process (extract, chunk, embed, index). Please wait."
    )
    uploaded_files = st.file_uploader(
        "Upload PDF documents", type="pdf", accept_multiple_files=True
    )

    if uploaded_files:
        with st.spinner("Uploading and processing… This can take a few minutes for large files."):
            for uploaded_file in uploaded_files:
                if uploaded_file.name in document_names:
                    st.warning(f"'{uploaded_file.name}' already exists.")
                    continue
                try:
                    upload_document(uploaded_file.getvalue(), uploaded_file.name)
                    st.session_state["documents"].append({
                        "filename": uploaded_file.name,
                        "file_path": None,
                    })
                    document_names.append(uploaded_file.name)
                except Exception as e:
                    st.error(f"Upload failed for {uploaded_file.name}: {e}")
        st.success("Upload complete.")

    if st.session_state["documents"]:
        st.markdown("### Uploaded Documents")
        with st.expander("Manage Uploaded Documents", expanded=True):
            for idx, doc in enumerate(st.session_state["documents"], 1):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.write(f"{idx}. {doc['filename']}")
                with col2:
                    if st.button(
                        "Delete",
                        key=f"del_{doc['filename']}_{idx}",
                        help=f"Delete {doc['filename']} from knowledge base and uploaded files",
                    ):
                        try:
                            result = delete_document(doc["filename"])
                            chunks_removed = result.get("deleted", 0)
                            st.session_state["documents"].pop(idx - 1)
                            st.session_state["deleted_file"] = doc["filename"]
                            st.session_state["deleted_message"] = (
                                f"Deleted '{doc['filename']}' from the knowledge base "
                                f"({chunks_removed} chunks removed) and from uploaded files."
                            )
                            time.sleep(0.5)
                            st.rerun()
                        except Exception as e:
                            st.error(str(e))


if __name__ == "__main__":
    _render_upload_page()
