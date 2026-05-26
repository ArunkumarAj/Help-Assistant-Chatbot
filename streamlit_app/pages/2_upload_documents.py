"""
Upload Documents page: modern document manager UI.
"""
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import settings
from streamlit_app.api_client import delete_document, list_documents, upload_document
from streamlit_app.config import API_BASE_URL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Styling
# -----------------------------------------------------------------------------


def _apply_page_style() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', system-ui, sans-serif;
        }

        .main .block-container {
            padding-top: 1.5rem;
            padding-bottom: 3rem;
            max-width: 52rem;
        }

        .page-hero h1 {
            font-size: 1.75rem;
            font-weight: 700;
            color: #0f172a;
            margin: 0 0 0.35rem 0;
            letter-spacing: -0.02em;
        }
        .page-hero p {
            color: #64748b;
            font-size: 0.95rem;
            margin: 0;
        }

        .section-header {
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #64748b;
            margin: 1.75rem 0 0.75rem 0;
        }

        .upload-dropzone {
            border: 2px dashed #cbd5e1;
            border-radius: 14px;
            background: #f8fafc;
            padding: 1.25rem 1rem 0.5rem;
            margin-bottom: 0.5rem;
        }
        .upload-dropzone p {
            color: #64748b;
            font-size: 0.85rem;
            margin: 0 0 0.5rem 0;
            text-align: center;
        }

        .file-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 0.9rem 1rem;
            margin-bottom: 0.65rem;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }
        .file-card .file-icon {
            font-size: 1.5rem;
            flex-shrink: 0;
        }
        .file-card .file-meta h4 {
            margin: 0;
            font-size: 0.95rem;
            font-weight: 600;
            color: #0f172a;
            word-break: break-word;
        }
        .file-card .file-meta span {
            font-size: 0.8rem;
            color: #64748b;
        }

        .summary-bar {
            background: #f1f5f9;
            border: 1px solid #e2e8f0;
            border-radius: 10px;
            padding: 0.65rem 1rem;
            font-size: 0.875rem;
            color: #475569;
            margin-bottom: 1rem;
        }

        .empty-state {
            text-align: center;
            padding: 2.5rem 1rem;
            color: #94a3b8;
            border: 1px dashed #e2e8f0;
            border-radius: 12px;
            background: #fafafa;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        }
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] .stMarkdown,
        [data-testid="stSidebar"] p {
            color: #e2e8f0 !important;
        }
        [data-testid="stSidebar"] hr {
            border-color: rgba(148, 163, 184, 0.3);
        }

        div[data-testid="stFileUploader"] {
            background: transparent;
        }
        div[data-testid="stFileUploader"] section {
            border: none !important;
            padding: 0 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _human_size(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.1f} KB"
    return f"{num_bytes / (1024 * 1024):.1f} MB"


def _local_file_size(filename: str) -> Optional[int]:
    path = settings.upload_dir / filename
    if path.is_file():
        return path.stat().st_size
    return None


def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown("### Document manager")
        st.caption("Index PDFs for RAG search")
        st.divider()
        st.caption(f"API: {API_BASE_URL}")
        st.divider()
        st.caption("© 2026 · Help Support Assistant")


def _show_flash_messages() -> None:
    if st.session_state.pop("upload_success", None):
        names = st.session_state.pop("upload_success_names", [])
        if len(names) == 1:
            st.success(f"Successfully indexed **{names[0]}**.")
        else:
            st.success(f"Successfully indexed **{len(names)}** document(s).")

    if "deleted_file" in st.session_state:
        msg = st.session_state.pop(
            "deleted_message",
            f"Removed **{st.session_state['deleted_file']}** from the knowledge base.",
        )
        st.session_state.pop("deleted_file", None)
        st.success(msg)

    if st.session_state.pop("delete_all_success", None):
        count = st.session_state.pop("delete_all_count", 0)
        st.success(f"Removed **{count}** document(s) from the knowledge base.")


def _ensure_processed_uploads_set() -> None:
    if "processed_upload_ids" not in st.session_state:
        st.session_state["processed_upload_ids"] = set()


def _upload_file_id(name: str, size: int) -> str:
    return f"{name}:{size}"


def _handle_uploads(
    uploaded_files: List[Any],
    existing_names: List[str],
) -> None:
    _ensure_processed_uploads_set()
    processed = st.session_state["processed_upload_ids"]
    to_upload = []
    for f in uploaded_files:
        uid = _upload_file_id(f.name, f.size)
        if uid in processed:
            continue
        if f.name in existing_names:
            st.warning(f"**{f.name}** is already in your library.")
            processed.add(uid)
            continue
        to_upload.append(f)

    if not to_upload:
        return

    succeeded: List[str] = []
    progress = st.progress(0, text="Processing documents…")
    total = len(to_upload)

    for i, uploaded_file in enumerate(to_upload):
        progress.progress(
            (i) / total,
            text=f"Uploading **{uploaded_file.name}** ({i + 1}/{total})…",
        )
        try:
            with st.spinner(f"Indexing **{uploaded_file.name}** — this may take a few minutes."):
                upload_document(uploaded_file.getvalue(), uploaded_file.name)
            processed.add(_upload_file_id(uploaded_file.name, uploaded_file.size))
            existing_names.append(uploaded_file.name)
            succeeded.append(uploaded_file.name)
        except Exception as e:
            st.error(f"Failed to upload **{uploaded_file.name}**: {e!s}")

    progress.progress(1.0, text="Done")
    progress.empty()

    if succeeded:
        st.session_state["upload_success"] = True
        st.session_state["upload_success_names"] = succeeded
        st.rerun()


def _render_file_row(filename: str, index: int) -> None:
    size_bytes = _local_file_size(filename)
    size_label = _human_size(size_bytes) if size_bytes is not None else "Size unknown"

    col_info, col_del = st.columns([5, 1], vertical_alignment="center")
    with col_info:
        st.markdown(f"**📄 {filename}**")
        st.caption(size_label)
    with col_del:
        if st.button(
            "Delete",
            key=f"del_{filename}_{index}",
            type="secondary",
            use_container_width=True,
            help=f"Remove {filename} from the knowledge base",
        ):
            try:
                result = delete_document(filename)
                chunks_removed = result.get("deleted", 0)
                st.session_state["deleted_file"] = filename
                st.session_state["deleted_message"] = (
                    f"Deleted **{filename}** ({chunks_removed} chunks removed) "
                    "from the knowledge base and uploaded files."
                )
                st.rerun()
            except Exception as e:
                st.error(str(e))


def _delete_all(document_names: List[str]) -> None:
    if not document_names:
        return
    progress = st.progress(0, text="Deleting all documents…")
    removed = 0
    total = len(document_names)
    for i, name in enumerate(document_names):
        progress.progress((i) / total, text=f"Removing **{name}**…")
        try:
            delete_document(name)
            removed += 1
        except Exception as e:
            st.error(f"Could not delete **{name}**: {e!s}")
    progress.empty()
    st.session_state["delete_all_success"] = True
    st.session_state["delete_all_count"] = removed
    st.session_state["processed_upload_ids"] = set()
    st.rerun()


# -----------------------------------------------------------------------------
# Page
# -----------------------------------------------------------------------------


st.set_page_config(
    page_title="Upload Documents",
    page_icon="📂",
    layout="centered",
    initial_sidebar_state="expanded",
)

_apply_page_style()
_render_sidebar()
_show_flash_messages()

st.markdown(
    """
    <div class="page-hero">
        <h1>Upload Documents</h1>
        <p>Add PDFs to your knowledge base for hybrid search and RAG chat.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    document_names = list_documents()
except Exception:
    st.error(f"Cannot reach the API at **{API_BASE_URL}**.")
    st.code("uvicorn api.main:app --reload --host 0.0.0.0 --port 8000", language="bash")
    st.stop()

# --- Upload section ---
st.markdown('<p class="section-header">Upload Documents</p>', unsafe_allow_html=True)

with st.container(border=True):
    st.markdown(
        """
        <div class="upload-dropzone">
            <p>Drag and drop PDF files here, or click to browse</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    uploaded_files = st.file_uploader(
        "Upload your documents (PDF)",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="visible",
        help="Large PDFs may take several minutes to extract, chunk, embed, and index.",
    )
    st.caption("Supported format: PDF only · Multiple files allowed")

if uploaded_files:
    _handle_uploads(uploaded_files, list(document_names))

st.divider()

# --- Your files section ---
st.markdown('<p class="section-header">Your Files</p>', unsafe_allow_html=True)

file_count = len(document_names)
st.markdown(
    f'<div class="summary-bar">📚 <strong>{file_count}</strong> '
    f'document{"s" if file_count != 1 else ""} in your knowledge base</div>',
    unsafe_allow_html=True,
)

if document_names:
    col_list, col_actions = st.columns([3, 1])
    with col_actions:
        if st.button(
            "Delete all",
            type="secondary",
            use_container_width=True,
            help="Remove every document from the vector store and uploads folder.",
        ):
            st.session_state["confirm_delete_all"] = True

    if st.session_state.get("confirm_delete_all"):
        st.warning("This will permanently remove all indexed documents. Continue?")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Yes, delete all", type="primary", use_container_width=True):
                st.session_state["confirm_delete_all"] = False
                _delete_all(list(document_names))
        with c2:
            if st.button("Cancel", use_container_width=True):
                st.session_state["confirm_delete_all"] = False
                st.rerun()

    for idx, name in enumerate(document_names):
        with st.container(border=True):
            _render_file_row(name, idx)
else:
    st.markdown(
        """
        <div class="empty-state">
            <p style="font-size: 2rem; margin-bottom: 0.5rem;">📭</p>
            <p><strong>No documents yet</strong></p>
            <p>Upload a PDF above to get started.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
