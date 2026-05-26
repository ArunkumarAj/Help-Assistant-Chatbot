"""
Dashboard chat logs table: reads JSONL from core.chat_log.
"""
import json
from datetime import datetime
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from core.chat_log import read_chat_logs
from core.config import settings


def _format_timestamp(iso_ts: str) -> str:
    if not iso_ts:
        return "—"
    try:
        dt = datetime.fromisoformat(iso_ts.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    except ValueError:
        return iso_ts


def _entry_to_row(entry: Dict[str, Any]) -> Dict[str, str]:
    json_request = entry.get("json_request")
    if json_request is None:
        json_request = {
            "query": entry.get("query", ""),
            "temperature": entry.get("temperature"),
        }
    json_response = entry.get("json_response")
    if json_response is None:
        json_response = {"response": entry.get("response_preview", "")}

    return {
        "time": _format_timestamp(entry.get("timestamp_utc", "")),
        "source": str(entry.get("source", "")),
        "chunks": str(entry.get("num_chunks", "")),
        "cached": "yes" if entry.get("from_cache") else "no",
        "json_request": json.dumps(json_request, ensure_ascii=False, indent=2),
        "json_response": json.dumps(json_response, ensure_ascii=False, indent=2),
    }


@st.cache_data(ttl=10, show_spinner=False)
def _load_log_rows() -> List[Dict[str, str]]:
    return [_entry_to_row(e) for e in read_chat_logs(reverse=True)]


def render_chat_logs_page() -> None:
    """Full Streamlit page: filterable table + row detail for chat JSONL logs."""
    st.title("Chat logs 📋")
    st.markdown(
        "Every chat turn from the API is logged here with **time**, **json_request**, and **json_response**."
    )
    log_path = getattr(settings, "chat_log_path", "logs/chat_logs.jsonl")
    st.caption(f"Log file: `{log_path}`")
    st.divider()
    _render_chat_logs_table()


def render_chat_logs_tab() -> None:
    """Same table UI embedded in the home dashboard tab."""
    _render_chat_logs_table()


def _render_chat_logs_table() -> None:
    toolbar = st.columns([1, 1, 2])
    with toolbar[0]:
        if st.button("Refresh logs", use_container_width=True):
            _load_log_rows.clear()
            st.rerun()
    with toolbar[1]:
        show_count = st.selectbox(
            "Show",
            options=[25, 50, 100, 500, 0],
            index=1,
            format_func=lambda n: "All" if n == 0 else f"Last {n}",
            label_visibility="collapsed",
        )

    rows = _load_log_rows()
    if not rows:
        st.info("No chat logs yet. Send a message in the Chatbot to create entries.")
        return

    df = pd.DataFrame(rows)

    with toolbar[2]:
        source_filter = st.multiselect(
            "Filter by source",
            options=sorted(df["source"].unique().tolist()),
            default=[],
            placeholder="All sources",
        )
    query_filter = st.text_input("Search in request/response JSON", placeholder="Filter text…")

    if source_filter:
        df = df[df["source"].isin(source_filter)]
    if query_filter.strip():
        q = query_filter.strip().lower()
        mask = (
            df["json_request"].str.lower().str.contains(q, na=False)
            | df["json_response"].str.lower().str.contains(q, na=False)
        )
        df = df[mask]

    if show_count > 0:
        df = df.head(show_count)

    st.metric("Matching log entries", len(df))

    st.dataframe(
        df,
        use_container_width=True,
        height=min(520, 80 + len(df) * 38),
        column_config={
            "time": st.column_config.TextColumn("Time", width="medium"),
            "source": st.column_config.TextColumn("Source", width="small"),
            "chunks": st.column_config.TextColumn("Chunks", width="small"),
            "cached": st.column_config.TextColumn("Cached", width="small"),
            "json_request": st.column_config.TextColumn(
                "json_request",
                width="large",
                help="POST /chat-shaped request body",
            ),
            "json_response": st.column_config.TextColumn(
                "json_response",
                width="large",
                help="API response body",
            ),
        },
        hide_index=True,
    )

    st.markdown("##### Row detail")
    if df.empty:
        st.caption("No rows match your filters.")
        return

    labels = [
        f"{row['time']} · {row['source']} · {(json.loads(row['json_request']).get('query') or '')[:60]}"
        for _, row in df.iterrows()
    ]
    pick = st.selectbox("Select a log entry", range(len(labels)), format_func=lambda i: labels[i])
    row = df.iloc[pick]
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**json_request**")
        st.code(row["json_request"], language="json")
    with c2:
        st.markdown("**json_response**")
        st.code(row["json_response"], language="json")

    export_payload = []
    for _, row in df.iterrows():
        export_payload.append({
            "time": row["time"],
            "source": row["source"],
            "json_request": json.loads(row["json_request"]),
            "json_response": json.loads(row["json_response"]),
        })
    st.download_button(
        "Download filtered logs (JSON)",
        data=json.dumps(export_payload, ensure_ascii=False, indent=2),
        file_name="chat_logs_export.json",
        mime="application/json",
        use_container_width=True,
    )
