"""
OpenAI-compatible base URL for ChatOpenAI / tool calling.

Maps API_URL (e.g. .../v1/chat/completions or .../v2/chat/completions) to the base the
client uses: it appends /chat/completions, so the base must be .../v1 or .../v2 — not
.../v2/v1 (a common bug when the gateway already uses /v2).
"""

import os
import re


def get_openai_compatible_base_url() -> str:
    """
    Return base URL for OpenAI-compatible client (e.g. https://host/v1 or https://host/.../v2).
    If API_URL is not set, returns empty string.
    """
    u = (os.environ.get("API_URL") or "").strip()
    if not u:
        return ""
    u = u.rstrip("/")
    # OpenAI client POSTs to {base}/chat/completions. Strip only that suffix so /v1 or /v2 stays in base.
    if u.endswith("/chat/completions"):
        u = u[: -len("/chat/completions")].rstrip("/")
    # e.g. .../llmrouter-api/v2 — do not add another /v1
    if re.search(r"/v\d+$", u):
        return u
    u = f"{u}/v1"
    return u.rstrip("/") or u
