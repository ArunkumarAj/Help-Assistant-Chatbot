# Agentic RAG (LangGraph)

The chat endpoint uses a **LangGraph** `create_react_agent` graph with **tools**—not a single fixed RAG call.

## Tools

| Tool | Role |
|------|------|
| `search_knowledge_base` | Hybrid Chroma + BM25 retrieval (same knowledge base as before) |
| `get_open_active_cases` | Read **Active** cases from local **SQLite** (`get_open_active_cases` in the LLM) |
| `create_active_support_case` | Insert a new **Active** case into SQLite |

The model decides when to call which tool (ReAct pattern).

## Prerequisites

- **Python 3.11+** (see `pyproject.toml`)
- **UV** (recommended) or `pip` for dependencies
- **OpenAI-compatible HTTP API** with **tool / function calling** (same `API_URL` as before, must support `POST .../v1/chat/completions` with `tools`)
- **`.env`**: `API_URL` (base ending in `/v1`), `API_KEY` (or `OPENAI_API_KEY` if your gateway uses it), `LLM_MODEL`
- **SQLite** for cases: `SQLITE_DB_PATH` optional (default `data/cases.db`)

## Install (UV)

From the project root:

```bash
uv sync
```

Or add packages manually:

```bash
uv add langgraph langchain-core langchain-openai
```

## Load mock cases (SQLite)

So `get_open_active_cases` has data to return:

```bash
uv run python -m database.seed_mock_data
```

Re-seed if the table already has rows:

```bash
uv run python -m database.seed_mock_data --force
```

See **`SETUP_DATABASE.md`** for the full mock dataset and schema.

## Run the API

```bash
uv run uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

(Use the same `reload` excludes for Windows as in `api/main.py` if you use `python -m api.main`.)

## `use_rag: false` in chat

With `use_rag: false`, the request uses a **single** `CustomLLM` call (no LangGraph, no tools). Use this if your API does not support tool calling and you need a non-agent path.

## Files

- `services/agent_graph.py` — `run_support_agent`, `ChatOpenAI`, system prompt
- `services/agent_tools.py` — LangChain `@tool` definitions
- `services/rag_helpers.py` — shared hybrid retrieval + citation metadata
- `llm/openai_compat.py` — `API_URL` → OpenAI `base_url`

## Troubleshooting

- **404 on `.../v2/v1/chat/completions`** — Your router may use **`/v2`** (not `/v1`). The app now keeps an existing `/v1` or `/v2` segment after stripping `.../chat/completions`. Set `API_URL` to the full URL you use with `CustomLLM` (e.g. `https://host/.../v2/chat/completions`); the agent will use base `https://host/.../v2` only.
- **"Set API_URL"** — Set a valid OpenAI-style URL; see `llm/openai_compat.get_openai_compatible_base_url()`.
- **400 from model on tools** — Your backend must accept the `tools` field on chat completions; older proxies may not.
- **Empty citations** — Citations are filled when `search_knowledge_base` runs; case-only answers may have an empty `citations` array in the API response.
