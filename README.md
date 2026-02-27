# RAG Document Assistant

> FastAPI (async) backend + Streamlit frontend for a RAG-based **Help Support Assistant**. Upload PDFs, index with **ChromaDB** (hybrid search), and chat with answers strictly from the knowledge base.

---

## Table of contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Project structure](#project-structure)
- [Architecture](#architecture)
- [API reference](#api-reference)
- [📊 RAG Evaluation](#-rag-evaluation)
- [Redis cache (optional)](#redis-cache-optional)
- [Notebooks](#notebooks)
- [Maintaining this README](#maintaining-this-readme)

---

## Features

- **Document ingestion:** Upload PDFs; text is extracted, chunked, embedded (SentenceTransformer), and stored in **ChromaDB** (local, persistent).
- **RAG chat:** **Hybrid search** (dense vectors + BM25 keyword) over ChromaDB; top-k context is passed to an LLM with a strict **Help Support Assistant** prompt (answer only from knowledge base, redirect otherwise).
- **Async API:** FastAPI with blocking work (embedding, vector store, LLM) offloaded to a thread pool.
- **Streamlit UI:** Welcome, Chatbot, and Upload Documents pages that call the API.

---

## Prerequisites

- **Python** 3.10+
- **LLM API:** OpenAI-compatible chat completions endpoint (URL + API key). Set in `.env` as `API_URL` and `API_KEY`.

---

## Installation

1. **Clone the repository** and go to the project root.

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate        # Windows
   # source .venv/bin/activate   # macOS / Linux
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Copy `.env.example` to `.env`** and set at least:
   - `API_URL` – your LLM chat completions URL
   - `API_KEY` – API key (sent as `X-API-KEY`)
   - `API_BASE_URL` – backend URL for Streamlit (default: `http://localhost:8000`)

---

## Configuration

| Source | Purpose |
|--------|--------|
| **.env** | `API_URL`, `API_KEY`, `LLM_MODEL`, `API_BASE_URL` |
| **core/config.py** (or env) | `EMBEDDING_MODEL_PATH`, `EMBEDDING_DIMENSION`, `TEXT_CHUNK_SIZE`, `TEXT_CHUNK_OVERLAP`, `VECTOR_STORE_PATH`, `LOG_FILE_PATH` |

---

## Usage

**1. Start the FastAPI backend** (from project root):

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

**2. Start the Streamlit UI** (from project root):

```bash
streamlit run streamlit_app/Welcome.py
```

Open the URL shown (e.g. `http://localhost:8501`). Use **Upload Documents** to add PDFs and **Chatbot** to ask questions (enable RAG to use the indexed documents as context).

---

## Project structure

```
jam-chatbot/
├── api/                    # FastAPI app & routes
│   ├── main.py
│   └── routes/
│       ├── health.py       # GET /health
│       ├── documents.py    # GET/POST/DELETE /documents
│       └── chat.py         # POST /chat
├── core/                   # Config, logging, text utils
├── embedding/              # SentenceTransformer model & embeddings
├── llm/                    # Custom LLM client (OpenAI-compatible)
├── vector_store/          # ChromaDB + BM25 (hybrid search)
├── services/               # Ingestion & RAG logic
├── streamlit_app/         # Streamlit UI
│   ├── Welcome.py
│   ├── config.py
│   ├── api_client.py
│   └── pages/
├── logs/                   # App logs (created at runtime)
├── data/                   # ChromaDB + BM25 index (created at runtime)
├── uploaded_files/         # Uploaded PDFs (created at runtime)
├── notebooks/              # API test notebooks (optional, in-process)
│   ├── 01_documents_api.ipynb
│   ├── 02_chat_api.ipynb
│   └── rag_eval_report.ipynb
├── .env.example
├── requirements.txt
└── README.md
```

---

## Architecture

### RAG pipeline

This project uses **ChromaDB** as the vector store with **hybrid search** (dense + sparse):

- **Indexing:** PDF → extract text → clean & chunk (word-based, overlap) → embed (SentenceTransformer) → store in **ChromaDB** (persistent, local) and update **BM25** index for keyword search.
- **Query:** User question → embed query + use query text → **hybrid search** (dense similarity + BM25 keyword, merged with RRF) → top-k chunks as context → single prompt (system + context + history + query) → one LLM call → response.

**Why ChromaDB instead of FAISS:** FAISS only supports dense (vector) similarity search and has no built-in hybrid or keyword search. ChromaDB gives us a persistent store, metadata filtering, and—combined with a BM25 index—**hybrid search** that improves retrieval for both semantic and exact-term queries (e.g. product IDs, policy names). We therefore replaced FAISS with ChromaDB and added BM25 + RRF for better RAG quality.

### High-level diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              USER (Browser)                                       │
└─────────────────────────────────────────────────────────────────────────────────┘
                    │                                    │
                    ▼                                    ▼
┌───────────────────────────────┐        ┌──────────────────────────────────────────┐
│   STREAMLIT UI                │        │   FASTAPI BACKEND (async)                │
│   streamlit_app/               │  HTTP  │   api/                                    │
│   • Welcome, Chatbot, Upload  │◄──────►│   /health, /documents, /chat              │
└───────────────────────────────┘        └──────────────────────────────────────────┘
                                                          │
         ┌────────────────────────────────────────────────┼────────────────────────────────┐
         │                        │                        │                                │
         ▼                        ▼                        ▼                                ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────────┐
│  core/           │    │  embedding/     │    │  vector_store/   │    │  llm/                   │
│  config, logging │    │  SentenceTrans-  │    │  ChromaDB + BM25 │    │  CustomLLM (your API)   │
│  text_utils      │    │  former         │    │  metadata        │    │  OpenAI-compatible      │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────────────┘
         │                        │                        │                                │
         └────────────────────────┴────────────────────────┴────────────────────────────────┘
                                                          │
                                          ┌───────────────┴───────────────┐
                                          │  services/                     │
                                          │  ingestion | rag               │
                                          └───────────────────────────────┘

  INGEST:  PDF → extract → chunk → embed → ChromaDB + BM25 add
  QUERY:   query → embed + text → hybrid search (Chroma + BM25, RRF) → prompt → LLM → response
```

**Mermaid** (for viewers that support it):

```mermaid
flowchart TB
    subgraph Client[" "]
        User[User Browser]
        Streamlit[Streamlit UI]
    end
    subgraph API["FastAPI"]
        Health[GET /health]
        Docs[GET/POST/DELETE /documents]
        Chat[POST /chat]
    end
    subgraph Backend[" "]
        Core[core]
        Embed[embedding]
        Chroma[vector_store]
        LLM[llm]
        Svc[services]
    end
    User --> Streamlit
    Streamlit --> Health
    Streamlit --> Docs
    Streamlit --> Chat
    Docs --> Svc
    Chat --> Svc
    Svc --> Core
    Svc --> Embed
    Svc --> Chroma
    Svc --> LLM
```

---

## API reference

Base URL: `http://localhost:8000` (or your `API_BASE_URL`).

### 1. Health check

| Method | Path    | Body | Description |
|--------|---------|------|-------------|
| GET    | /health | -    | Check if API is up |

**Response (200):**
```json
{
  "status": "ok",
  "service": "rag-api"
}
```

---

### 2. List documents

| Method | Path       | Body | Description |
|--------|------------|------|-------------|
| GET    | /documents | -    | List unique document names in the vector store |

**Process:** Ensure ChromaDB collection exists → return unique `document_name` values (sorted).

**Response (200):**
```json
{
  "documents": [
    "Bank-Policy-Development-pol-fin(world-Bank).pdf",
    "FAQ.pdf"
  ]
}
```

---

### 3. Upload document

| Method | Path               | Body               | Description |
|--------|--------------------|--------------------|-------------|
| POST   | /documents/upload  | multipart/form-data | Upload PDF: extract text, chunk, embed, index |

**Process:** Validate PDF → read bytes → extract text (PyPDF2) → save file under `uploaded_files/` → ensure collection exists and name not duplicate → chunk text → embed (SentenceTransformer, thread pool) → add to ChromaDB + BM25 index → return counts.

**Request:** Form field `file` = PDF file.

**Example:**
```bash
curl -X POST "http://localhost:8000/documents/upload" \
  -F "file=@/path/to/document.pdf"
```

**Response (200):**
```json
{
  "filename": "document.pdf",
  "chunks_indexed": 42,
  "errors": []
}
```

**Errors:** 400 (not PDF / no text), 409 (document already exists).

---

### 4. Delete document

| Method | Path                     | Body | Description |
|--------|--------------------------|------|-------------|
| DELETE | /documents/{document_name} | -    | Remove all chunks for this document and delete file from uploads |

**Process:** Load index and metadata → filter out entries with given `document_name` → rebuild index and save → delete file from `uploaded_files/` if present.

**Response (200):**
```json
{
  "deleted": 42
}
```

---

### 5. Chat (RAG)

| Method | Path  | Body | Description |
|--------|-------|------|-------------|
| POST   | /chat | JSON | One support-style reply; optional RAG retrieval |

**Process:** Validate `query` → if `use_rag`: embed query → hybrid search (ChromaDB + BM25, RRF) → top-k chunks as context → build prompt (Help Support Assistant system + context + history + query) → call LLM → return response.

**Request body:**

| Field         | Type    | Required | Default | Description        |
|---------------|---------|----------|---------|--------------------|
| query         | string  | Yes      | -       | User question      |
| use_rag       | boolean | No       | true    | Use RAG context    |
| num_results   | int     | No       | 5       | Chunks to retrieve |
| temperature   | number  | No       | 0.7     | LLM temperature    |
| chat_history  | array   | No       | []      | `[{ "role", "content" }]` |

**Request sample:**
```json
{
  "query": "What is the refund policy?",
  "use_rag": true,
  "num_results": 5,
  "temperature": 0.7,
  "chat_history": [
    { "role": "user", "content": "Hi" },
    { "role": "assistant", "content": "Hello! How can I help you today?" }
  ]
}
```

**Response (200):**
```json
{
  "response": "According to the knowledge base, refunds are processed within 5–7 business days after approval."
}
```

**Errors:** 400 if `query` is empty.

---

## 📊 RAG Evaluation

This section documents how to **evaluate** the "Help support assistant" RAG pipeline across **retrieval**, **generation** (faithfulness/hallucinations), **end‑to‑end quality**, and **system** metrics.

### Goals
- Ensure the assistant's answers are **grounded in retrieved context** (high faithfulness, low hallucinations).
- Verify **retrieval quality** (recall@k, MRR, nDCG), **attribution**, and **answer relevance**.
- Track **latency** and **cost**, with reproducible runs and reports.

### 🧩 Components
```
eval/
├── evaluator.py
├── metrics.py
├── judge_prompts/
│   ├── faithfulness.json
│   └── relevance.json
├── datasets/
│   └── eval.jsonl
└── reports/
```
Generation prompt: `prompts/grounded_answer.txt` (enforces grounded answers + inline citations [1], [2]).

### 📦 Installation
```bash
pip install -r requirements.txt
pip install -r requirements-eval.txt
# Or: pip install transformers sentencepiece nltk
python -c "import nltk; nltk.download('punkt')"
# Optional: ragas, trulens-eval
```

### 🏃 One-command run
```bash
# From project root
python -m eval.evaluator --data eval/datasets/eval.jsonl --k 5 --judge nli --out eval/reports/run_YYYYMMDD_HHMM/
python -m eval.evaluator --data eval/datasets/eval.jsonl --k 5 --judge llm --out eval/reports/run_YYYYMMDD_HHMM/
```
Reports are written to the `--out` directory as `report.json`, `report.csv`, `report.md`, and `report.html`.

### Metrics
- **Retrieval:** Recall@k, MRR@k, nDCG@k, Coverage, Redundancy.
- **Generation:** Faithfulness (NLI and/or LLM-as-judge), Hallucination rate, Answer relevance, Attribution (precision & recall), Context utilization, Conciseness.
- **End-to-end:** Exact Match, F1, Nugget F1 (when gold answers/nuggets exist).
- **System:** Latency (p50/p95) per stage, tokens in/out.

### Eval dataset
Populate `eval/datasets/eval.jsonl` with one JSON object per line (one line per test case). Each line is a single JSON object with the keys below.

| Key | Required | Meaning |
|-----|----------|---------|
| **query** | Yes | The user question to run through RAG (retrieval + generation). |
| **ground_truth** | No | Reference answer; used for **Exact Match** and **token F1**. |
| **gold_passages** | No | List of passages that *should* be retrieved for this query; used for **Recall@k**, **MRR**, **nDCG**, **Coverage**. |
| **nuggets** | No | List of key facts the answer should contain; used for **Nugget F1**. |

**Understanding the keys**

- **query** — The exact question you send to the RAG pipeline. The evaluator runs retrieval (embed + ChromaDB hybrid search) and then generation (LLM) for this string. Every test case must have a `query`.
- **ground_truth** — The ideal or reference answer for this question. Used to score the model’s answer with **Exact Match** (full string match after normalizing) and **token F1** (word-overlap). Omit if you don’t have a single “correct” answer; those metrics will be skipped or zero for that row.
- **gold_passages** — Passages from your knowledge base that *should* appear in the top‑k retrieval for this query. The evaluator compares what was actually retrieved to this list to compute **Recall@k** (did any gold passage appear in top‑k?), **MRR** (rank of first relevant passage), **nDCG@k**, and **Coverage** (fraction of gold passages present in retrieved set). Use text that matches or closely overlaps your indexed chunks. Omit or leave empty if you’re not measuring retrieval quality.
- **nuggets** — Short, atomic facts that the answer ought to include (e.g. “10 MB”, “Member Country Guarantee”). Used for **Nugget F1**: the evaluator checks how many of these appear in the model’s answer. Good for checking that key information is not missing, without requiring a full sentence match like `ground_truth`.

### Optional: latency logging
Set in `.env`: `EVAL_LOGGING_ENABLED=true`. Then retrieve and generate latencies are logged (e.g. `eval_latency_retrieve_seconds=...`, `eval_latency_generate_seconds=...`).

---

## Redis cache (optional)

Optional Redis-backed caching reduces **LLM cost**, **response time**, and **repeated embedding/retrieval work**. Caching is **off** unless `REDIS_URL` and `CACHE_ENABLED=true` in `.env`.

### Benefits

| Benefit | How Redis cache helps |
|--------|------------------------|
| **Reduce LLM cost** | Cached responses (when `temperature=0`) avoid repeated API calls and token usage for the same question + context. |
| **Improve response speed** | Cache hits return in milliseconds instead of running embedding, vector search, and LLM. |
| **Avoid recomputing embeddings** | Query embeddings are cached by text; repeated or similar queries reuse the same vector. |
| **Maintain consistent responses** | With `temperature=0`, the same prompt always returns the same cached answer. |
| **Handle repeated searches in a conversation** | Repeated user questions hit retrieval and/or response cache without re-embedding or re-calling the LLM. |
| **Improve throughput** | Less CPU (embeddings) and fewer outbound LLM requests, so the system handles more concurrent users. |
| **Support session memory without high cost** | Conversation turns that repeat earlier context benefit from cached retrieval and responses without storing full history in the LLM call. |

### What is cached

- **Embeddings:** `rag:embed:{hash(query)}` → query embedding vector (TTL: `CACHE_TTL_EMBEDDING`, default 24h).
- **Retrieval:** `rag:retrieve:{hash(query)}:{top_k}` → list of retrieved chunks (TTL: `CACHE_TTL_RETRIEVAL`, default 1h).
- **LLM response:** `rag:response:{hash(prompt)}` → model output (TTL: `CACHE_TTL_RESPONSE`, default 1h). Response cache is **only used when `temperature=0`**.

### Redis setup

**1. Install:** `redis` is in `requirements.txt` (`pip install redis`).

**2. Enable in `.env`:**
```env
REDIS_URL=redis://localhost:6379/0
CACHE_ENABLED=true
```
Optional TTLs (seconds): `CACHE_TTL_EMBEDDING=86400`, `CACHE_TTL_RETRIEVAL=3600`, `CACHE_TTL_RESPONSE=3600`.

If `REDIS_URL` is missing or `CACHE_ENABLED` is not `true`, the app runs without cache (no Redis required).

#### Local

- **Docker (recommended):** `docker run -d --name redis-rag -p 6379:6379 redis:7-alpine`
- **Windows (WSL):** `sudo apt install redis-server` then `redis-server`; or [Memurai](https://www.memurai.com/).
- **macOS:** `brew install redis` then `brew services start redis`
- **Linux:** `sudo apt install redis-server` then `sudo systemctl start redis-server`

#### Cloud

- **Redis Cloud:** Create a database at [Redis Cloud](https://redis.com/try-free/); use the public endpoint and password: `REDIS_URL=redis://default:PASSWORD@host:port` (or `rediss://` and port 6380 for TLS).
- **AWS ElastiCache:** Create a Redis cluster; use cluster endpoint: `REDIS_URL=redis://your-cluster.xxxxx.cache.amazonaws.com:6379/0` (add `:AUTH_TOKEN` after `redis://` if using AUTH).
- **Azure Cache for Redis:** Create an instance; under Access keys use the connection string: `REDIS_URL=redis://:KEY@your-cache.redis.cache.windows.net:6380?ssl=True`.

### Verifying cache

With cache enabled, the app log shows `Redis cache connected: ...` on first use. If Redis is down or misconfigured, you’ll see `Redis cache disabled: ...` and the app continues without cache.

### LLM caching architecture

When Redis cache is enabled, the RAG pipeline uses a three-layer cache:

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              CHAT REQUEST (query, history)                               │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                              │
                                              ▼
                    ┌─────────────────────────────────────────┐
                    │  Build RAG context (if use_rag)          │
                    │  prefix = "passage: {query}" or query   │
                    └─────────────────────────────────────────┘
                                              │
         ┌────────────────────────────────────┼────────────────────────────────────┐
         ▼                                    ▼                                    ▼
┌─────────────────┐                ┌─────────────────┐                ┌─────────────────┐
│  EMBEDDING      │                │  RETRIEVAL       │                │  CONTEXT         │
│  Cache key:     │                │  Cache key:      │                │  Build prompt:   │
│  rag:embed:     │                │  rag:retrieve:   │                │  system +       │
│  {query_hash}   │                │  {query}:{top_k} │                │  [1]...[k] +    │
│  Hit → vector   │                │  Hit → chunks    │                │  history + query│
│  Miss → encode  │                │  Miss → Chroma  │                └────────┬────────┘
│  + cache set    │                │  search + set   │                         │
└─────────────────┘                └─────────────────┘                         │
         │                                    │                                  │
         └────────────────────────────────────┴──────────────────────────────────┘
                                              │
                                              ▼
                    ┌─────────────────────────────────────────┐
                    │  RESPONSE CACHE (only if temperature=0)  │
                    │  Key: rag:response:{hash(prompt)}        │
                    │  Hit → return cached answer              │
                    └─────────────────────────────────────────┘
                                              │
                                    Miss (or temp>0)
                                              │
                                              ▼
                    ┌─────────────────────────────────────────┐
                    │  LLM (OpenAI-compatible API)             │
                    │  → response → cache set (if temp=0)      │
                    └─────────────────────────────────────────┘
                                              │
                                              ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              CHAT RESPONSE (answer)                                      │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

**Mermaid:**

```mermaid
flowchart LR
    subgraph Request
        Q[query + history]
    end
    subgraph Cache["Redis cache"]
        E[embed cache]
        R[retrieve cache]
        Resp[response cache]
    end
    subgraph Compute["Compute path"]
        Emb[embed query]
        Chroma[Chroma hybrid]
        LLM[LLM API]
    end
    Q --> E
    E -->|hit| R
    E -->|miss| Emb
    Emb --> E
    Emb --> Chroma
    Chroma --> R
    R -->|hit| Resp
    R -->|miss| Build[build prompt]
    Chroma --> Build
    Build --> Resp
    Resp -->|hit| Out[response]
    Resp -->|miss| LLM
    LLM --> Resp
    LLM --> Out
```

---

## Notebooks

The `notebooks/` folder contains **in-process** test notebooks (no FastAPI server required).

### Prerequisites

- Dependencies: `pip install -r requirements.txt` (from project root).
- **02_chat_api:** `.env` must have `API_URL` and `API_KEY`.
- **rag_eval_report:** `pip install -r requirements-eval.txt` and ensure `eval/datasets/eval.jsonl` exists.

### Notebooks

| File | What it tests |
|------|----------------|
| **01_documents_api.ipynb** | `create_index()`, `list_document_names()`, `process_and_index_document(text, name)`, `delete_documents_by_document_name(name)` |
| **02_chat_api.ipynb** | `chat_response(query, use_rag, num_results, temperature, chat_history)` from `services.rag` |
| **rag_eval_report.ipynb** | RAG evaluation: `run_eval_sync()` from `eval.evaluator`; writes reports to `eval/reports/run_*` |

### How to run

1. Open the notebook (e.g. from project root: `jupyter notebook notebooks/01_documents_api.ipynb`, or in VS Code/Cursor).
2. Run the **Setup** cell first (project root on `sys.path`, `.env` loaded).
3. Run the rest. In **01_documents_api**, set `PDF_PATH` to a real PDF before upload/full flow.

No need to start the FastAPI server; everything runs in-process (including **rag_eval_report.ipynb**).

---

## Maintaining this README

When you change code, add/remove dependencies, or alter the API, update this README so it stays accurate for future reference.
