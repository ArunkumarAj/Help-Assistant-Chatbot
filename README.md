# RAG Document Assistant (FastAPI + Streamlit)

RAG app with a **FastAPI (async) backend** and a **Streamlit frontend**. Documents are uploaded and indexed via the API; the chatbot calls the API for RAG responses.

## Project layout

```
jam-chatbot/
├── api/                 # FastAPI app & routes (async)
│   ├── main.py           # App entry, CORS, lifespan
│   └── routes/
│       ├── health.py     # GET /health
│       ├── documents.py  # GET /documents, POST /documents/upload, DELETE /documents/{name}
│       └── chat.py       # POST /chat
├── core/                 # Shared config, logging, text utils
│   ├── config.py
│   ├── logging_config.py
│   └── text_utils.py
├── embedding/            # SentenceTransformer model & embeddings
│   └── model.py
├── llm/                  # Custom LLM client (OpenAI-compatible REST)
│   └── client.py
├── vector_store/         # FAISS index & metadata
│   └── store.py
├── services/             # Business logic (ingestion, RAG)
│   ├── ingestion.py
│   └── rag.py
├── streamlit_app/        # Streamlit UI (calls API)
│   ├── Welcome.py        # Entry: streamlit run streamlit_app/Welcome.py
│   ├── config.py         # API_BASE_URL
│   ├── api_client.py     # HTTP client for API
│   └── pages/
│       ├── 1_🤖_Chatbot.py
│       └── 2_📄_Upload_Documents.py
├── logs/                 # Application logs (created at runtime)
├── data/                 # FAISS index files (created at runtime)
├── uploaded_files/       # Uploaded PDFs (created at runtime)
├── .env.example
├── requirements.txt
└── README.md
```

## Prerequisites

- Python 3.10+
- LLM API: OpenAI-compatible endpoint (API_URL + API_KEY in `.env`)

## Setup

1. Clone and create a venv:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate   # Windows
   # source .venv/bin/activate  # macOS/Linux
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and set:
   - `API_URL` – your LLM chat completions URL
   - `API_KEY` – API key (sent as `X-API-KEY`)
   - `API_BASE_URL` – default `http://localhost:8000` (used by Streamlit to call the API)

## Run

**1. Start the FastAPI backend (from project root):**

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

**2. Start the Streamlit UI (from project root):**

```bash
streamlit run streamlit_app/Welcome.py
```

Open the URL shown (e.g. http://localhost:8501). Use **Upload Documents** to add PDFs and **Chatbot** to chat (enable RAG to use documents as context).

## API summary

| Method | Path | Description |
|--------|------|-------------|
| GET | /health | Health check |
| GET | /documents | List document names |
| POST | /documents/upload | Upload PDF (file) → extract text, chunk, embed, index |
| DELETE | /documents/{name} | Delete document and its chunks |
| POST | /chat | RAG chat (body: query, use_rag, num_results, temperature, chat_history) |

Blocking work (embedding, FAISS, LLM) runs in thread pool via `run_in_executor` so the API stays async.

## Configuration

- **core/config.py** (and env): `EMBEDDING_MODEL_PATH`, `EMBEDDING_DIMENSION`, `TEXT_CHUNK_SIZE`, `FAISS_INDEX_PATH`, `LOG_FILE_PATH`, etc.
- **.env**: `API_URL`, `API_KEY`, `LLM_MODEL`, `API_BASE_URL` (for Streamlit).

When you change code or libraries, update this README to keep it in sync.
