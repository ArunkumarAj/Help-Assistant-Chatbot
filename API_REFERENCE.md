# API Reference (Developer Deep Dive)

Every API endpoint, what happens when you hit it, which functions run, and why they exist in the RAG pipeline.

## Overview

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/health` | Liveness check |
| GET | `/documents` | List all indexed documents |
| POST | `/documents/upload` | Upload a PDF, chunk, embed, index |
| DELETE | `/documents/{document_name}` | Remove document from store and disk |
| POST | `/chat` | Ask a question; get a RAG-powered answer |

Base URL: `http://localhost:8000`

---

## 1. GET /health

**File:** `api/routes/health.py`

**Request:** `GET /health` ? no params, no auth.

**Response:** `{ "status": "ok", "service": "rag-api" }`

**Internal flow:** Returns a static dict. Nothing else.

**Why it exists:** The Streamlit frontend and load balancers call this to verify the backend is up.

---

## 2. GET /documents

**File:** `api/routes/documents.py`

**Response:** `{ "documents": ["policy_v2.pdf", "faq.pdf"] }`

### Internal call chain

```
list_documents()                              [api/routes/documents.py]
  +-- create_index()                          [services/ingestion.py]
  |     +-- store_create_index()              [vector_store/store.py]
  |           +-- _get_knowledge_base_collection()
  |           +-- _load_bm25_index()
  +-- list_document_names()                   [vector_store/store.py]
        +-- collection.get(metadatas)
              -> unique document_name values -> sorted list
```

| Function | Why needed |
|----------|-----------|
| `create_index()` | Guarantees the store is ready; idempotent. |
| `list_document_names()` | ChromaDB stores chunks not files; scans metadata for unique names. |
| `_get_knowledge_base_collection()` | Central accessor for the Chroma collection. |

---

## 3. POST /documents/upload

**File:** `api/routes/documents.py`

**Request:** `POST /documents/upload`, multipart/form-data, field `file=<PDF>`

**Response:** `{ "filename": "policy.pdf", "chunks_indexed": 42, "errors": [] }`

**Errors:** 400 (not PDF / no text), 409 (already exists)

### Internal call chain

```
upload_document(file)                                   [api/routes/documents.py]
  +-- validate PDF extension
  +-- pdf_bytes = await file.read()
  +-- _extract_text_per_page(pdf_bytes)                 [PdfReader per page]
  +-- validate text not empty
  +-- save PDF to uploaded_files/
  +-- create_index()                                    [services/ingestion.py]
  +-- list_document_names() -> duplicate check
  +-- process_and_index_document_with_pages()           [services/ingestion.py]
        +-- chunk_text(text) per page                   [core/text_utils.py]
        |     +-- clean_text()  fix hyphens, whitespace
        |     +-- split into overlapping word chunks
        +-- generate_embeddings(chunks)                 [embedding/model.py]
        |     +-- get_embedding_model()  SentenceTransformer singleton
        |     +-- model.encode(chunk) -> 768-dim vector
        +-- _prepare_chunks_for_store()                 [services/ingestion.py]
        |     +-- add "passage: " prefix if asymmetric
        +-- store_add_documents()                       [vector_store/store.py]
              +-- collection.add(ids, embeddings, docs, metadatas)
              +-- rebuild BM25 index with new chunks
              +-- _save_bm25_index() -> bm25_index.pkl
```

### Why each step matters

| Step | Why |
|------|-----|
| PDF text extraction | RAG searches text; PDFs are binary. |
| Page-level extraction | Citations can say "page 5". |
| Text cleaning | Raw PDF has broken hyphens and whitespace. |
| Chunking with overlap | LLMs have token limits; overlap prevents info gaps. |
| Embedding | Dense search needs each chunk as a 768-dim vector. |
| ChromaDB insert | Fast similarity search at query time. |
| BM25 update | Keyword search for exact terms like "DDO". |
