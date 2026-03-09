# Code Review: chatbot

Summary of architecture, strengths, and recommended fixes. Applied fixes are noted at the end.

---

## 1. Architecture Overview

- **API:** FastAPI (`api/main.py`) with routes for health, documents (list/upload/delete), and chat. CORS allows all origins; lifespan creates log/data/upload dirs.
- **RAG:** `services/rag.py` orchestrates optional hybrid search (ChromaDB + BM25, RRF), prompt build, LLM call, and optional Redis caching. Blocking work runs in `run_in_executor`.
- **Vector store:** `vector_store/store.py` — ChromaDB (dense) + BM25 (pickle), hybrid search with RRF. Add/delete keep both in sync.
- **Ingestion:** `services/ingestion.py` — chunk (per-page or full text), embed (with optional `passage: ` prefix), index via store.
- **Config:** `core/config.py` — env-based settings namespace; no secrets in code.
- **Eval:** `eval/metrics.py` and `eval/evaluator.py` — retrieval/generation metrics and runner.

Layering is clear: routes → services → embedding/llm/vector_store; config and cross-cutting (cache, chat_log, logging) in `core/`.

---

## 2. Strengths

- **Single responsibility:** Routes handle HTTP; RAG service handles retrieval + prompt + LLM; store handles index and search; ingestion handles chunk/embed/index.
- **Config in one place:** All tuning (chunk size, top_k, Redis, paths) via env and `core/config.py`.
- **Caching:** Embedding, retrieval, and response cache (Redis) are optional and no-op when disabled; cache keys use hashing for long strings.
- **Chat logging:** JSONL + optional app log line with source (RAG / RAG_NO_HITS / LLM_ONLY) and cache flag; preview length limits logged content.
- **Hybrid search:** Dense + BM25 with RRF gives good recall for both semantic and keyword queries.
- **Evaluation:** Retrieval (recall@k, MRR, nDCG) and generation (faithfulness, relevance, etc.) are implemented and documented in `RAG_TUNING.md`.

---

## 3. Issues and Fixes

### 3.1 Bug: Dense-only search drops `id` on hits (fixed)

**Where:** `vector_store/store.py` — `vector_search()` when `query_text` is `None`.

**Issue:** The dense-only branch returns `[{"_source": h["_source"]} for h in hits]`, which strips `id`. Downstream `_build_citation_meta_list()` uses `hit.get("id")`, so citations get empty `doc_id`.

**Fix:** Return the same shape as hybrid: full hits with `id` and `_source` (e.g. `return hits` instead of mapping to `_source` only).

---

### 3.2 Security: Do not expose exception to user (fixed)

**Where:** `services/rag.py` — `chat_response()` except block.

**Issue:** `error_message = f"Sorry, an error occurred: {e!s}"` sends internal exception text to the client. Security guidance: log the exception server-side and return a generic message.

**Fix:** Log `logger.exception("LLM invocation failed")` (already present) and set `error_message = "Sorry, an error occurred. Please try again."` (or similar) without `e`.

---

### 3.3 Security: Path traversal on document delete (fixed)

**Where:** `api/routes/documents.py` — `delete_document(document_name: path)`.

**Issue:** `document_name` is used as `settings.upload_dir / document_name`. A value like `../other/file.pdf` or `..\\other\\file.pdf` could escape the upload directory.

**Fix:** Reject `document_name` if it contains path segments: e.g. `/`, `\`, or `..`. Allow only a single filename (e.g. `re.match(r"^[^/\\]+$", document_name)` and no `..`). Return 400 with a clear message if invalid.

---

### 3.4 Minor: ChatMessage role validation

**Where:** `api/routes/chat.py` — `ChatMessage.role`.

**Issue:** Any string is accepted. If the UI or downstream code assumes only `"user"` or `"assistant"`, invalid roles could cause confusion.

**Recommendation:** Optional: add a Pydantic validator or `Literal["user", "assistant"]` for `role`. Low priority if the client is trusted.

---

### 3.5 Config attribute access

**Where:** `core/config.py` — `Settings` uses class attributes (e.g. `settings.vector_store_path`).

**Issue:** `vector_store/store.py` uses `getattr(settings, "vector_store_path", None)`. The actual attribute is `vector_store_path`; in config it’s defined. No bug, but the store could use `settings.vector_store_path` directly for consistency.

**Recommendation:** Use `settings.vector_store_path` in the store; optional cleanup.

---

### 3.6 BM25 delete rebuild

**Where:** `vector_store/store.py` — `delete_documents_by_document_name()`.

**Issue:** After deleting from ChromaDB, BM25 is rebuilt from `collection.get()`. If the collection is large, loading all documents into memory can be heavy.

**Recommendation:** Acceptable for moderate corpus sizes; for very large indices, consider incremental BM25 updates or a separate process. No change required for current scope.

---

## 4. Testing and Standards

- Workspace rules require high coverage and AAA tests. No test files were in scope for this review; ensure `api/routes/chat.py`, `services/rag.py`, and `vector_store/store.py` have unit/integration tests (including error paths and cache).
- Security: avoid logging or returning raw user input or internal errors; validate and sanitize path parameters (e.g. document_name).

---

## 5. Applied Fixes (in codebase)

1. **vector_store/store.py:** Dense-only `vector_search()` now returns full hit dicts (with `id` and `_source`) so citation metadata gets a valid `doc_id`.
2. **services/rag.py:** On LLM failure, the user sees a generic message; the real exception is only logged.
3. **api/routes/documents.py:** `delete_document` validates `document_name` to a single filename (no `/`, `\`, or `..`) and returns 400 if invalid.

---

## 6. File-by-file Summary

| File | Notes |
|------|--------|
| `api/main.py` | Clean lifespan and router wiring; CORS allow_origins=["*"] is permissive (tighten for production if needed). |
| `api/routes/chat.py` | Thin layer; validation on empty query. Consider role validation for ChatMessage. |
| `api/routes/documents.py` | PDF validation and duplicate-name check; path traversal check added on delete. |
| `api/routes/health.py` | (Not read; assume simple liveness.) |
| `core/cache.py` | Redis lazy init, safe key hashing, no sensitive data in logs. |
| `core/chat_log.py` | Truncation for preview; no raw secrets in logs. |
| `core/config.py` | Single source of env; typo `ASSYMETRIC_EMBEDDING` (asymmetric) is historical. |
| `core/text_utils.py` | clean_text + chunk_text with configurable size/overlap; clear. |
| `embedding/model.py` | Lazy singleton; encode in sequence (consider batching for large ingest). |
| `llm/client.py` | Retries on 429; timeout 120s; no response body logged. |
| `services/ingestion.py` | Clear flow; passage prefix aligned with config. |
| `services/rag.py` | Clear flow; cache and executor used correctly; exception message fixed. |
| `vector_store/store.py` | Chroma + BM25 + RRF; dense-only return value fixed. |
| `eval/*` | Metrics and evaluator align with RAG_TUNING.md terminology. |
