# API test notebooks (direct function calls)

These notebooks test the **actual Python functions** used by the Documents and Chat flows—no HTTP or running API server. Useful for fast local testing.

## Prerequisites

- Dependencies installed: `pip install -r requirements.txt` (from project root).
- For **02_chat_api**: `.env` must have `API_URL` and `API_KEY` for your LLM.

## Notebooks

| File | What it tests |
|------|----------------|
| **01_documents_api.ipynb** | `create_index()`, `list_document_names()`, `process_and_index_document(text, name)`, `delete_documents_by_document_name(name)` |
| **02_chat_api.ipynb** | `chat_response(query, use_rag, num_results, temperature, chat_history)` from `services.rag` |

## How to run

1. Open the notebook from the repo (e.g. from project root: `jupyter notebook notebooks/01_documents_api.ipynb`, or open in VS Code/Cursor).
2. Run the **Setup** cell first so the project root is on `sys.path` and `.env` is loaded.
3. Run the rest of the cells. In **01_documents_api**, set `PDF_PATH` (or the path in the full-flow cell) to a real PDF before running upload/full flow.

No need to start the FastAPI server; everything runs in-process.
