# 📝 Local RAG System with LLMs

A **Retrieval-Augmented Generation (RAG)** app for querying your documents locally: upload PDFs, search by meaning with FAISS, and chat using your own LLM via a REST API. No OpenSearch or Ollama required.

### 🌟 Key Features
- **Privacy-friendly:** Documents and vectors stay on your machine (FAISS + local embeddings).
- **Vector search:** FAISS for fast similarity search over document chunks.
- **Your own LLM:** Chat uses a custom LLM (LangChain) that calls your OpenAI-compatible REST API; configure `API_URL` and `API_KEY` in `.env`.
- **Streamlit UI:** Welcome page, Chatbot (with RAG on/off and temperature), and Upload Documents.

---

## Prerequisites

- **Python:** 3.10 or 3.11 recommended (3.9+ may work).
- **LLM API:** An OpenAI-compatible chat completions endpoint (URL + API key) for the chatbot.
- **Optional – OCR:** For PDFs that are scanned images, [Tesseract](https://github.com/tesseract-ocr/tesseract) installed and on `PATH` (used by `pytesseract` in `src/ocr.py` for image-based PDF pages).

---

## Python setup and installation

1. **Install Python**  
   - Windows: [python.org](https://www.python.org/downloads/) — during setup, check “Add Python to PATH”.  
   - macOS: `brew install python@3.11` or use python.org.  
   - Linux: `sudo apt install python3.11 python3.11-venv` (or your distro’s package).

2. **Clone the repo and go into the project folder**
   ```bash
   git clone <your-repo-url>
   cd jam-chatbot
   ```

3. **Create and activate a virtual environment**
   ```bash
   python -m venv .venv
   ```
   - Windows (PowerShell): `.\.venv\Scripts\Activate.ps1`  
   - Windows (cmd): `.\.venv\Scripts\activate.bat`  
   - macOS/Linux: `source .venv/bin/activate`

4. **Upgrade pip and install dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## Configuration

- **Embeddings and FAISS:** Edit `src/constants.py` as needed:
  - `EMBEDDING_MODEL_PATH` – Sentence Transformers model (e.g. `microsoft/mpnet-base` or local path).
  - `EMBEDDING_DIMENSION` – Must match the model (768 for mpnet-base).
  - `TEXT_CHUNK_SIZE` – Chunk size (in words) for splitting documents.
  - `FAISS_INDEX_PATH` – Where to store the FAISS index and metadata (default `data/faiss_index`).

- **Chat LLM (your API):** Copy `.env.example` to `.env` in the project root and set:
  - `API_URL` – Your LLM API chat completions URL (OpenAI-compatible).
  - `API_KEY` – API key; sent as `X-API-KEY` by default (see `src/custom_llm.py` to change header).
  - `LLM_MODEL` – (Optional) Model name (default used in code: `gpt-5-mini`).

---

## How to run

1. Ensure the virtual environment is activated and dependencies are installed (see above).
2. Ensure `.env` is configured (and `src/constants.py` if you changed defaults).
3. From the project root, run:
   ```bash
   streamlit run Welcome.py
   ```
4. Open the URL shown in the terminal (e.g. `http://localhost:8501`).
5. Use **Upload Documents** to add PDFs (they are chunked, embedded, and stored in FAISS). Use **Chatbot** to ask questions; enable “RAG mode” to use the uploaded documents as context.

---

## Main libraries and concepts

| Purpose            | Library / concept                                      |
|--------------------|--------------------------------------------------------|
| Web UI             | Streamlit                                              |
| Embeddings         | sentence-transformers                                  |
| Vector store       | FAISS (faiss-cpu)                                      |
| Chat LLM           | Custom REST client (LangChain `LLM`) in `src/custom_llm.py` |
| Config             | `src/constants.py`, `.env` (python-dotenv)             |
| PDF text           | PyPDF2; optional OCR via pytesseract + Pillow           |

When you change code concepts or add/remove libraries, update this README so it stays in sync with the project.

---

Enjoy building your local RAG system.
