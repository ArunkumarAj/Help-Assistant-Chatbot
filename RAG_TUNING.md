# RAG Tuning Guide

This guide explains **RAG terminology**, **why your answer can be wrong or indirect**, and **which knobs to turn** to get better outputs—without hardcoding any specific query or response.

---

## 1. RAG in one sentence

**RAG = Retrieve** relevant chunks from the knowledge base → **Augment** the LLM prompt with them → **Generate** an answer from that context only.

If the answer is wrong or vague, the cause is usually one of:

- **Retrieval:** The right chunk was not in the top‑k, or it was outranked by less relevant chunks.
- **Chunking:** The right sentence (e.g. “IMF means the International Monetary Fund”) is buried in a big chunk or split across chunks, so retrieval or the model doesn’t focus on it.
- **Generation:** The model prefers a cautious, indirect phrasing (“the knowledge base only says…”) instead of giving the direct fact (“IMF means the International Monetary Fund”).

Tuning = improving retrieval, chunking, and generation so that **when the definition (or fact) exists in the docs, it shows up in the answer**.

---

## 2. RAG metrics (what people usually mean)

These are the terms you’ll see when people talk about “metrics” and “tuning” in RAG.

### Retrieval metrics (did we fetch the right chunks?)

| Metric | Meaning | Why it matters |
|--------|--------|----------------|
| **Recall@k** | Of all relevant chunks, how many are in the top‑k we actually send to the LLM? | Low recall → the chunk that says “IMF = International Monetary Fund” might be chunk #6 and we only use top 5. |
| **Precision** | Of the top‑k we retrieved, how many are actually relevant? | Low precision → context is noisy, model may focus on the wrong part. |
| **MRR (Mean Reciprocal Rank)** | How high in the list is the **first** relevant chunk? | If the definition chunk is always at rank 1, MRR is 1.0; if it’s at 6 and we use top 5, we never see it. |

So when we say “tune retrieval,” we mean: **get the chunk that contains the answer (e.g. the definition) into the top‑k and, ideally, near the top.**

### Generation metrics (did the model answer correctly from context?)

| Metric | Meaning | Why it matters |
|--------|--------|----------------|
| **Faithfulness** | Is the answer supported by the retrieved context? No invented facts? | We want the model to say only what the chunks say. |
| **Answer relevance** | Does the answer match what the user asked? | “What is IMF?” should get a definition, not only “the knowledge base says IMF provides views…”. |
| **Grounding** | Are claims tied to the right source (e.g. [1], [2])? | Keeps answers traceable. |

So when we say “tune generation,” we mean: **when the context has the definition, the model should give it directly (e.g. “IMF means the International Monetary Fund”) and not only describe what the knowledge base “says” in general.**

---

## 3. Why “what is IMF?” can give a bad answer

Example:

- **Query:** “what is IMF?”
- **Bad answer:** “The knowledge base only says the IMF provides views on a country’s macroeconomic policy framework via a ‘Fund Relations Note’…”
- **Good answer (when the doc has it):** “IMF means the International Monetary Fund.”

Possible causes:

1. **Retrieval**
   - The chunk that says “IMF = International Monetary Fund” is **not in the top‑k** (e.g. top 5). Chunks that mention “IMF” many times in a policy context (Fund Relations Note, IMF-supported program) may rank higher.
   - So the model never sees the definition chunk.

2. **Chunking**
   - The definition might be in a long chunk dominated by other content; the chunk’s embedding is more “policy / programs” than “acronym definition,” so it ranks lower for “what is IMF?”
   - Or the definition is split across two chunks and neither chunk alone is retrieved.

3. **Generation**
   - Even when the definition **is** in the retrieved context, the model may default to a cautious summary: “the knowledge base only says…” instead of extracting the one line: “IMF means the International Monetary Fund.”

Tuning addresses (1) and (2) by improving **what** we retrieve and **how** we chunk, and (3) by **how we instruct** the model (generic instructions, not query-specific).

---

## 4. Tuning levers in this project

All of these are **generic**—they improve RAG behavior in general, not only for “what is IMF?”

### 4.1 Retrieval: how many chunks we use (`top_k` / `num_results`)

- **What it is:** Number of chunks we retrieve and put into the prompt (e.g. 5).
- **Where:**  
  - API: `POST /chat` body → `num_results` (default 5).  
  - Code: `services/rag.py` → `chat_response(..., num_results=...)`; `vector_search(..., top_k=num_results)`.
- **Effect:**  
  - **Larger (e.g. 8–10):** More chance the “definition” chunk is included; more context for the model.  
  - **Smaller (e.g. 3):** Less noise, but higher risk of missing the right chunk.
- **Tuning tip:** If definitions often sit at ranks 6–8, try `num_results=8` or `10` (via API or UI). No code change needed if the client sends `num_results`.

### 4.2 Chunking: size and overlap

- **What they are:**  
  - **Chunk size:** Max words per chunk (e.g. 300).  
  - **Overlap:** How many words overlap between consecutive chunks (e.g. 100).
- **Where:**  
  - Env: `TEXT_CHUNK_SIZE`, `TEXT_CHUNK_OVERLAP` (see `.env.example`).  
  - Code: `core/config.py` → `TEXT_CHUNK_SIZE`, `TEXT_CHUNK_OVERLAP`; `core/text_utils.py` → `chunk_text()`.
- **Effect:**  
  - **Smaller chunks (e.g. 200):** Short, focused chunks; a single sentence like “IMF means the International Monetary Fund” can be its own chunk and match “what is IMF?” well.  
  - **Larger chunks (e.g. 400):** More context per chunk but definitions can be diluted by surrounding text.  
  - **More overlap:** Reduces the chance a definition is cut at a chunk boundary.
- **Tuning tip:** For definition/acronym-heavy docs, try smaller chunks (e.g. 200) and overlap 80–100. **Re-upload the PDF** after changing these (chunking is done at ingest time).

### 4.3 Asymmetric embedding (query vs passage)

- **What it is:** When enabled, we prefix the **query** with `"passage: "` before encoding (passages are already stored with that prefix). Some models (e.g. certain SentenceTransformers) are trained for this and improve retrieval.
- **Where:**  
  - Env: `ASSYMETRIC_EMBEDDING=true` (or `false`).  
  - Code: `core/config.py` → `ASSYMETRIC_EMBEDDING`; `services/rag.py` → `query_prefix = "passage: " + query` when true; ingestion adds the same prefix to chunk text.
- **Effect:** Can improve retrieval quality for “question → passage” matching. Only applies to **new** ingestions (and re-encoding at query time). Existing ChromaDB data was ingested with or without the prefix; for a fair test, re-upload after changing.
- **Tuning tip:** Try `ASSYMETRIC_EMBEDDING=true` and re-ingest; compare retrieval and answer quality.

### 4.4 System prompt (how the model uses the context)

- **What it is:** The instructions that tell the model to answer only from the RAG context, be short, cite [1]/[2], etc.
- **Where:** `services/rag.py` → `SUPPORT_ASSISTANT_SYSTEM_PROMPT`.
- **Effect:**  
  - If we **never** tell the model to prefer a direct definition when the user asks “what is X?”, it may default to “the knowledge base says…” and not “X means Y.”  
  - A **generic** line like “When the user asks what something is, give the definition from the context when it appears (e.g. ‘X means Y’)” fixes this **for any acronym/term**, not only IMF.
- **Tuning tip:** Add one short, generic instruction for definition-style questions (see next section). Do **not** add query-specific text like “for IMF say International Monetary Fund.”

---

## 5. What we changed in this repo (generic only)

- **System prompt:** One sentence was added so that when the user asks **what** something is, the model is instructed to give the **definition from the context** when it’s there (e.g. “X means Y”). No mention of IMF or any specific term.
- So: same retrieval and chunking, but when the definition chunk is in the top‑k, the model is more likely to output the direct definition instead of only “the knowledge base only says…”.

If the definition chunk is still **not** in the top‑k, then retrieval or chunking must be improved (e.g. higher `num_results`, smaller chunks, re-upload, or try `ASSYMETRIC_EMBEDDING`).

---

## 6. Quick tuning checklist (no hardcoded query/answer)

1. **Increase `num_results`** (e.g. 8 or 10) in the chat request so more chunks—including a possible “definition” chunk—are sent to the LLM.
2. **Chunking:** Try `TEXT_CHUNK_SIZE=200` and `TEXT_CHUNK_OVERLAP=80` (or 100), then **re-upload** the PDFs.
3. **Asymmetric embedding:** Set `ASSYMETRIC_EMBEDDING=true` in `.env` and **re-upload** so queries and passages match the model’s expected format.
4. **Prompt:** Use the updated system prompt that includes the generic “for ‘what is X?’ give the definition from the context when present.” No static query/response.

After each change, test with “what is IMF?” and a few other definition-style questions. If the definition is in the docs but still not in the answer, the next step is usually **retrieval** (more chunks or better ranking) or **chunking** (smaller chunks so the definition is in a chunk that retrieves well).
