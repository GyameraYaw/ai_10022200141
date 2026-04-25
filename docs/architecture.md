# Architecture & System Design (PART F)
**Student:** Yaw Acheampong Ahenkora Gyamera | **Index:** 10022200141

---

## Overview

This system is a fully manual RAG chatbot with no dependency on LangChain, LlamaIndex, or any pre-built RAG framework. Every component — from chunking to vector storage to retrieval to prompt construction — is implemented from scratch in Python.

The architecture splits into two phases: a **build-time pipeline** that processes documents once and creates a persistent vector index, and a **query-time pipeline** that answers user questions in real time.

See `diagrams/architecture.txt` for the full ASCII diagram and `diagrams/architecture.png` for the visual version.

---

## Build-Time Pipeline

### 1. Data Ingestion (`src/data_loader.py`)

**Ghana_Election_Result.csv** is loaded via `pandas` with dtype=str to prevent silent type coercion. Cleaning steps: normalize column names (lowercase, underscores), strip whitespace, drop rows with null constituency or party, coerce vote columns to int, remove duplicates. Result: a clean list of row dicts.

**2025-Budget-Statement.pdf** is extracted via `pypdf`'s raw text extractor (not a RAG framework — just a PDF library). Each page is cleaned: headers/footers removed via regex, hyphenated line-breaks reattached, whitespace normalised. Pages below 50 characters are discarded (blank/cover pages).

**Why these choices**: `pandas` and `pypdf` are foundational data libraries, not RAG frameworks. They operate at the IO/parsing layer only.

### 2. Chunking (`src/chunker.py`)

Three strategies were implemented and compared (see `experiments/chunking_results.md`):

| Strategy | Method | Pros | Cons |
|----------|--------|------|------|
| `fixed_char` | Sliding window of 500 chars, 50 overlap | Simple, predictable | Cuts mid-sentence, hurts embedding quality |
| `sentence` | Groups sentences to ≈400 tokens, 50-token overlap | Preserves semantics | Slightly longer to compute |
| `structure` | CSV row-as-chunk; PDF section headings | High precision for direct lookups | Low recall for cross-section queries |

**Decision: `sentence` strategy** is used for the production index. It achieves the highest MRR@5 and Hit@3 in the evaluation. The CSV row structure (constituency + candidate + votes) maps naturally to a single sentence-level chunk; the PDF's flowing prose benefits most from sentence-aware segmentation.

CSV metadata (constituency, region, party, votes) is preserved in `Chunk.metadata` regardless of chunking strategy.

### 3. Embedding (`src/embedder.py`)

Model: `sentence-transformers/all-MiniLM-L6-v2` — 384-dimensional dense vectors, runs on CPU, MIT-licensed, no API cost on the embedding path.

Manual implementation: texts are batched (batch_size=64), passed through the SentenceTransformer, then L2-normalised by hand (`v / ‖v‖`). Normalisation means cosine similarity reduces to a dot product, making the vector store's search a simple matrix multiplication.

### 4. Vector Store (`src/vector_store.py`)

A custom `VectorStore` class holds:
- `embeddings: np.ndarray` of shape (N, 384) — L2-normalised
- `chunks: list[Chunk]` — parallel metadata list

Persisted to `data/processed/index.npz` (compressed NumPy) and `chunks.json`. No FAISS, Chroma, or external vector database. All search is `store.embeddings @ query_vec` (matrix-vector dot product).

---

## Query-Time Pipeline

### Stage 1: Anaphora Resolution (`src/memory.py`)

If the query begins with an anaphoric token ("and", "what about", "also"), the previous query is prepended. This lets follow-up questions like "and what about the NDC?" retrieve correctly without requiring users to repeat context.

### Stage 2: Intent Classification (`src/memory.py`)

Regex-based classifier labels the query as `election_numeric`, `budget_policy`, or `general`. Token hits for each domain are counted; the higher-scoring domain wins.

### Stage 3: Domain Boosts (`src/memory.py`)

Chunks whose `source_intent` matches the detected intent receive a +β (≈0.10) score bonus. Additional signals:
- **Election queries**: chunks with high numeric density (many vote counts) get an extra +β×0.5 × density.
- **Budget queries**: chunks with monetary patterns (GH¢, percentages, billion) get +β×0.5 × density.
- Chunks retrieved in the previous turn receive a −0.05 recency penalty to encourage diversity.

### Stage 4: Hybrid Retrieval (`src/retriever.py`)

**combined_score = α × cosine + (1-α) × BM25_norm + domain_boost**

- α = 0.7 (70% weight to dense semantic signal)
- BM25 is a custom hand-rolled implementation (Okapi BM25, k1=1.5, b=0.75)
- BM25 scores are normalised to [0,1] before combining

**Why hybrid**: Dense embeddings handle semantically similar queries well but lose specificity on rare proper nouns (constituency names, acronyms). BM25 provides exact-term matching that rescues these cases (demonstrated in `experiments/retrieval_failures.md`).

### Stage 5: Context Selection (`src/prompt_engine.py`)

1. Filter chunks below `MIN_SCORE` (0.20)
2. Truncate each chunk to `MAX_CHUNK_CHARS` (3,000 chars)
3. Greedily fill up to `MAX_CONTEXT_CHARS` (12,000 chars) from highest-scored chunks

### Stage 6: Prompt Construction (`src/prompt_engine.py`)

Three templates were compared in `experiments/prompt_iterations.md`:
- `v1_naive` — baseline, no instructions
- `v2_grounded` — grounding + refusal clause
- `v3_grounded_cite` — grounding + citations + hallucination controls (chosen)

The chosen template (v3) instructs the LLM to: (a) use only the numbered context chunks, (b) cite chunk IDs inline, (c) refuse rather than guess when context is insufficient.

### Stage 7: LLM Call (`src/llm_client.py`)

Raw call to OpenAI `gpt-4o-mini` via the official `openai` Python SDK. No LangChain wrapper. `temperature=0.2` for reproducibility.

### Stage 8: Memory Update (`src/memory.py`)

The ring buffer (N=5 turns) stores the query, retrieved chunk IDs, and answer. Persisted to `data/processed/session.json` between Streamlit sessions.

---

## Why This Design Is Suitable for the Domain

1. **Hybrid retrieval** is essential here because Ghana election data contains hundreds of highly specific proper nouns (constituency names, party abbreviations) that sentence transformers struggle with. BM25's exact-match signal is domain-critical.

2. **Domain-specific scoring** respects the dual-source nature of the corpus: election queries should bias toward CSV chunks (numeric, structured); budget queries should bias toward PDF chunks (policy prose).

3. **Session memory** addresses how users naturally interact with domain data: they often ask a general question ("Who won the election?") then drill down ("What about in the Volta region?"). Anaphora resolution makes this conversation flow naturally.

4. **Row-as-sentence chunking** for the CSV preserves each result's integrity: splitting a row mid-way (e.g., constituency name in chunk N, vote count in chunk N+1) would break retrieval for numeric queries.

5. **NumPy vector store** is appropriate at this corpus scale (thousands of chunks). The linear scan `O(N)` is fast enough — sub-second on a CPU — and avoids the complexity of an external vector database for an exam-scale dataset.
