# Manual RAG Chatbot — Ghana Election & Budget Assistant

**Student:** Yaw Acheampong Ahenkora Gyamera | **Index:** 10022200141
**Institution:** Academic City University College
**Deployed:** [https://rag-ghana-chatbot.streamlit.app](https://rag-ghana-chatbot.streamlit.app)

---

## Overview

A fully manual Retrieval-Augmented Generation (RAG) chatbot that answers questions grounded in two Ghanaian datasets:
- **Ghana Presidential Election Results (1992–2020)** — regional-level CSV
- **Ghana 2025 Budget Statement** — full 252-page PDF

**No LangChain, LlamaIndex, or pre-built RAG frameworks were used.**
Every component — chunking, embedding, vector storage, retrieval, and prompt construction — is implemented from scratch in Python.

---

## Data Sources

**Ghana Presidential Election Results (1992–2020)**
Regional vote totals for all major presidential candidates across eight general elections. Covers parties including NDC, NPP, PNC, and CPP at the regional granularity. Does not contain constituency-level breakdowns or 2024 results.

**Ghana 2025 Budget Statement**
The full 252-page budget statement released by the Ministry of Finance, covering macroeconomic targets, sector-by-sector expenditure allocations, revenue measures, and fiscal policy narratives for the 2025 financial year.

---

## Project Structure

```
.
├── src/               # Core RAG modules (chunker, embedder, retriever, pipeline…)
├── scripts/           # Build index + experiment runners
├── experiments/       # Logged outputs from all experiments (Parts A–G)
├── docs/              # Architecture, design decisions, report
├── diagrams/          # Architecture diagram (PNG + drawio)
├── app/               # Streamlit UI
└── data/
    ├── raw/           # Source CSV + PDF (not committed)
    └── processed/     # Cached embeddings + chunk index (committed)
```

---

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your OpenAI API key
cp .env.example .env
# Edit .env and fill in OPENAI_API_KEY

# 4. Download the data
python scripts/download_data.py

# 5. Build the vector index
python scripts/build_index.py

# 6. Launch the app
streamlit run app/streamlit_app.py
```

---

## Running Experiments

```bash
python scripts/exp_chunking.py           # Part A — chunking strategy comparison
python scripts/exp_retrieval_failures.py  # Part B — failure cases + hybrid fix
python scripts/exp_prompts.py            # Part C — prompt iteration experiment
python scripts/exp_adversarial.py        # Part E — adversarial evaluation
```

Results are written to `experiments/`. Manual analysis notes are in `experiments/manual_experiment_log.md`.

---

## Tech Stack (no RAG frameworks)

| Component      | Library / Approach                                     |
|----------------|--------------------------------------------------------|
| PDF extraction | `pypdf` (raw text extraction)                          |
| CSV loading    | `pandas` (standard data library)                       |
| Embeddings     | `sentence-transformers` — `all-MiniLM-L6-v2` (manual) |
| Vector store   | Custom NumPy store (hand-rolled, no FAISS)             |
| Keyword search | Custom BM25 (hand-rolled, no `rank_bm25`)              |
| LLM            | OpenAI `gpt-4o-mini` via official `openai` SDK         |
| UI             | Streamlit                                              |

---

## Key Experiment Results

| Experiment | Finding |
|------------|---------|
| Chunking (Part A) | Sentence strategy: 1,010 chunks, MRR@5 = 0.533, Hit@3 = 0.600 — best balance of quality and index size |
| Retrieval (Part B) | Hybrid BM25 + vector moved ABFA chunk from rank 3 (score 0.661) to rank 1 (score 0.720) via exact keyword match |
| Adversarial (Part E) | RAG: consistency 1.0, hallucination rate 0.0 — Pure LLM: consistency 0.67, hallucination rate 1.0 |

---

## Novel Feature (Part G)

**Domain-specific scoring + session memory:**
- Classifies each query as `election_numeric`, `budget_policy`, or `general`
- Applies a +0.10 score boost to chunks matching the detected domain
- Maintains a per-session ring buffer (5 turns) to resolve anaphoric follow-up questions

---

## Example Questions

**These work:**
- How many votes did the NPP get in the Ashanti Region in 2016?
- What is Ghana's GDP growth target for 2025?
- What revenue measures were introduced in the 2025 budget?
- *(follow-up)* What about in 2012? ← session memory resolves the reference

**These will return a refusal (data not in corpus):**
- Who won the 2024 presidential election? ← dataset ends at 2020
- How many votes did the NDC get in Tema East constituency? ← regional data only
- What is the 2026 budget allocation for education? ← document is dated 2025

---

## Known Limitations

- The election CSV contains **regional presidential vote totals only** — no constituency-level results, no parliamentary results, no 2024 data
- Budget queries that require aggregating figures across multiple sections (e.g. "total health sector allocation") may return a refusal — the grounding prompt refuses to synthesise a number it cannot directly cite
- The system is grounded in its source documents only; it will not answer general knowledge questions

---

## Deployment

Deployed on Streamlit Community Cloud.
**Live URL:** [https://rag-ghana-chatbot.streamlit.app](https://rag-ghana-chatbot.streamlit.app)
