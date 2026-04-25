# CS4241 — Introduction to Artificial Intelligence (2026)
# Manual RAG Chatbot — Ghana Election & Budget Assistant

**Student:** Yaw Acheampong Ahenkora Gyamera  
**Index Number:** 10022200141  
**Course:** CS4241 — Introduction to Artificial Intelligence  
**Lecturer:** Godwin N. Danso  
**Institution:** Academic City University, Faculty of Computational Sciences and Informatics

---

## Overview

A fully manual Retrieval-Augmented Generation (RAG) chatbot that answers questions grounded in:
- **Ghana 2024 Election Results** (constituency-level CSV)
- **Ghana 2025 Budget Statement** (full PDF)

**No LangChain, LlamaIndex, or pre-built RAG frameworks were used.**  
Every component — chunking, embedding, vector storage, retrieval, and prompt construction — is implemented from scratch.

---

## Project Structure

```
.
├── src/               # Core RAG modules (chunker, embedder, retriever, pipeline…)
├── scripts/           # Build index + experiment runners
├── experiments/       # Logged outputs from all experiments
├── docs/              # Architecture, design decisions, walkthrough script
├── diagrams/          # ASCII + PNG architecture diagrams
├── app/               # Streamlit UI
└── data/
    ├── raw/           # Source CSV + PDF
    └── processed/     # Cached embeddings + chunk index
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
python scripts/exp_chunking.py          # PART A  — chunking strategy comparison
python scripts/exp_retrieval_failures.py # PART B  — failure cases + hybrid fix
python scripts/exp_prompts.py           # PART C  — prompt iteration experiment
python scripts/exp_adversarial.py       # PART E  — adversarial evaluation
```

Results are written to `experiments/`.

---

## Tech Stack (no RAG frameworks)

| Component        | Library / Approach                                      |
|------------------|---------------------------------------------------------|
| PDF extraction   | `pypdf` (raw text extraction)                           |
| CSV loading      | `pandas` (standard data library)                        |
| Embeddings       | `sentence-transformers` — `all-MiniLM-L6-v2` (manual)  |
| Vector store     | Custom NumPy store (hand-rolled, no FAISS)              |
| Keyword search   | Custom BM25 (hand-rolled, no `rank_bm25`)               |
| LLM              | OpenAI `gpt-4o-mini` via official `openai` SDK          |
| UI               | Streamlit                                               |

---

## Novel Feature (PART G)

**Domain-specific scoring + session memory:**
- Classifies each query as `election_numeric`, `budget_policy`, or `general`
- Boosts retrieval scores for chunks that match the detected domain
- Maintains a ring buffer of recent turns to resolve follow-up questions

---

## Deployment

Deployed on Streamlit Community Cloud.  
Live URL: *(to be added after deploy)*

---

## Repository

GitHub: `https://github.com/yawgyamera/ai_10022200141`  
Collaborator invited: `GodwinDansoAcity`
