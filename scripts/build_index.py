# CS4241 Manual RAG Chatbot — Build Vector Index
# Student: Yaw Acheampong Ahenkora Gyamera | Index: 10022200141
#
# Pipeline: CSV + PDF → clean → chunk (sentence strategy) → embed → save
#
# Usage: python scripts/build_index.py
# Run this once before starting the app or any experiment scripts.

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tqdm import tqdm

from src.config import EMBEDDING_BATCH_SIZE, INDEX_PATH, CHUNKS_PATH
from src.data_loader import load_csv, load_pdf
from src.chunker import chunk_all
from src.embedder import encode
from src.vector_store import VectorStore
from src.logger import banner, log_stage


def main() -> None:
    banner("BUILD VECTOR INDEX")

    # ── 1. Load & clean data ─────────────────────────────────────────────────
    print("Loading CSV…")
    csv_rows = load_csv()
    print(f"  {len(csv_rows)} election rows loaded")

    print("Loading PDF…")
    pdf_pages = load_pdf()
    print(f"  {len(pdf_pages)} budget pages loaded")

    # ── 2. Chunk (sentence strategy — chosen) ────────────────────────────────
    print("\nChunking with sentence strategy…")
    chunks = chunk_all(csv_rows, pdf_pages, strategy="sentence")
    print(f"  {len(chunks)} chunks created")
    log_stage("BUILD_CHUNKS", {
        "total": len(chunks),
        "csv": sum(1 for c in chunks if c.source == "election_csv"),
        "pdf": sum(1 for c in chunks if c.source == "budget_pdf"),
    })

    # ── 3. Embed ─────────────────────────────────────────────────────────────
    print("\nEmbedding chunks (this may take a few minutes on CPU)…")
    texts = [c.text for c in chunks]

    all_vecs = []
    for start in tqdm(range(0, len(texts), EMBEDDING_BATCH_SIZE), desc="Embedding", unit="batch"):
        batch_texts = texts[start: start + EMBEDDING_BATCH_SIZE]
        batch_vecs = encode(batch_texts)
        all_vecs.append(batch_vecs)

    import numpy as np
    embeddings = np.vstack(all_vecs)
    print(f"  Embeddings shape: {embeddings.shape}")

    # ── 4. Save ──────────────────────────────────────────────────────────────
    print("\nSaving vector store…")
    store = VectorStore()
    store.add(chunks, embeddings)
    store.save(INDEX_PATH, CHUNKS_PATH)
    print(f"  Saved → {INDEX_PATH}")
    print(f"  Saved → {CHUNKS_PATH}")

    print("\nIndex built successfully. Run `streamlit run app/streamlit_app.py` to start.")


if __name__ == "__main__":
    main()
