# CS4241 Manual RAG Chatbot — Embedding Pipeline (PART B)
# Student: Yaw Acheampong Ahenkora Gyamera | Index: 10022200141
#
# Manual embedding pipeline using sentence-transformers.
# No LangChain / LlamaIndex / pre-built RAG framework is used.
#
# Design choices:
#   - Model: all-MiniLM-L6-v2 (384 dims, fast CPU inference, MIT license)
#   - Batching: configurable batch size to avoid OOM on large corpora
#   - Normalisation: L2-normalise every vector so cosine similarity = dot product
#   - Caching: skip re-encoding if the index already exists (see build_index.py)

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE, EMBEDDING_DIM
from src.logger import log_stage

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """Lazy-load the embedding model (singleton)."""
    global _model
    if _model is None:
        log_stage("EMBEDDER_LOAD", {"model": EMBEDDING_MODEL})
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def encode(texts: list[str], batch_size: int = EMBEDDING_BATCH_SIZE) -> np.ndarray:
    """
    Encode a list of texts into L2-normalised embedding vectors.

    Parameters
    ----------
    texts      : list of strings to embed
    batch_size : number of texts to process per forward pass

    Returns
    -------
    np.ndarray of shape (len(texts), EMBEDDING_DIM), dtype float32, L2-normalised.
    """
    if not texts:
        return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)

    model = get_model()

    # Manual batching so we can log progress on large corpora
    all_vecs: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start: start + batch_size]
        vecs = model.encode(
            batch,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,  # we normalise ourselves below
        )
        all_vecs.append(vecs)

    embeddings = np.vstack(all_vecs).astype(np.float32)

    # L2 normalisation: v / ||v||
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-9, norms)
    embeddings = embeddings / norms

    log_stage("EMBEDDER_ENCODE", {
        "n_texts": len(texts),
        "shape": list(embeddings.shape),
        "norm_min": float(np.linalg.norm(embeddings, axis=1).min()),
        "norm_max": float(np.linalg.norm(embeddings, axis=1).max()),
    })

    return embeddings


def encode_single(text: str) -> np.ndarray:
    """Encode and normalise a single string. Returns shape (EMBEDDING_DIM,)."""
    return encode([text])[0]
