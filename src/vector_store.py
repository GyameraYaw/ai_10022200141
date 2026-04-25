# CS4241 Manual RAG Chatbot — Custom Vector Store (PART B)
# Student: Yaw Acheampong Ahenkora Gyamera | Index: 10022200141
#
# Hand-rolled vector store using NumPy — no FAISS, Chroma, or any
# third-party vector-database library.
#
# Storage layout on disk:
#   data/processed/index.npz   — numpy array of shape (N, 384)
#   data/processed/chunks.json — list of Chunk dicts (parallel to rows in .npz)
#
# All similarity search is done via matrix-vector dot product, which equals
# cosine similarity when both sides are L2-normalised.

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.chunk import Chunk
from src.config import CHUNKS_PATH, INDEX_PATH, EMBEDDING_DIM
from src.logger import log_stage


class VectorStore:
    """
    A simple in-memory + on-disk vector store backed by NumPy.

    Attributes
    ----------
    embeddings : np.ndarray, shape (N, EMBEDDING_DIM)   — L2-normalised
    chunks     : list[Chunk], length N                   — parallel metadata
    """

    def __init__(self) -> None:
        self.embeddings: np.ndarray = np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
        self.chunks: list[Chunk] = []

    # ── Build ─────────────────────────────────────────────────────────────────

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        """
        Add chunks and their embeddings to the store.

        Parameters
        ----------
        chunks     : list of Chunk objects
        embeddings : np.ndarray of shape (len(chunks), EMBEDDING_DIM),
                     must already be L2-normalised.
        """
        assert len(chunks) == len(embeddings), "chunks and embeddings must be same length"

        if self.embeddings.shape[0] == 0:
            self.embeddings = embeddings.astype(np.float32)
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings.astype(np.float32)])

        self.chunks.extend(chunks)

        log_stage("VECTOR_STORE_ADD", {
            "added": len(chunks),
            "total": len(self.chunks),
        })

    # ── Persist ───────────────────────────────────────────────────────────────

    def save(self, index_path: Path = INDEX_PATH, chunks_path: Path = CHUNKS_PATH) -> None:
        """Persist embeddings (.npz) and chunk metadata (.json) to disk."""
        index_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(index_path), embeddings=self.embeddings)

        chunks_path.parent.mkdir(parents=True, exist_ok=True)
        with open(chunks_path, "w", encoding="utf-8") as f:
            json.dump([c.to_dict() for c in self.chunks], f, ensure_ascii=False, indent=2)

        log_stage("VECTOR_STORE_SAVE", {
            "index_path": str(index_path),
            "chunks_path": str(chunks_path),
            "n_chunks": len(self.chunks),
            "embedding_shape": list(self.embeddings.shape),
        })

    @classmethod
    def load(cls, index_path: Path = INDEX_PATH, chunks_path: Path = CHUNKS_PATH) -> "VectorStore":
        """Load a previously saved vector store from disk."""
        store = cls()
        data = np.load(str(index_path))
        store.embeddings = data["embeddings"].astype(np.float32)

        with open(chunks_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        store.chunks = [Chunk.from_dict(d) for d in raw]

        log_stage("VECTOR_STORE_LOAD", {
            "n_chunks": len(store.chunks),
            "embedding_shape": list(store.embeddings.shape),
        })
        return store

    # ── Search ────────────────────────────────────────────────────────────────

    def search(
        self,
        query_vec: np.ndarray,
        top_k: int = 5,
    ) -> list[tuple[Chunk, float]]:
        """
        Find the top-k most similar chunks via cosine similarity.

        Parameters
        ----------
        query_vec : 1-D L2-normalised vector of shape (EMBEDDING_DIM,)
        top_k     : number of results to return

        Returns
        -------
        List of (Chunk, cosine_score) tuples, sorted descending by score.
        """
        if self.embeddings.shape[0] == 0:
            return []

        # Cosine similarity = dot product (both sides normalised)
        scores: np.ndarray = self.embeddings @ query_vec          # shape (N,)
        top_k = min(top_k, len(self.chunks))
        top_idx = np.argpartition(scores, -top_k)[-top_k:]        # unsorted top-k
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]      # sort descending

        results = [(self.chunks[i], float(scores[i])) for i in top_idx]

        log_stage("VECTOR_STORE_SEARCH", {
            "top_k": top_k,
            "top_score": round(results[0][1], 4) if results else None,
            "bottom_score": round(results[-1][1], 4) if results else None,
        })

        return results
