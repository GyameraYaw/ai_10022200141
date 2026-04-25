# CS4241 Manual RAG Chatbot — Retrieval Logic (PART B)
# Student: Yaw Acheampong Ahenkora Gyamera | Index: 10022200141
#
# Implements:
#   1. Top-k vector retrieval (cosine similarity via VectorStore.search)
#   2. Custom BM25 keyword scoring (hand-rolled, no rank_bm25 library)
#   3. Hybrid search: α * vector_score + (1-α) * normalised_bm25_score
#
# The retrieval enhancement chosen is HYBRID SEARCH because the domain
# contains many exact-match tokens (constituency names, Cedi amounts,
# ministry names) that dense embeddings sometimes map ambiguously.

import math
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.chunk import Chunk
from src.config import TOP_K, MIN_SCORE, HYBRID_ALPHA
from src.embedder import encode_single
from src.logger import log_stage
from src.vector_store import VectorStore


# ── Tokeniser ─────────────────────────────────────────────────────────────────

_NON_ALPHA = re.compile(r"[^a-z0-9\s]")


def _tokenise(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    return _NON_ALPHA.sub(" ", text.lower()).split()


# ── Custom BM25 ───────────────────────────────────────────────────────────────

class BM25:
    """
    Hand-rolled BM25 (Okapi BM25) scorer.

    Reference: Robertson & Zaragoza (2009).
    k1 and b are standard defaults; not tuned for this dataset.
    """

    k1 = 1.5
    b = 0.75

    def __init__(self, corpus: list[str]) -> None:
        self.n_docs = len(corpus)
        self.tokenised = [_tokenise(doc) for doc in corpus]
        self.dl = [len(t) for t in self.tokenised]
        self.avgdl = sum(self.dl) / max(self.n_docs, 1)

        # Document frequency for each term
        self.df: dict[str, int] = {}
        for tokens in self.tokenised:
            for term in set(tokens):
                self.df[term] = self.df.get(term, 0) + 1

        # IDF (smooth)
        self.idf: dict[str, float] = {
            term: math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1.0)
            for term, df in self.df.items()
        }

    def score(self, query: str, doc_idx: int) -> float:
        """BM25 score for a single (query, document) pair."""
        q_tokens = _tokenise(query)
        tokens = self.tokenised[doc_idx]
        tf_map = Counter(tokens)
        dl = self.dl[doc_idx]

        score = 0.0
        for term in q_tokens:
            if term not in self.idf:
                continue
            tf = tf_map.get(term, 0)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            score += self.idf[term] * (numerator / denominator)
        return score

    def scores_for_query(self, query: str) -> np.ndarray:
        """Return BM25 scores for all documents given a query."""
        return np.array([self.score(query, i) for i in range(self.n_docs)], dtype=np.float32)


# ── Retriever ─────────────────────────────────────────────────────────────────

class Retriever:
    """
    Hybrid retriever combining dense vector search and sparse BM25.

    Parameters
    ----------
    store : loaded VectorStore
    alpha : weight given to vector score; (1-alpha) to BM25
    top_k : number of final results
    """

    def __init__(
        self,
        store: VectorStore,
        alpha: float = HYBRID_ALPHA,
        top_k: int = TOP_K,
    ) -> None:
        self.store = store
        self.alpha = alpha
        self.top_k = top_k

        # Pre-build BM25 index over all chunk texts
        log_stage("RETRIEVER_BUILD_BM25", {"n_chunks": len(store.chunks)})
        self._bm25 = BM25([c.text for c in store.chunks])

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        min_score: float = MIN_SCORE,
        domain_boosts: dict[str, float] | None = None,
    ) -> list[tuple[Chunk, float, float, float]]:
        """
        Retrieve top-k chunks using hybrid scoring.

        Parameters
        ----------
        query        : user query string
        top_k        : override default top_k
        min_score    : drop results below this combined score
        domain_boosts: {chunk_id: boost_value} from the domain scorer (PART G)

        Returns
        -------
        List of (Chunk, combined_score, vector_score, bm25_score) sorted by combined_score desc.
        """
        k = top_k or self.top_k

        # ── Vector scores ──────────────────────────────────────────────────
        q_vec = encode_single(query)
        vector_results = self.store.search(q_vec, top_k=len(self.store.chunks))
        # dict: chunk_id → (chunk, raw_vector_score)
        vec_map: dict[str, tuple[Chunk, float]] = {c.id: (c, s) for c, s in vector_results}

        # ── BM25 scores ────────────────────────────────────────────────────
        bm25_raw = self._bm25.scores_for_query(query)
        bm25_max = bm25_raw.max()
        bm25_norm = bm25_raw / (bm25_max + 1e-9)   # normalise to [0, 1]

        # ── Combine ────────────────────────────────────────────────────────
        candidates: list[tuple[Chunk, float, float, float]] = []
        for i, chunk in enumerate(self.store.chunks):
            vec_score = vec_map.get(chunk.id, (chunk, 0.0))[1]
            bm25_score = float(bm25_norm[i])
            combined = self.alpha * vec_score + (1 - self.alpha) * bm25_score

            # Apply domain boost from PART G if provided
            if domain_boosts and chunk.id in domain_boosts:
                combined += domain_boosts[chunk.id]

            candidates.append((chunk, combined, vec_score, bm25_score))

        # Sort by combined score descending
        candidates.sort(key=lambda x: x[1], reverse=True)

        # Filter by minimum score and take top_k
        results = [(c, cs, vs, bs) for c, cs, vs, bs in candidates if cs >= min_score][:k]

        log_stage("RETRIEVER_RESULTS", {
            "query": query[:80],
            "top_k": k,
            "returned": len(results),
            "scores": [{"id": c.id, "combined": round(cs, 4), "vec": round(vs, 4), "bm25": round(bs, 4)}
                       for c, cs, vs, bs in results],
        })

        return results

    def retrieve_vector_only(
        self,
        query: str,
        top_k: int | None = None,
        min_score: float = MIN_SCORE,
    ) -> list[tuple[Chunk, float]]:
        """Pure vector retrieval (no BM25) — used for failure case demonstration."""
        k = top_k or self.top_k
        q_vec = encode_single(query)
        results = self.store.search(q_vec, top_k=k)
        return [(c, s) for c, s in results if s >= min_score]
