# CS4241 Manual RAG Chatbot — Full RAG Pipeline (PART D)
# Student: Yaw Acheampong Ahenkora Gyamera | Index: 10022200141
#
# End-to-end pipeline:
#   User Query → (anaphora resolve) → Intent classify → Domain boosts →
#   Hybrid Retrieval → Context selection → Prompt build → LLM → Response
#
# Every stage is logged to experiments/runs/<timestamp>.jsonl

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chunk import Chunk
from src.config import TOP_K, MIN_SCORE, INDEX_PATH, CHUNKS_PATH, RUNS
from src.embedder import encode_single
from src.vector_store import VectorStore
from src.retriever import Retriever
from src.prompt_engine import select_context, build_prompt
from src.llm_client import call_llm
from src.memory import SessionMemory, classify_intent, compute_domain_boosts
from src.logger import log_stage, add_file_sink, clear_sinks, banner


class RAGPipeline:
    """
    Orchestrates the full RAG pipeline with per-stage logging.

    Parameters
    ----------
    prompt_version : which prompt template to use (default: v3_grounded_cite)
    top_k          : number of chunks to retrieve
    min_score      : minimum combined score to include a chunk
    use_memory     : whether to use session memory (PART G novel feature)
    """

    def __init__(
        self,
        prompt_version: str = "v3_grounded_cite",
        top_k: int = TOP_K,
        min_score: float = MIN_SCORE,
        use_memory: bool = True,
    ) -> None:
        self.prompt_version = prompt_version
        self.top_k = top_k
        self.min_score = min_score
        self.use_memory = use_memory

        # Load vector store and build retriever
        self.store = VectorStore.load(INDEX_PATH, CHUNKS_PATH)
        self.retriever = Retriever(self.store, top_k=top_k)
        self.memory = SessionMemory()

    def answer(self, query: str) -> dict:
        """
        Run the full pipeline for a user query.

        Returns
        -------
        dict with keys:
            query          : str
            resolved_query : str (after anaphora resolution)
            intent         : str
            retrieved      : list of {chunk_id, text, source, vector_score, bm25_score, combined_score}
            selected       : list of {chunk_id, text, score}
            prompt         : str (final prompt sent to LLM)
            answer         : str (LLM response)
            timestamp      : str
        """
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = RUNS / f"query_{ts}.jsonl"
        RUNS.mkdir(parents=True, exist_ok=True)
        clear_sinks()
        add_file_sink(log_path)

        banner(f"RAG PIPELINE — {ts}")
        log_stage("QUERY_RECEIVED", {"query": query})

        # ── Stage 1: Anaphora resolution (PART G) ─────────────────────────
        resolved_query = query
        if self.use_memory:
            resolved_query = self.memory.resolve_anaphora(query)
        log_stage("STAGE_1_ANAPHORA", {"original": query, "resolved": resolved_query})

        # ── Stage 2: Intent classification (PART G) ───────────────────────
        intent = classify_intent(resolved_query)
        log_stage("STAGE_2_INTENT", {"intent": intent})

        # ── Stage 3: Domain boosts (PART G) ───────────────────────────────
        domain_boosts: dict[str, float] = {}
        if self.use_memory:
            domain_boosts = compute_domain_boosts(self.store.chunks, intent)
            # Add recency penalties to discourage re-fetching same chunks
            penalties = self.memory.recency_penalties(self.store.chunks)
            for cid, pen in penalties.items():
                domain_boosts[cid] = domain_boosts.get(cid, 0.0) + pen

        log_stage("STAGE_3_DOMAIN_BOOSTS", {
            "n_boosted": len([v for v in domain_boosts.values() if v > 0]),
            "n_penalised": len([v for v in domain_boosts.values() if v < 0]),
        })

        # ── Stage 4: Retrieval ─────────────────────────────────────────────
        raw_results = self.retriever.retrieve(
            resolved_query,
            top_k=self.top_k,
            min_score=0.0,          # filter later in select_context
            domain_boosts=domain_boosts if domain_boosts else None,
        )
        log_stage("STAGE_4_RETRIEVAL", {
            "query": resolved_query[:80],
            "n_results": len(raw_results),
            "top_chunks": [
                {"id": c.id, "combined": round(cs, 4), "vec": round(vs, 4), "bm25": round(bs, 4)}
                for c, cs, vs, bs in raw_results[:self.top_k]
            ],
        })

        # ── Stage 5: Context selection ─────────────────────────────────────
        selected = select_context(raw_results, min_score=self.min_score)
        log_stage("STAGE_5_CONTEXT", {
            "n_selected": len(selected),
            "chunk_ids": [c.id for c, _ in selected],
        })

        # ── Stage 6: Prompt construction ───────────────────────────────────
        prompt = build_prompt(resolved_query, selected, version=self.prompt_version)
        log_stage("STAGE_6_PROMPT", {
            "version": self.prompt_version,
            "prompt_chars": len(prompt),
            "prompt_preview": prompt[:300],
        })

        # ── Stage 7: LLM call ──────────────────────────────────────────────
        answer_text = call_llm(prompt)
        log_stage("STAGE_7_LLM", {"answer": answer_text})

        # ── Stage 8: Memory update (PART G) ───────────────────────────────
        if self.use_memory:
            retrieved_ids = [c.id for c, _ in selected]
            self.memory.push(query, retrieved_ids, answer_text)
            self.memory.save()

        log_stage("STAGE_8_MEMORY_UPDATED", {"turns_stored": len(self.memory.as_list())})

        # ── Return structured result ───────────────────────────────────────
        result = {
            "query": query,
            "resolved_query": resolved_query,
            "intent": intent,
            "retrieved": [
                {
                    "chunk_id": c.id,
                    "text": c.text,
                    "source": c.source,
                    "combined_score": round(cs, 4),
                    "vector_score": round(vs, 4),
                    "bm25_score": round(bs, 4),
                }
                for c, cs, vs, bs in raw_results[:self.top_k]
            ],
            "selected": [
                {"chunk_id": c.id, "text": c.text, "score": round(score, 4)}
                for c, score in selected
            ],
            "prompt": prompt,
            "answer": answer_text,
            "timestamp": ts,
            "log_path": str(log_path),
        }

        return result
