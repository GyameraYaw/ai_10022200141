# Design Decisions (Parts A–G)
**Student:** Yaw Acheampong Ahenkora Gyamera | **Index:** 10022200141

---

## PART A — Data Engineering

### Chunking Strategy: Why Sentence-Aware?

The core tradeoff in chunking is **semantic coherence vs retrieval precision**.

**Fixed-char** splits blindly. A 500-character window might cut "The NPP candidate in Ablekuma Central received 45,2" and leave "34 votes" in the next chunk. The embedding model will then represent an incomplete thought, degrading cosine similarity for vote-count queries.

**Sentence-aware** (chosen) respects linguistic boundaries. Each chunk contains complete sentences, so the embedding captures the full semantic meaning. A 50-token overlap ensures that sentences near boundaries appear in two adjacent chunks — preventing information loss at chunk edges.

**Structure-aware** (CSV row-as-chunk) was evaluated and achieves high precision for direct lookup queries ("How many votes did X get?") but poor recall for thematic questions ("What were the major electoral trends?") that span multiple rows or sections. It is best used as a retrieval filter, not as the primary strategy.

**Chunk size (≈400 tokens)**: large enough to carry meaningful context, small enough that one chunk doesn't dominate the entire context window. At 400 tokens × 5 chunks = 2,000 tokens of context — well within gpt-4o-mini's 128K context but focused enough to avoid noise.

**Overlap (50 tokens)**: empirically chosen to prevent information loss at boundaries without excessive duplication. Larger overlaps increase index size and embed redundant content.

---

## PART B — Retrieval System

### Why all-MiniLM-L6-v2?

- 384 dimensions: compact but expressive. Larger models (e.g., all-mpnet-base-v2, 768 dims) give marginal gains on retrieval quality but double memory and inference time.
- CPU-friendly: the exam machine may not have a GPU. MiniLM-L6 runs comfortably on CPU at ~100ms per batch.
- MIT license: freely usable without API cost on the embedding path.

### Why Custom NumPy Store Over FAISS?

The exam requires demonstrating manual implementation. A custom NumPy store explicitly shows:
- How embeddings are stored and loaded (`.npz`)
- How cosine similarity is computed (`emb @ q.T`)
- How top-k is extracted (`np.argpartition`)

FAISS abstracts all of this. For corpus sizes under ~50K chunks (our dataset is well below this), NumPy linear scan is sub-second and FAISS provides no meaningful speed benefit.

### Why Hybrid Search as the Enhancement?

The two other options (query expansion, re-ranking) were considered:
- **Query expansion**: adds synonyms/related terms before retrieval. Useful for thematic queries but can hurt precision for specific lookups (adding synonyms to "Ablekuma Central" might pull in wrong constituencies).
- **Re-ranking**: runs a cross-encoder over the top-k results. More accurate but requires a second model and doubles latency.
- **Hybrid search** (chosen): simple, fast, and directly addresses the primary failure mode — rare proper nouns. One model (BM25) handles exact token matching; the other (dense) handles semantic similarity. The combination covers both failure modes without adding a second neural model.

---

## PART C — Prompt Engineering

### Why v3 (grounded + citations)?

Tested three templates on the same query. Key observations:
- **v1 (naive)**: The LLM answered confidently but fabricated data when chunks were marginal.
- **v2 (grounded)**: Hallucination rate dropped significantly. But answers were vague when context was borderline.
- **v3 (grounded + cite)**: Citing chunk IDs forces the LLM to ground each claim explicitly. Any claim without a citation is immediately suspicious. The refusal clause ("I don't have enough information") was triggered correctly for out-of-scope queries.

### Context Window Management

- **Score filter (MIN_SCORE=0.20)**: removes noise chunks that are unlikely to be helpful.
- **Per-chunk truncation (3,000 chars)**: prevents a single very long PDF section from consuming the entire context budget.
- **Total budget (12,000 chars ≈ 3,000 tokens)**: leaves ≈4,000 tokens for the prompt template and ~1,000 for the response within gpt-4o-mini's limits.
- **Greedy ranking**: highest-scored chunks are included first. If the budget runs out, lower-scored chunks are dropped rather than truncated mid-sentence.

---

## PART D — Full Pipeline

### Logging Design

Each stage writes a JSON line to `experiments/runs/<timestamp>.jsonl`. This gives:
- A per-query audit trail
- Machine-readable data for post-hoc analysis
- Evidence for the manual experiment logs

The `log_stage(stage, payload)` helper is used by every module — there is no ad-hoc `print()` scattered through the code.

---

## PART E — Adversarial Testing

### Query Design Rationale

**ADV1 ("Who won?")**: Tests whether the system handles maximal ambiguity. A well-grounded RAG system should refuse or ask for clarification. A hallucinating system will pick a plausible-sounding winner from training memory.

**ADV2 ("2026 budget allocation for education")**: Tests year-grounding. The documents are clearly dated 2025. The system must either correct the year or explicitly refuse — not fabricate 2026 figures.

### Metrics

- **Consistency (Jaccard mean, N=5)**: measures whether the system gives stable answers. RAG should be more consistent than pure LLM because it grounds answers in the same retrieved chunks.
- **Hallucination rate (numeric)**: targets the most verifiable claim type. If the LLM invents a vote count, it will not appear in any retrieved chunk.
- **Accuracy**: human-verified. Requires the student to check each factual claim against the actual source documents.

---

## PART G — Novel Feature Rationale

### Domain-Specific Scoring + Session Memory

**Why domain scoring?** The corpus contains two very different domains (structured election data vs. flowing budget prose). A single embedding space treats both equally. Domain scoring applies a prior: if the query is about elections, election chunks should rank higher than structurally similar budget chunks. This is not hard-coded filtering (which would miss cross-domain queries) but a soft boost that can be overridden by strong embedding similarity.

**Why session memory?** Chatbot UX research consistently shows that users ask follow-up questions using anaphora ("what about in the Volta region?", "and the NPP?"). Without memory, each query is treated independently and the system fails to resolve these references. The ring buffer approach is deliberately lightweight — 5 turns of history, stored as plain JSON — rather than a vector memory (which would add complexity without clear benefit at this interaction scale).

**Why not feedback loop?** The feedback loop variant requires repeated user interactions to accumulate signal. For an exam demo, it would appear non-functional unless the student runs many queries. Domain scoring provides immediate, visible value from the first query.
