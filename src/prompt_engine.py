# CS4241 Manual RAG Chatbot — Prompt Engineering (PART C)
# Student: Yaw Acheampong Ahenkora Gyamera | Index: 10022200141
#
# Three prompt templates are defined and compared (see scripts/exp_prompts.py):
#   v1 — naive    : just dumps context + query, no instructions
#   v2 — grounded : adds source-grounding and refusal rules
#   v3 — grounded + cite (CHOSEN) : adds citation requirement + hallucination control
#
# Context window management:
#   - rank by score, take top_k
#   - truncate each chunk to MAX_CHUNK_CHARS
#   - filter chunks below MIN_SCORE
#   - enforce MAX_CONTEXT_CHARS hard cap (drop lowest-scored chunks to fit)

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chunk import Chunk
from src.config import MAX_CONTEXT_CHARS, MAX_CHUNK_CHARS, MIN_SCORE
from src.logger import log_stage

# ── Context window management ─────────────────────────────────────────────────

def select_context(
    results: list[tuple[Chunk, float, float, float]],
    min_score: float = MIN_SCORE,
    max_context_chars: int = MAX_CONTEXT_CHARS,
    max_chunk_chars: int = MAX_CHUNK_CHARS,
) -> list[tuple[Chunk, float]]:
    """
    Filter, truncate, and budget-fit retrieved chunks for injection into the prompt.

    Steps:
    1. Drop chunks below min_score.
    2. Truncate each chunk's text to max_chunk_chars.
    3. Take chunks greedily (highest score first) until the char budget is exhausted.

    Returns list of (Chunk, combined_score) in score-descending order.
    """
    # Support both (chunk, combined, vec, bm25) and (chunk, score) tuples
    normalised: list[tuple[Chunk, float]] = []
    for item in results:
        if len(item) == 4:
            chunk, combined, _, _ = item
        else:
            chunk, combined = item
        if combined >= min_score:
            normalised.append((chunk.truncated(max_chunk_chars), combined))

    # Sort descending by score
    normalised.sort(key=lambda x: x[1], reverse=True)

    # Apply character budget
    selected: list[tuple[Chunk, float]] = []
    used_chars = 0
    for chunk, score in normalised:
        if used_chars + len(chunk.text) > max_context_chars:
            break
        selected.append((chunk, score))
        used_chars += len(chunk.text)

    log_stage("CONTEXT_SELECTION", {
        "total_candidates": len(results),
        "after_filter": len(normalised),
        "selected": len(selected),
        "total_chars": used_chars,
    })

    return selected


# ── Context block builder ─────────────────────────────────────────────────────

def _build_context_block(selected: list[tuple[Chunk, float]]) -> str:
    """Format selected chunks into a numbered context string."""
    parts = []
    for i, (chunk, score) in enumerate(selected, 1):
        source_label = "Ghana Election Results" if chunk.source == "election_csv" else "Ghana 2025 Budget Statement"
        parts.append(
            f"[CHUNK {i} | Source: {source_label} | Score: {score:.4f} | ID: {chunk.id}]\n"
            f"{chunk.text}"
        )
    return "\n\n".join(parts)


# ── Prompt templates ──────────────────────────────────────────────────────────

def build_prompt_v1_naive(query: str, selected: list[tuple[Chunk, float]]) -> str:
    """
    v1 — Naive: no system role, no grounding instructions.
    Used as the baseline in the prompt comparison experiment.
    """
    context = _build_context_block(selected)
    return f"""Context:
{context}

Question: {query}

Answer:"""


def build_prompt_v2_grounded(query: str, selected: list[tuple[Chunk, float]]) -> str:
    """
    v2 — Grounded: adds source-grounding rule and a refusal clause.
    No citation requirement yet.
    """
    context = _build_context_block(selected)
    return f"""You are a knowledgeable assistant for Academic City University.

RULES:
- Answer ONLY using the context provided below.
- If the context does not contain enough information, respond with:
  "I don't have enough information in the provided documents to answer this."
- Do not make up facts, numbers, or names.

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""


def build_prompt_v3_grounded_cite(query: str, selected: list[tuple[Chunk, float]]) -> str:
    """
    v3 — Grounded + Citations (CHOSEN): adds citation requirement, explicit
    hallucination-control rules, and a structured response format.
    """
    context = _build_context_block(selected)
    return f"""You are a precise, factual assistant for Academic City University.
You answer questions about Ghana's 2024 election results and 2025 national budget.

## STRICT RULES (follow all of these):
1. Base your answer ONLY on the numbered CONTEXT CHUNKS below.
2. After each factual claim, cite the chunk(s) used as [CHUNK N].
3. If the context is insufficient or the question falls outside the documents, say:
   "The provided documents do not contain enough information to answer this question."
4. Do NOT invent statistics, names, or policies not present in the chunks.
5. If a question is ambiguous, ask for clarification rather than guessing.
6. Keep your answer concise and factual. Use bullet points for lists.

## CONTEXT CHUNKS:
{context}

## QUESTION:
{query}

## ANSWER (cite chunk numbers for every factual claim):"""


# ── Public API ────────────────────────────────────────────────────────────────

PROMPT_VERSIONS = {
    "v1_naive": build_prompt_v1_naive,
    "v2_grounded": build_prompt_v2_grounded,
    "v3_grounded_cite": build_prompt_v3_grounded_cite,
}


def build_prompt(
    query: str,
    selected: list[tuple[Chunk, float]],
    version: str = "v3_grounded_cite",
) -> str:
    """Build a prompt using the specified version template."""
    if version not in PROMPT_VERSIONS:
        raise ValueError(f"Unknown prompt version: {version}. Choose from {list(PROMPT_VERSIONS)}")

    prompt = PROMPT_VERSIONS[version](query, selected)

    log_stage("PROMPT_BUILD", {
        "version": version,
        "query": query[:80],
        "n_chunks": len(selected),
        "prompt_chars": len(prompt),
    })

    return prompt
