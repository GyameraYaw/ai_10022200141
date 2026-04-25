# CS4241 Manual RAG Chatbot — Evaluation Metrics (PART E)
# Student: Yaw Acheampong Ahenkora Gyamera | Index: 10022200141
#
# Implements three measurable metrics:
#   1. Accuracy     — fraction of factual claims that are verifiable against source docs
#   2. Hallucination rate — fraction of claims NOT supported by retrieved chunks
#   3. Consistency  — Jaccard overlap across N repeated runs (temperature=0.2)
#
# NOTE: Accuracy and hallucination rate require human verification of the outputs
# (the exam requires manual logs, not AI-auto-summaries). These functions compute
# scaffold metrics that the student fills in with their own analysis.

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.logger import log_stage


# ── Consistency metric ────────────────────────────────────────────────────────

def _tokenise(text: str) -> set[str]:
    return set(re.sub(r"[^a-z0-9\s]", " ", text.lower()).split())


def jaccard_similarity(a: str, b: str) -> float:
    """Token-level Jaccard similarity between two strings."""
    ta, tb = _tokenise(a), _tokenise(b)
    if not ta and not tb:
        return 1.0
    intersection = len(ta & tb)
    union = len(ta | tb)
    return intersection / union if union else 0.0


def consistency_score(answers: list[str]) -> dict:
    """
    Compute pairwise Jaccard similarity across N repeated answers.

    Returns dict with 'mean', 'min', 'max', 'n_pairs'.
    """
    if len(answers) < 2:
        return {"mean": 1.0, "min": 1.0, "max": 1.0, "n_pairs": 0}

    scores = []
    for i in range(len(answers)):
        for j in range(i + 1, len(answers)):
            scores.append(jaccard_similarity(answers[i], answers[j]))

    result = {
        "mean": round(sum(scores) / len(scores), 4),
        "min": round(min(scores), 4),
        "max": round(max(scores), 4),
        "n_pairs": len(scores),
    }
    log_stage("CONSISTENCY", result)
    return result


# ── Hallucination detection scaffold ─────────────────────────────────────────

def extract_numeric_claims(text: str) -> list[str]:
    """
    Extract sentences containing numeric values from LLM output.
    These are the most verifiable claims (vote counts, percentages, budget figures).
    """
    sentences = re.split(r"[.!?]", text)
    return [s.strip() for s in sentences if re.search(r"\d", s) and len(s.strip()) > 10]


def check_claims_in_context(claims: list[str], context_chunks: list[dict]) -> dict:
    """
    For each claim, check whether its key numeric token appears in any retrieved chunk.
    Returns dict with supported/unsupported counts.

    This is a surface-level check — human judgment is still required for full verification.
    """
    context_text = " ".join(c.get("text", "") for c in context_chunks).lower()
    supported = 0
    unsupported = []

    for claim in claims:
        numbers = re.findall(r"\d[\d,]*", claim)
        found = any(num.replace(",", "") in context_text.replace(",", "") for num in numbers)
        if found:
            supported += 1
        else:
            unsupported.append(claim)

    result = {
        "total_claims": len(claims),
        "supported": supported,
        "unsupported": len(unsupported),
        "hallucination_rate": round(len(unsupported) / max(len(claims), 1), 4),
        "unsupported_claims": unsupported,
    }
    log_stage("HALLUCINATION_CHECK", result)
    return result
