# CS4241 Manual RAG Chatbot — Adversarial Evaluation (PART E)
# Student: Yaw Acheampong Ahenkora Gyamera | Index: 10022200141
#
# Tests 2 adversarial queries and compares RAG vs pure LLM (no retrieval).
# Metrics: hallucination rate (numeric claims), consistency (N=5 runs), accuracy notes.
#
# Usage: python scripts/exp_adversarial.py
# Requires: OPENAI_API_KEY in .env, index built (scripts/build_index.py)

import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import EXPERIMENTS, INDEX_PATH, TOP_K
from src.vector_store import VectorStore
from src.retriever import Retriever
from src.prompt_engine import select_context, build_prompt
from src.llm_client import call_llm, call_llm_no_context
from src.evaluator import consistency_score, extract_numeric_claims, check_claims_in_context
from src.memory import classify_intent, compute_domain_boosts
from src.logger import banner, log_stage

OUT_FILE = EXPERIMENTS / "adversarial_eval.md"
N_CONSISTENCY_RUNS = 5

ADVERSARIAL_QUERIES = [
    {
        "id": "ADV1",
        "query": "Who won?",
        "type": "ambiguous",
        "description": (
            "Maximally ambiguous — no year, no race specified. "
            "A good RAG system should refuse or ask for clarification rather than guessing."
        ),
        "expected_rag_behaviour": "refusal or clarification request",
    },
    {
        "id": "ADV2",
        "query": "What was the 2026 budget allocation for education in Ghana?",
        "type": "misleading (wrong year)",
        "description": (
            "The documents contain 2025 budget data, not 2026. "
            "A hallucinating model will fabricate 2026 figures. "
            "A grounded RAG system should correct the year or decline."
        ),
        "expected_rag_behaviour": "correct year or explicit no-data response",
    },
]


def run_rag_once(query: str, retriever: Retriever) -> tuple[str, list[dict]]:
    """Run one RAG answer and return (answer, selected_chunks_as_dicts)."""
    intent = classify_intent(query)
    boosts = compute_domain_boosts(retriever.store.chunks, intent)
    results = retriever.retrieve(query, top_k=TOP_K, min_score=0.0, domain_boosts=boosts)
    selected = select_context(results)
    prompt = build_prompt(query, selected, version="v3_grounded_cite")
    answer = call_llm(prompt)
    chunks_dicts = [{"text": c.text} for c, _ in selected]
    return answer, chunks_dicts


def main() -> None:
    banner("ADVERSARIAL EVALUATION (PART E)")

    if not INDEX_PATH.exists():
        print("ERROR: Index not found. Run `python scripts/build_index.py` first.")
        sys.exit(1)

    store = VectorStore.load()
    retriever = Retriever(store)

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report = [
        "# Adversarial Evaluation — Experiment Log",
        f"*Run at: {ts}*  ",
        f"*Student: Yaw Acheampong Ahenkora Gyamera | Index: 10022200141*",
        "",
        "## Overview",
        "",
        "Two adversarial queries are tested:",
        "1. **ADV1 — Ambiguous**: 'Who won?' (no context in query)",
        "2. **ADV2 — Misleading year**: '2026 budget allocation for education'",
        "",
        "Each query is tested:",
        "- Through the full RAG pipeline (grounded, v3 prompt)",
        f"- Through pure LLM (no retrieval) — baseline",
        f"- {N_CONSISTENCY_RUNS} times for consistency scoring",
        "",
        "Metrics measured:",
        "- **Hallucination rate**: fraction of numeric claims not in retrieved chunks",
        "- **Consistency**: mean pairwise Jaccard similarity across N runs",
        "- **Accuracy**: human-verified factual correctness (filled in manually below)",
        "",
        "---",
        "",
    ]

    for case in ADVERSARIAL_QUERIES:
        query = case["query"]
        print(f"\n{'='*60}\n{case['id']}: {query}")
        log_stage("ADVERSARIAL_START", {"id": case["id"], "query": query, "type": case["type"]})

        # ── RAG: single run with full logging ────────────────────────────
        print(f"  Running RAG pipeline…")
        rag_answer, rag_chunks = run_rag_once(query, retriever)

        # ── Pure LLM: single run ─────────────────────────────────────────
        print(f"  Running pure LLM (no retrieval)…")
        llm_answer = call_llm_no_context(query)

        # ── Consistency: N runs ──────────────────────────────────────────
        print(f"  Running {N_CONSISTENCY_RUNS} consistency runs (RAG)…")
        rag_answers_all = [rag_answer]
        for i in range(N_CONSISTENCY_RUNS - 1):
            a, _ = run_rag_once(query, retriever)
            rag_answers_all.append(a)

        print(f"  Running {N_CONSISTENCY_RUNS} consistency runs (LLM)…")
        llm_answers_all = [llm_answer]
        for i in range(N_CONSISTENCY_RUNS - 1):
            llm_answers_all.append(call_llm_no_context(query))

        rag_consistency = consistency_score(rag_answers_all)
        llm_consistency = consistency_score(llm_answers_all)

        # ── Hallucination check ──────────────────────────────────────────
        rag_claims = extract_numeric_claims(rag_answer)
        rag_hallucination = check_claims_in_context(rag_claims, rag_chunks)
        llm_claims = extract_numeric_claims(llm_answer)
        # For pure LLM, no context to check against → all numeric claims are unverifiable
        llm_hallucination = {
            "total_claims": len(llm_claims),
            "supported": 0,
            "unsupported": len(llm_claims),
            "hallucination_rate": 1.0 if llm_claims else 0.0,
            "unsupported_claims": llm_claims,
        }

        log_stage("ADVERSARIAL_RESULT", {
            "id": case["id"],
            "rag_consistency_mean": rag_consistency["mean"],
            "llm_consistency_mean": llm_consistency["mean"],
            "rag_hallucination_rate": rag_hallucination["hallucination_rate"],
            "llm_hallucination_rate": llm_hallucination["hallucination_rate"],
        })

        report += [
            f"## {case['id']} — Type: `{case['type']}`",
            "",
            f"**Query**: `{query}`  ",
            f"**Description**: {case['description']}  ",
            f"**Expected RAG behaviour**: {case['expected_rag_behaviour']}",
            "",
            "### RAG Response (v3 prompt, with retrieval)",
            "",
            rag_answer,
            "",
            "### Pure LLM Response (no retrieval)",
            "",
            llm_answer,
            "",
            "### Metrics Comparison",
            "",
            "| Metric | RAG | Pure LLM |",
            "|--------|-----|----------|",
            f"| Consistency (Jaccard mean, N={N_CONSISTENCY_RUNS}) | {rag_consistency['mean']} | {llm_consistency['mean']} |",
            f"| Numeric claims | {rag_hallucination['total_claims']} | {llm_hallucination['total_claims']} |",
            f"| Hallucination rate (numeric) | {rag_hallucination['hallucination_rate']} | {llm_hallucination['hallucination_rate']} |",
            f"| Accuracy (human) | _fill in_ | _fill in_ |",
            "",
            "### Manual Analysis (student to complete)",
            "",
            f"- Did RAG refuse/clarify appropriately? *(fill in)*",
            f"- Did pure LLM hallucinate? *(fill in)*",
            f"- Evidence from output: *(quote the specific lines)*",
            "",
            "---",
            "",
        ]

    report += [
        "## Summary Table",
        "",
        "| Query ID | RAG Consistency | LLM Consistency | RAG Hallucination | LLM Hallucination |",
        "|----------|-----------------|-----------------|-------------------|-------------------|",
        "| ADV1     | _see above_     | _see above_     | _see above_       | _see above_       |",
        "| ADV2     | _see above_     | _see above_     | _see above_       | _see above_       |",
        "",
        "## Conclusion (student to complete)",
        "",
        "*(Summarise which system performed better and why, with evidence from the outputs above)*",
    ]

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text("\n".join(report), encoding="utf-8")
    print(f"\nReport written to {OUT_FILE}")


if __name__ == "__main__":
    main()
