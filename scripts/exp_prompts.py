# CS4241 Manual RAG Chatbot — Prompt Engineering Experiment (PART C)
# Student: Yaw Acheampong Ahenkora Gyamera | Index: 10022200141
#
# Runs the same query through all 3 prompt versions and logs the outputs.
# Demonstrates how prompt design affects response quality and hallucination rate.
#
# Usage: python scripts/exp_prompts.py
# Requires: OPENAI_API_KEY in .env, vector index built (scripts/build_index.py)

import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import EXPERIMENTS, INDEX_PATH, CHUNKS_PATH
from src.vector_store import VectorStore
from src.retriever import Retriever
from src.prompt_engine import select_context, build_prompt, PROMPT_VERSIONS
from src.llm_client import call_llm
from src.logger import banner, log_stage

OUT_FILE = EXPERIMENTS / "prompt_iterations.md"

# One representative query per domain
TEST_QUERIES = [
    "How many votes did the NPP candidate get in Assin North constituency?",
    "What is the 2025 budget allocation for the health sector?",
]


def main() -> None:
    banner("PROMPT ENGINEERING EXPERIMENT")

    if not INDEX_PATH.exists():
        print("ERROR: Index not found. Run `python scripts/build_index.py` first.")
        sys.exit(1)

    print("Loading vector store…")
    store = VectorStore.load()
    retriever = Retriever(store)

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report = [
        "# Prompt Engineering Experiment Log",
        f"*Run at: {ts}*  ",
        f"*Student: Yaw Acheampong Ahenkora Gyamera | Index: 10022200141*",
        "",
        "## Experiment Design",
        "",
        "Same query → 3 prompt versions → compare LLM outputs",
        "",
        "| Version | Description |",
        "|---------|-------------|",
        "| v1_naive | No instructions, just context + question |",
        "| v2_grounded | Grounding rule + refusal clause |",
        "| v3_grounded_cite | Grounding + citations + hallucination controls (CHOSEN) |",
        "",
        "---",
        "",
    ]

    for query in TEST_QUERIES:
        print(f"\nQuery: {query}")

        # Retrieve (hybrid, top_k=5)
        results = retriever.retrieve(query, top_k=5, min_score=0.0)
        selected = select_context(results)

        report += [
            f"## Query: `{query}`",
            "",
            f"**Retrieved chunks**: {len(selected)}  ",
            f"**Top score**: {f'{selected[0][1]:.4f}' if selected else 'N/A'}",
            "",
        ]

        for version_name, _ in PROMPT_VERSIONS.items():
            print(f"  Prompt version: {version_name} …", end="", flush=True)
            prompt = build_prompt(query, selected, version=version_name)
            answer = call_llm(prompt)
            print(" done")

            log_stage("PROMPT_EXPERIMENT", {
                "query": query[:80],
                "version": version_name,
                "answer_preview": answer[:200],
            })

            report += [
                f"### Version: `{version_name}`",
                "",
                "**Prompt sent to LLM** (first 400 chars):",
                "",
                "```",
                prompt[:400] + ("…" if len(prompt) > 400 else ""),
                "```",
                "",
                "**LLM Response**:",
                "",
                answer,
                "",
                "---",
                "",
            ]

        report += [
            "### Manual Analysis (fill in after reviewing above)",
            "",
            "- v1 issues:",
            "- v2 improvements:",
            "- v3 improvements:",
            "- Chosen version justification:",
            "",
            "===",
            "",
        ]

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text("\n".join(report), encoding="utf-8")
    print(f"\nReport written to {OUT_FILE}")


if __name__ == "__main__":
    main()
