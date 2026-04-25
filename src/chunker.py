# CS4241 Manual RAG Chatbot — Chunking Strategies (PART A)
# Student: Yaw Acheampong Ahenkora Gyamera | Index: 10022200141
#
# Three chunking strategies are implemented and compared:
#   1. fixed_char   — fixed character window with overlap (baseline)
#   2. sentence     — sentence-aware sliding window (chosen strategy)
#   3. structure    — structure-aware (row-as-chunk for CSV; section for PDF)
#
# The chosen strategy is sentence-aware because:
#   - Preserves semantic unit boundaries (sentences are natural thought units)
#   - Avoids mid-sentence cuts that confuse the embedding model
#   - Overlap ensures boundary context is not lost between chunks
#   - Outperforms fixed-char on MRR@5 and Hit@3 (see experiments/chunking_results.md)

import re
import sys
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chunk import Chunk
from src.config import (
    CHUNK_FIXED_SIZE, CHUNK_FIXED_OVERLAP,
    CHUNK_SENTENCE_TOKENS, CHUNK_SENTENCE_OVERLAP,
)
from src.logger import log_stage

ChunkStrategy = Literal["fixed_char", "sentence", "structure"]

# ── Helpers ───────────────────────────────────────────────────────────────────

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_SECTION_HEAD = re.compile(r"\n(\d{1,2}\.\d{0,2}\s+[A-Z][^\n]{5,})")


def _approx_tokens(text: str) -> int:
    """Rough token count: ~4 chars per token."""
    return max(1, len(text) // 4)


def _csv_row_to_text(row: dict) -> str:
    """
    Render a Ghana election CSV row as a human-readable sentence for embedding.

    Actual CSV columns (after normalisation):
        year, old_region, new_region, code, candidate, party, votes, votes_(%)
    Each row is one candidate's regional vote tally in a presidential election year.
    """
    year = row.get("year", "")
    old_region = row.get("old_region", "")
    new_region = row.get("new_region", old_region)  # prefer new_region if present
    candidate = row.get("candidate", "")
    party = row.get("party", "")
    votes = row.get("votes", "")
    pct = row.get("votes_(%)", row.get("votes(%)", ""))

    region_label = new_region if new_region and new_region != "nan" else old_region

    parts = []
    if year:
        parts.append(f"In the {year} Ghana presidential election,")
    if candidate:
        parts.append(f"candidate {candidate} ({party})")
    if region_label:
        parts.append(f"received {votes} votes ({pct}) in {region_label}.")
    elif votes:
        parts.append(f"received {votes} votes ({pct}).")

    if not parts:
        parts = [f"{k}: {v}" for k, v in row.items() if str(v) not in ("", "nan", "None")]

    return " ".join(parts)


# ── Strategy 1: Fixed character window ────────────────────────────────────────

def chunk_fixed_char(
    text: str,
    source: str,
    source_intent: str,
    base_id: str,
    size: int = CHUNK_FIXED_SIZE,
    overlap: int = CHUNK_FIXED_OVERLAP,
    metadata: dict | None = None,
) -> list[Chunk]:
    """Split text into fixed-size character windows with overlap."""
    chunks: list[Chunk] = []
    start = 0
    idx = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(Chunk(
                id=f"{base_id}_fc{idx:04d}",
                text=chunk_text,
                source=source,
                source_intent=source_intent,
                metadata={**(metadata or {}), "strategy": "fixed_char", "char_start": start},
            ))
            idx += 1
        start += size - overlap
    return chunks


# ── Strategy 2: Sentence-aware sliding window (CHOSEN) ────────────────────────

def chunk_sentence(
    text: str,
    source: str,
    source_intent: str,
    base_id: str,
    token_budget: int = CHUNK_SENTENCE_TOKENS,
    token_overlap: int = CHUNK_SENTENCE_OVERLAP,
    metadata: dict | None = None,
) -> list[Chunk]:
    """
    Split text into sentence-aware chunks.

    Groups consecutive sentences until the token budget is exceeded,
    then starts a new chunk with the last `token_overlap` tokens of
    the previous chunk as context prefix.
    """
    sentences = _SENT_SPLIT.split(text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks: list[Chunk] = []
    current: list[str] = []
    current_tokens = 0
    idx = 0

    for sent in sentences:
        sent_tokens = _approx_tokens(sent)

        if current_tokens + sent_tokens > token_budget and current:
            # Emit chunk
            chunk_text = " ".join(current)
            chunks.append(Chunk(
                id=f"{base_id}_sw{idx:04d}",
                text=chunk_text,
                source=source,
                source_intent=source_intent,
                metadata={**(metadata or {}), "strategy": "sentence", "sentences": len(current)},
            ))
            idx += 1

            # Build overlap: keep tail sentences that fit within token_overlap
            overlap_sents: list[str] = []
            overlap_tok = 0
            for s in reversed(current):
                t = _approx_tokens(s)
                if overlap_tok + t > token_overlap:
                    break
                overlap_sents.insert(0, s)
                overlap_tok += t

            current = overlap_sents
            current_tokens = overlap_tok

        current.append(sent)
        current_tokens += sent_tokens

    # Emit the final chunk
    if current:
        chunks.append(Chunk(
            id=f"{base_id}_sw{idx:04d}",
            text=" ".join(current),
            source=source,
            source_intent=source_intent,
            metadata={**(metadata or {}), "strategy": "sentence", "sentences": len(current)},
        ))

    return chunks


# ── Strategy 3: Structure-aware ───────────────────────────────────────────────

def chunk_structure_pdf(
    pages: list[dict],
    source: str,
    source_intent: str,
    base_id: str,
) -> list[Chunk]:
    """
    For PDF: split on heading boundaries (numbered sections like "3.2 Education").
    Falls back to sentence chunking within each section.
    """
    # Concatenate all page text with page markers
    full_text = "\n\n".join(f"[Page {p['page']}]\n{p['text']}" for p in pages)

    # Split on section headings
    parts = _SECTION_HEAD.split(full_text)
    # parts alternates: [pre_text, heading1, body1, heading2, body2, ...]

    chunks: list[Chunk] = []
    section_title = "Introduction"
    idx = 0

    for i, part in enumerate(parts):
        if i % 2 == 1:
            # This is a heading
            section_title = part.strip()
            continue

        body = part.strip()
        if not body:
            continue

        # Sentence-chunk within each section
        sub_chunks = chunk_sentence(
            text=body,
            source=source,
            source_intent=source_intent,
            base_id=f"{base_id}_sec{idx:03d}",
            metadata={"strategy": "structure", "section_title": section_title},
        )
        chunks.extend(sub_chunks)
        idx += 1

    return chunks


def chunk_structure_csv(
    rows: list[dict],
    source: str,
    source_intent: str,
    base_id: str,
) -> list[Chunk]:
    """
    For CSV: each row becomes its own chunk (natural semantic unit).
    Groups of related rows (same constituency) could be merged but
    row-level granularity gives the best precision for vote lookups.
    """
    chunks: list[Chunk] = []
    for i, row in enumerate(rows):
        text = _csv_row_to_text(row)
        chunks.append(Chunk(
            id=f"{base_id}_row{i:05d}",
            text=text,
            source=source,
            source_intent=source_intent,
            metadata={**row, "strategy": "structure"},
        ))
    return chunks


# ── Public API: chunk all data with a given strategy ─────────────────────────

def chunk_all(
    csv_rows: list[dict],
    pdf_pages: list[dict],
    strategy: ChunkStrategy = "sentence",
) -> list[Chunk]:
    """
    Produce all Chunk objects for both data sources using the given strategy.

    Parameters
    ----------
    csv_rows  : cleaned row dicts from data_loader.load_csv()
    pdf_pages : cleaned page dicts from data_loader.load_pdf()
    strategy  : "fixed_char" | "sentence" | "structure"

    Returns a flat list of Chunk objects (CSV first, then PDF).
    """
    chunks: list[Chunk] = []

    # ── CSV ──────────────────────────────────────────────────────────────────
    if strategy == "structure":
        csv_chunks = chunk_structure_csv(csv_rows, "election_csv", "election_numeric", "csv")
    else:
        for i, row in enumerate(csv_rows):
            text = _csv_row_to_text(row)
            meta = {**row, "strategy": strategy}
            if strategy == "fixed_char":
                csv_chunks_i = chunk_fixed_char(text, "election_csv", "election_numeric", f"csv_r{i:05d}", metadata=meta)
            else:  # sentence
                csv_chunks_i = chunk_sentence(text, "election_csv", "election_numeric", f"csv_r{i:05d}", metadata=meta)
            chunks.extend(csv_chunks_i)
        csv_chunks = []  # already added

    chunks.extend(csv_chunks)

    # ── PDF ──────────────────────────────────────────────────────────────────
    pdf_full_text = "\n\n".join(p["text"] for p in pdf_pages)

    if strategy == "structure":
        pdf_chunks = chunk_structure_pdf(pdf_pages, "budget_pdf", "budget_policy", "pdf")
    elif strategy == "fixed_char":
        pdf_chunks = chunk_fixed_char(pdf_full_text, "budget_pdf", "budget_policy", "pdf")
    else:  # sentence
        pdf_chunks = chunk_sentence(pdf_full_text, "budget_pdf", "budget_policy", "pdf")

    chunks.extend(pdf_chunks)

    log_stage("CHUNKING_DONE", {
        "strategy": strategy,
        "total_chunks": len(chunks),
        "csv_source_chunks": sum(1 for c in chunks if c.source == "election_csv"),
        "pdf_source_chunks": sum(1 for c in chunks if c.source == "budget_pdf"),
    })

    return chunks
