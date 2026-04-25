# CS4241 Manual RAG Chatbot — Data Loader & Cleaner (PART A)
# Student: Yaw Acheampong Ahenkora Gyamera | Index: 10022200141
#
# Loads and cleans:
#   1. Ghana_Election_Result.csv  → list of dicts (one per row)
#   2. 2025-Budget-Statement.pdf  → list of dicts (one per page, with page text)

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import pypdf

from src.config import CSV_PATH, PDF_PATH
from src.logger import log_stage


# ── CSV loader ────────────────────────────────────────────────────────────────

def load_csv(path: Path = CSV_PATH) -> list[dict]:
    """
    Load and clean the Ghana election results CSV.

    Cleaning steps:
      1. Normalize column names (strip, lower, replace spaces with underscores).
      2. Strip leading/trailing whitespace from string columns.
      3. Drop rows where constituency or party is null/empty.
      4. Coerce vote-count columns to int (fill NaN with 0).
      5. Drop exact duplicate rows.

    Returns a list of cleaned row dicts.
    """
    df = pd.read_csv(path, encoding="utf-8", dtype=str)
    log_stage("CSV_LOAD_RAW", {"rows": len(df), "cols": list(df.columns)})

    # 1. Normalize column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # 2. Strip strings
    str_cols = df.select_dtypes(include="object").columns
    df[str_cols] = df[str_cols].apply(lambda s: s.str.strip())

    # 3. Drop rows missing key fields
    key_cols = [c for c in df.columns if any(k in c for k in ("constituency", "party", "candidate"))]
    before = len(df)
    df.dropna(subset=key_cols, inplace=True)
    df = df[df[key_cols].apply(lambda r: r.str.len().gt(0).all(), axis=1)]
    dropped_null = before - len(df)

    # 4. Coerce vote columns to int
    vote_cols = [c for c in df.columns if "vote" in c or "result" in c or "seat" in c]
    for col in vote_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # 5. Drop duplicates
    before2 = len(df)
    df.drop_duplicates(inplace=True)
    dropped_dup = before2 - len(df)

    records = df.to_dict(orient="records")
    log_stage("CSV_LOAD_CLEAN", {
        "rows_kept": len(records),
        "dropped_null": dropped_null,
        "dropped_dup": dropped_dup,
        "columns": list(df.columns),
    })
    return records


# ── PDF loader ────────────────────────────────────────────────────────────────

_JUNK_PATTERNS = [
    re.compile(r"LG\s*-\s*Public", re.IGNORECASE),
    re.compile(r"^\s*\d+\s*$", re.MULTILINE),          # bare page numbers
    re.compile(r"Page\s+\d+\s+of\s+\d+", re.IGNORECASE),
    re.compile(r"2025 Budget Statement.*?Economic Policy", re.IGNORECASE),
]

_HYPHEN_BREAK = re.compile(r"-\n(\w)")


def _clean_pdf_page(text: str) -> str:
    """Strip headers/footers/artifacts from a single PDF page's text."""
    # Reattach hyphenated line-breaks ("eco-\nnomy" → "economy")
    text = _HYPHEN_BREAK.sub(r"\1", text)

    for pat in _JUNK_PATTERNS:
        text = pat.sub("", text)

    # Collapse runs of blank lines to a single blank line
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse runs of spaces (but not newlines)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def load_pdf(path: Path = PDF_PATH) -> list[dict]:
    """
    Load and clean the 2025 Budget Statement PDF.

    Returns a list of page dicts:
        {page: int, text: str}

    Pages with < 50 chars after cleaning are discarded (TOC/blank pages).
    """
    reader = pypdf.PdfReader(str(path))
    total = len(reader.pages)
    log_stage("PDF_LOAD_START", {"path": str(path), "total_pages": total})

    pages = []
    for i, page in enumerate(reader.pages):
        raw = page.extract_text() or ""
        cleaned = _clean_pdf_page(raw)
        if len(cleaned) >= 50:
            pages.append({"page": i + 1, "text": cleaned})

    log_stage("PDF_LOAD_CLEAN", {
        "pages_kept": len(pages),
        "pages_discarded": total - len(pages),
    })
    return pages
