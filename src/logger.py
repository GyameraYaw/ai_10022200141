# CS4241 Manual RAG Chatbot — Structured Stage Logger
# Student: Yaw Acheampong Ahenkora Gyamera | Index: 10022200141

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


_LOG_SINKS: list[Path] = []   # file paths to additionally write to


def add_file_sink(path: Path) -> None:
    """Register a file that should receive all subsequent log_stage calls."""
    path.parent.mkdir(parents=True, exist_ok=True)
    _LOG_SINKS.append(path)


def clear_sinks() -> None:
    _LOG_SINKS.clear()


def log_stage(stage: str, payload: Any, *, to_stdout: bool = True) -> str:
    """
    Emit a structured log entry for one pipeline stage.

    Parameters
    ----------
    stage   : short label, e.g. "RETRIEVAL", "PROMPT", "LLM_RESPONSE"
    payload : anything JSON-serialisable (dict, list, str, …)
    """
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = {
        "ts": ts,
        "stage": stage,
        "data": payload,
    }
    line = json.dumps(entry, ensure_ascii=False, default=str)

    if to_stdout:
        print(f"[{ts}] [{stage}] {_preview(payload)}", file=sys.stderr)

    for sink in _LOG_SINKS:
        with open(sink, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    return line


def _preview(payload: Any, max_len: int = 120) -> str:
    """One-line human-readable preview for stdout."""
    raw = json.dumps(payload, ensure_ascii=False, default=str)
    return raw[:max_len] + ("…" if len(raw) > max_len else "")


def banner(title: str) -> None:
    """Print a section header to stderr."""
    sep = "=" * 60
    print(f"\n{sep}\n  {title}\n{sep}", file=sys.stderr)
