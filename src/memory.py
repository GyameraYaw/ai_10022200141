# CS4241 Manual RAG Chatbot — Novel Feature: Domain Scoring + Session Memory (PART G)
# Student: Yaw Acheampong Ahenkora Gyamera | Index: 10022200141
#
# INNOVATION: Domain-specific scoring function + session memory
#
# 1. Intent classification: labels each query as election_numeric, budget_policy, or general
# 2. Domain boosts: chunks whose source_intent matches the query intent receive +β score
#    Additional signals: numeric-density bonus for election queries, monetary-pattern
#    bonus for budget queries.
# 3. Session memory: ring buffer (N=5 turns) for anaphora resolution and chunk dedup

import json
import re
import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chunk import Chunk
from src.config import DOMAIN_BOOST_BETA, MEMORY_MAX_TURNS, SESSION_PATH
from src.logger import log_stage

# ── Intent classifier ─────────────────────────────────────────────────────────

_ELECTION_TOKENS = re.compile(
    r"\b(vote|votes|won|winner|constituency|region|npp|ndc|pnc|cpp|candidate|"
    r"seat|seats|majority|runoff|presidential|parliamentary|polling|election|"
    r"result|results|tally|ballot|party|parties|mp|lawmaker|district)\b",
    re.IGNORECASE,
)
_BUDGET_TOKENS = re.compile(
    r"\b(budget|allocation|gdp|cedis|expenditure|revenue|ministry|minister|"
    r"tax|taxes|fiscal|deficit|surplus|debt|spending|economic|policy|fund|"
    r"funds|ghc|gh¢|abfa|getfund|nhil|levy|vat|sector|programme|program|"
    r"infrastructure|education|health|agriculture|2025)\b",
    re.IGNORECASE,
)
_ANAPHORA_STARTERS = re.compile(
    r"^(and|what about|how about|why|also|similarly|what of|tell me more|"
    r"what were|can you|could you|please|furthermore|additionally)\b",
    re.IGNORECASE,
)


def classify_intent(query: str) -> str:
    """
    Classify a query into one of three domain intents.

    Returns: 'election_numeric' | 'budget_policy' | 'general'
    """
    election_hits = len(_ELECTION_TOKENS.findall(query))
    budget_hits = len(_BUDGET_TOKENS.findall(query))

    if election_hits == 0 and budget_hits == 0:
        intent = "general"
    elif election_hits >= budget_hits:
        intent = "election_numeric"
    else:
        intent = "budget_policy"

    log_stage("INTENT_CLASSIFY", {
        "query": query[:80],
        "election_hits": election_hits,
        "budget_hits": budget_hits,
        "intent": intent,
    })
    return intent


# ── Domain-specific scoring signals ──────────────────────────────────────────

_NUMERIC_RE = re.compile(r"\b\d[\d,]*\b")
_MONETARY_RE = re.compile(r"(GH[Cc¢]|cedis?|GHS|\d[\d,]*\s*(billion|million|thousand)|\d+\.?\d*%)", re.IGNORECASE)


def _numeric_density(text: str) -> float:
    """Proportion of whitespace-separated tokens that are numeric."""
    tokens = text.split()
    if not tokens:
        return 0.0
    numeric = sum(1 for t in tokens if _NUMERIC_RE.match(t.replace(",", "")))
    return numeric / len(tokens)


def _monetary_density(text: str) -> float:
    """Proportion of sentences containing a monetary pattern."""
    sentences = re.split(r"[.!?]", text)
    if not sentences:
        return 0.0
    hits = sum(1 for s in sentences if _MONETARY_RE.search(s))
    return hits / len(sentences)


def compute_domain_boosts(
    chunks: list[Chunk],
    intent: str,
    beta: float = DOMAIN_BOOST_BETA,
) -> dict[str, float]:
    """
    Return a dict mapping chunk_id → boost_value based on domain intent.

    The boost has two components:
    1. Intent-match bonus: +β if chunk.source_intent matches query intent
    2. Signal bonus: numeric-density bonus (election) or monetary-density bonus (budget)
    """
    boosts: dict[str, float] = {}

    for chunk in chunks:
        boost = 0.0

        # Component 1: intent match
        if chunk.source_intent == intent:
            boost += beta

        # Component 2: content signal
        if intent == "election_numeric":
            boost += beta * 0.5 * _numeric_density(chunk.text)
        elif intent == "budget_policy":
            boost += beta * 0.5 * _monetary_density(chunk.text)

        if boost > 0:
            boosts[chunk.id] = boost

    log_stage("DOMAIN_BOOSTS", {
        "intent": intent,
        "beta": beta,
        "boosted_chunks": len(boosts),
        "max_boost": max(boosts.values()) if boosts else 0,
    })

    return boosts


# ── Session memory ────────────────────────────────────────────────────────────

class SessionMemory:
    """
    Ring buffer of the last MEMORY_MAX_TURNS interactions.

    Each turn stores:
        query          : str
        retrieved_ids  : list of chunk ids retrieved
        answer         : str (LLM response)
    """

    def __init__(self, max_turns: int = MEMORY_MAX_TURNS) -> None:
        self.max_turns = max_turns
        self._turns: deque[dict] = deque(maxlen=max_turns)

    def push(self, query: str, retrieved_ids: list[str], answer: str) -> None:
        self._turns.append({
            "query": query,
            "retrieved_ids": retrieved_ids,
            "answer": answer,
        })
        log_stage("MEMORY_PUSH", {"query": query[:80], "n_ids": len(retrieved_ids)})

    def resolve_anaphora(self, query: str) -> str:
        """
        If the query starts with an anaphoric expression, prepend the
        previous query to give the retriever context.
        """
        if not self._turns:
            return query
        if _ANAPHORA_STARTERS.match(query.strip()):
            prev_query = self._turns[-1]["query"]
            rewritten = f"{prev_query} — {query}"
            log_stage("MEMORY_ANAPHORA", {"original": query, "rewritten": rewritten})
            return rewritten
        return query

    def recently_seen_ids(self) -> set[str]:
        """IDs of chunks retrieved in the previous turn (for dedup penalty)."""
        if not self._turns:
            return set()
        return set(self._turns[-1]["retrieved_ids"])

    def recency_penalties(self, chunks: list[Chunk], penalty: float = 0.05) -> dict[str, float]:
        """
        Return a dict of chunk_id → negative boost for chunks seen in the last turn.
        Used to encourage diversity in follow-up queries.
        """
        seen = self.recently_seen_ids()
        return {c.id: -penalty for c in chunks if c.id in seen}

    def save(self, path: Path = SESSION_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(list(self._turns), f, ensure_ascii=False, indent=2)

    def load(self, path: Path = SESSION_PATH) -> None:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                turns = json.load(f)
            self._turns = deque(turns, maxlen=self.max_turns)

    def clear(self) -> None:
        self._turns.clear()

    def as_list(self) -> list[dict]:
        return list(self._turns)
