# CS4241 Manual RAG Chatbot — Chunk dataclass
# Student: Yaw Acheampong Ahenkora Gyamera | Index: 10022200141

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class Chunk:
    """A single retrievable unit of text with its provenance metadata."""

    id: str                        # unique identifier, e.g. "csv_042" or "pdf_0137"
    text: str                      # the actual text content
    source: str                    # "election_csv" or "budget_pdf"
    source_intent: str             # "election_numeric" | "budget_policy" | "general"
    metadata: dict[str, Any] = field(default_factory=dict)
    # metadata keys (CSV): constituency, region, party, candidate, votes, position
    # metadata keys (PDF): page, section_title, strategy

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Chunk":
        return Chunk(
            id=d["id"],
            text=d["text"],
            source=d["source"],
            source_intent=d["source_intent"],
            metadata=d.get("metadata", {}),
        )

    def truncated(self, max_chars: int) -> "Chunk":
        """Return a copy with text truncated to max_chars."""
        if len(self.text) <= max_chars:
            return self
        return Chunk(
            id=self.id,
            text=self.text[:max_chars] + "…",
            source=self.source,
            source_intent=self.source_intent,
            metadata=self.metadata,
        )
