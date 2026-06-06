from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AgentResponse:
    """Assistant reply with optional chain-of-thought shown in the UI."""

    answer: str
    thinking: str | None = None

    @classmethod
    def from_text(cls, text: str) -> "AgentResponse":
        return cls(answer=text)
