from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, List, Tuple, Dict


class DomainAdapter(ABC):
    """Abstract interface to adapt CCAD to new domains."""

    @abstractmethod
    def canonicalize(self, item: Any) -> str:
        """Return canonical string representation used for exact dedup."""

    @abstractmethod
    def signature(self, item: Any) -> Any:
        """Return a signature/embedding/fingerprint for fuzzy similarity."""

    @abstractmethod
    def similarity(self, sig_a: Any, sig_b: Any) -> float:
        """Return similarity in [0,1]; 1 means identical."""

    @abstractmethod
    def validate(self, item: Any) -> Dict[str, Any] | bool:
        """Return validity info (dict or bool)."""

    @abstractmethod
    def render_memory(self, exemplars: List[str], k: int = 12) -> str:
        """Turn canonical exemplars into a compact, human‑readable prompt section."""

    def parse_model_output(self, text: str) -> List[str]:
        """Optional: parse raw LLM output into item list (defaults: one per line)."""
        return [s.strip() for s in text.splitlines() if s.strip()]


