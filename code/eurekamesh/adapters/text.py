from __future__ import annotations
from typing import Any, List, Dict
from .base import DomainAdapter

def _lower(s: str) -> str:
    return s.strip().lower()

def _jaccard(a: str, b: str) -> float:
    sa, sb = set(a.split()), set(b.split())
    if not sa and not sb:
        return 1.0
    inter = len(sa & sb)
    union = len(sa | sb) or 1
    return inter / union

class TextAdapter(DomainAdapter):
    def canonicalize(self, item: Any) -> str:
        return _lower(str(item))

    def signature(self, item: Any) -> Any:
        return self.canonicalize(item)

    def similarity(self, sig_a: Any, sig_b: Any) -> float:
        return _jaccard(str(sig_a), str(sig_b))

    def validate(self, item: Any) -> Dict[str, Any] | bool:
        s = str(item).strip()
        return {"valid": bool(s)}

    def render_memory(self, exemplars: List[str], k: int = 12) -> str:
        take = exemplars[-k:]
        if not take:
            return ""
        lines = "\n".join(f"- {t}" for t in take)
        return f"Recent unique items (do not repeat):\n{lines}\n"


